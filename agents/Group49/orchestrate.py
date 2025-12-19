import os
import shutil
import subprocess
import glob
import re
import time
import sys
import random

# --- CONFIGURATION ---
ITERATION_START = 1
ITERATIONS = 100

MODEL_BLOCKS = 10
MODEL_FILTERS = 256

# Data Generation
GAMES_PER_ITER = 384
MCTS_SIMS_GEN = 192
GOLD_SAMPLE_SIZE = 2048
# Training
TRAIN_EPOCHS = 3
WINDOW_SIZE = 15

# Evaluation
EVAL_GAMES = 64
EVAL_SIMS = 192
WIN_THRESHOLD = 0.52

# --- PATHS SETUP ---
# 1. Define the relative path from where you RUN the script (3 levels up)
HEX_NN_RELATIVE = "../../../hex_nn"

# 2. Convert it to a full path (e.g., /home/skynet/git/hex_nn)
# This removes all ambiguity for the C++ engine.
HEX_NN_ABS = os.path.abspath(HEX_NN_RELATIVE)

PATHS = {
    # Executables
    "self_play": "./cmake-build-release/Group49SelfPlay",
    "arbiter":   "./cmake-build-release/Arbiter",
    "engine":    "./cmake-build-release/Group49",

    # Python Scripts (Use HEX_NN_ABS)
    "train":     os.path.join(HEX_NN_ABS, "train.py"),
    "export":    os.path.join(HEX_NN_ABS, "export_onnx.py"),

    # Directories (Use HEX_NN_ABS)
    "model_dir": os.path.join(HEX_NN_ABS, "models"),
    "data_dir":  os.path.join(HEX_NN_ABS, "data"),
    "gold_dir":  os.path.join(HEX_NN_ABS, "data/goldstandard"),

    # Files (Use HEX_NN_ABS)
    "initial_checkpoint": os.path.join(HEX_NN_ABS, "checkpoints/run_001/best.pt"),
    "champion_pt":        os.path.join(HEX_NN_ABS, "models/champion.pt"),
    "best_onnx":          os.path.join(HEX_NN_ABS, "models/best.onnx"),

    "candidate_dir":      os.path.join(HEX_NN_ABS, "models/candidate"),
    "candidate_pt":       os.path.join(HEX_NN_ABS, "models/candidate/best.pt"),
    "candidate_onnx":     os.path.join(HEX_NN_ABS, "models/candidate.onnx"),
}

def run_command(cmd, log_file=None):
    """Helper to run shell commands and log output."""
    cmd_str = " ".join(cmd)
    print(f"   [CMD] {cmd_str}")

    try:
        if log_file:
            with open(log_file, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   [ERROR] Command failed with exit code {e.returncode}")
        raise e

def initialize_workspace():
    """Sets up directories and ensures the first Champion exists."""
    os.makedirs("logs", exist_ok=True)
    # Create the destination dirs in hex_nn if they don't exist
    os.makedirs(f"{PATHS['model_dir']}/archive", exist_ok=True)
    os.makedirs(PATHS['data_dir'], exist_ok=True)

    # 1. Check if we have a champion.pt to train from
    if not os.path.exists(PATHS["champion_pt"]):
        print(f">>> Initializing Champion from {PATHS['initial_checkpoint']}...")
        if not os.path.exists(PATHS["initial_checkpoint"]):
            # Print current working directory to help debug if it fails again
            print(f"   [Debug] CWD: {os.getcwd()}")
            raise FileNotFoundError(f"Cannot find initial checkpoint at {PATHS['initial_checkpoint']}")
        shutil.copy(PATHS["initial_checkpoint"], PATHS["champion_pt"])

    # 2. Check if we have a best.onnx to generate data with
    if not os.path.exists(PATHS["best_onnx"]):
        print(f">>> Warning: {PATHS['best_onnx']} not found. Attempting to export from champion...")
        export_model(PATHS["champion_pt"], PATHS["best_onnx"])

def cleanup_iteration_data(iter_num):
    """Deletes data files for the current iteration to prevent stale/corrupt reads."""
    target_file = f"{PATHS['data_dir']}/gen_{iter_num:03d}.jsonl"
    if os.path.exists(target_file):
        print(f"   [Cleanup] Removing stale file: {target_file}")
        os.remove(target_file)

def generate_data(iter_num):
    print(f"\n>>> [Iter {iter_num}] GENERATING DATA")
    cleanup_iteration_data(iter_num)

    output_file = f"{PATHS['data_dir']}/gen_{iter_num:03d}.jsonl"

    cmd = [
        PATHS["self_play"],
        "agent",
        str(GAMES_PER_ITER),
        str(MCTS_SIMS_GEN),
        output_file,
        "0",
        PATHS["best_onnx"]
    ]
    run_command(cmd, log_file=f"logs/gen_{iter_num}.log")

def sample_gold_standard(output_path, num_samples):
    """
    Scans PATHS['gold_dir'] for all .jsonl files, filters out junk,
    and randomly samples 'num_samples' lines into 'output_path'.
    """
    print(f"   [Data] Sampling {num_samples} games from Gold Standard corpus...")

    # 1. Find all valid jsonl files
    all_files = glob.glob(os.path.join(PATHS['gold_dir'], "*.jsonl"))

    # Filter out Zone.Identifier or other metadata junk
    valid_files = [f for f in all_files if "Zone.Identifier" not in f and os.path.getsize(f) > 0]

    if not valid_files:
        print("   [Warning] No Gold Standard files found! Training only on self-play.")
        return

    # 2. Gather lines
    all_lines = []
    for fpath in valid_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # strict check to ensure it looks like json
                valid_lines = [l for l in lines if l.strip().startswith("{")]
                all_lines.extend(valid_lines)
        except Exception as e:
            print(f"   [Warning] Could not read {os.path.basename(fpath)}: {e}")

    if not all_lines:
        print("   [Warning] Gold Standard files were empty or invalid.")
        return

    # 3. Sample
    sample_count = min(len(all_lines), num_samples)
    selected_lines = random.sample(all_lines, sample_count)

    # 4. Write to temp file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)

    print(f"   [Data] Successfully mixed in {len(selected_lines)} expert games.")

def train_student(iter_num):
    print(f">>> [Iter {iter_num}] TRAINING STUDENT (Fine-Tuning)")

    # 1. Identify Self-Play History (Last N valid files)
    all_files = sorted(glob.glob(f"{PATHS['data_dir']}/gen_*.jsonl"))
    valid_files = [f for f in all_files if os.path.getsize(f) > 0]
    replay_files = valid_files[-WINDOW_SIZE:]

    if not replay_files:
        raise Exception("No valid self-play data found!")

    # --- DYNAMIC SAMPLING LOGIC START ---
    # Calculate how many self-play games we actually have
    # (Assuming roughly GAMES_PER_ITER games per file)
    current_self_play_count = len(replay_files) * GAMES_PER_ITER

    # Target a 1:1 ratio, but cap it at our maximum desired gold size (e.g., 3000)
    MAX_GOLD_SIZE = 6000
    dynamic_gold_count = int(min(current_self_play_count * 1, MAX_GOLD_SIZE))

    print(f"   [Data Balance] Self-Play: ~{current_self_play_count} | Gold Standard: {dynamic_gold_count}")
    # --- DYNAMIC SAMPLING LOGIC END ---

    # 2. Generate Gold Standard Subset
    gold_subset_file = os.path.join(PATHS['data_dir'], "temp_gold_subset.jsonl")

    # Pass the calculated number, not the global constant
    sample_gold_standard(gold_subset_file, dynamic_gold_count)

    # 3. Combine Lists (Replay + Gold Subset)
    # Only add gold file if it was actually created and has content
    training_data_args = list(replay_files)
    if os.path.exists(gold_subset_file) and os.path.getsize(gold_subset_file) > 0:
        training_data_args.append(gold_subset_file)

    print(f"   Training sources: {len(replay_files)} self-play files + Gold Subset")

    # 4. Run train.py
    cmd = [
              "python3", PATHS["train"],
              "--data"] + training_data_args + [
              "--epochs", str(TRAIN_EPOCHS),
              "--resume", PATHS["champion_pt"],
              "--run-name", "candidate",
              "--output-dir", PATHS["model_dir"],
              "--lr", "0.00005",
            "--weight-decay", "0.01",
            "--num-blocks", str(MODEL_BLOCKS),
            "--channels", str(MODEL_FILTERS)
          ]

    run_command(cmd)

def export_model(pt_path, onnx_path):
    cmd = [
        "python3", PATHS["export"],
        pt_path,
        onnx_path,
        "--blocks", str(MODEL_BLOCKS),
        "--filters", str(MODEL_FILTERS)
    ]
    run_command(cmd)

def get_latest_candidate_pt():
    """
    Finds the best available candidate file.
    If 'best.pt' is missing (because loss didn't improve), returns the last epoch.
    """
    # 1. Try the explicit best.pt
    if os.path.exists(PATHS["candidate_pt"]):
        return PATHS["candidate_pt"]

    # 2. Fallback: Find the latest epoch_X.pt
    pattern = f"{PATHS['candidate_dir']}/epoch_*.pt"
    epoch_files = glob.glob(pattern)

    if not epoch_files:
        raise FileNotFoundError(f"No candidate models found in {PATHS['candidate_dir']}")

    # Sort by epoch number (e.g., epoch_2.pt > epoch_1.pt)
    # Extract number from filename to sort correctly
    latest_file = max(epoch_files, key=lambda f: int(re.search(r"epoch_(\d+).pt", f).group(1)))

    print(f"   [!] 'best.pt' not found (loss didn't improve). Using latest epoch: {os.path.basename(latest_file)}")
    return latest_file

def evaluate_student(iter_num):
    print(f">>> [Iter {iter_num}] EVALUATING (Candidate vs Champion)")

    # --- CHANGED SECTION START ---
    # 1. Resolve which .pt file to use
    actual_candidate_pt = get_latest_candidate_pt()

    # 2. Export that specific file to ONNX
    # We overwrite the standard 'candidate_onnx' path so the rest of the script works as is
    export_model(actual_candidate_pt, PATHS["candidate_onnx"])

    # Update our PATHS reference for promotion later (CRITICAL)
    # If we promote, we want to copy the file we actually used, not the missing 'best.pt'
    PATHS["current_actual_candidate_pt"] = actual_candidate_pt
    # --- CHANGED SECTION END ---

    cmd_candidate = f"{PATHS['engine']} {PATHS['candidate_onnx']}"
    cmd_champion  = f"{PATHS['engine']} {PATHS['best_onnx']}"

    cmd = [
        PATHS["arbiter"],
        cmd_candidate,
        cmd_champion,
        str(EVAL_GAMES // 2),
        "16", # THREAD COUNT
        str(EVAL_SIMS)
    ]
    print(cmd)

    result = subprocess.run(cmd, capture_output=True, text=True)

    match = re.search(r"Agent A Wins: (\d+)", result.stdout)

    if not match:
        print("   [!] Error parsing Arbiter output. Dumping stdout:")
        print(result.stdout)
        return False

    wins = int(match.group(1))
    win_rate = wins / EVAL_GAMES
    print(f"   Results: Candidate won {wins}/{EVAL_GAMES} ({win_rate*100:.1f}%)")

    return win_rate >= WIN_THRESHOLD

def copy_onnx(src_path, dst_path):
    """
    Copies an ONNX file AND its .data file (if it exists).
    Also cleans up stale .data files at the destination.
    """
    # 1. Copy the main .onnx file
    shutil.copy(src_path, dst_path)

    # 2. Handle the .data file
    src_data = src_path + ".data"
    dst_data = dst_path + ".data"

    if os.path.exists(src_data):
        # If source has data, copy it
        print(f"   [IO] Copying external data: {os.path.basename(src_data)}")
        shutil.copy(src_data, dst_data)
    elif os.path.exists(dst_data):
        # If source has NO data, but destination DOES (from a previous big model),
        # delete the stale destination data to avoid 'file mismatch' errors.
        print(f"   [IO] Cleaning up stale data file: {os.path.basename(dst_data)}")
        os.remove(dst_data)

def promote_student(iter_num):
    print(f">>> PROMOTING CANDIDATE TO CHAMPION")

    archive_name = f"champion_iter_{iter_num-1}.pt"
    shutil.copy(PATHS["champion_pt"], f"{PATHS['model_dir']}/archive/{archive_name}")

    # 1. Promote Weights (PT)
    source_pt = PATHS.get("current_actual_candidate_pt", PATHS["candidate_pt"])
    shutil.copy(source_pt, PATHS["champion_pt"])

    # 2. Promote Engine (ONNX + Data) --- CHANGED ---
    copy_onnx(PATHS["candidate_onnx"], PATHS["best_onnx"])
    # -----------------------------------------------

    print(f"   Champion updated. Old champion archived to {archive_name}")

def main():
    global MCTS_SIMS_GEN, EVAL_SIMS
    try:
        initialize_workspace()
    except Exception as e:
        print(f"[FATAL] Could not initialize workspace: {e}")
        return

    for i in range(ITERATION_START, ITERATIONS + 1):
        start_time = time.time()
        print(f"\n{'='*60}\nSTARTING ITERATION {i}\n{'='*60}")

        try:
            MCTS_SIMS_GEN = 160 + (i * 32)
            EVAL_SIMS     = 160 + (i * 32)
            generate_data(i)
            train_student(i)

            if evaluate_student(i):
                promote_student(i)
            else:
                print(f">>> REJECTED. Keeping current Champion.")

        except KeyboardInterrupt:
            print("\n>>> Stopping requested by user.")
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] Iteration {i} failed: {e}")
            break

        print(f"--- Iteration {i} finished in {(time.time()-start_time)/60:.1f} mins ---")

if __name__ == "__main__":
    main()