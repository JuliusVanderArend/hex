import os
import shutil
import subprocess
import glob
import re
import time
import sys

# --- CONFIGURATION ---
ITERATION_START = 1
ITERATIONS = 100

# Data Generation
GAMES_PER_ITER = 400
MCTS_SIMS_GEN = 200

# Training
TRAIN_EPOCHS = 2
WINDOW_SIZE = 15

# Evaluation
EVAL_GAMES = 40
EVAL_SIMS = 400
WIN_THRESHOLD = 0.55

# --- PATHS SETUP ---
HEX_NN_RELATIVE = "../../../hex_nn"

HEX_NN_ABS = os.path.abspath(HEX_NN_RELATIVE)

PATHS = {
    # Executables
    "self_play": "./cmake-build-release/Group49SelfPlay",
    "arbiter":   "./cmake-build-release/Arbiter",
    "engine":    "./cmake-build-release/Group49",

    # Python Scripts
    "train":     os.path.join(HEX_NN_ABS, "train.py"),
    "export":    os.path.join(HEX_NN_ABS, "export_onnx.py"),

    # Directories
    "model_dir": os.path.join(HEX_NN_ABS, "models"),
    "data_dir":  os.path.join(HEX_NN_ABS, "data"),

    # Files
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
    os.makedirs(f"{PATHS['model_dir']}/archive", exist_ok=True)
    os.makedirs(PATHS['data_dir'], exist_ok=True)

    if not os.path.exists(PATHS["champion_pt"]):
        print(f">>> Initializing Champion from {PATHS['initial_checkpoint']}...")
        if not os.path.exists(PATHS["initial_checkpoint"]):
            print(f"   [Debug] CWD: {os.getcwd()}")
            raise FileNotFoundError(f"Cannot find initial checkpoint at {PATHS['initial_checkpoint']}")
        shutil.copy(PATHS["initial_checkpoint"], PATHS["champion_pt"])

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

def train_student(iter_num):
    print(f">>> [Iter {iter_num}] TRAINING STUDENT (Fine-Tuning)")

    all_files = sorted(glob.glob(f"{PATHS['data_dir']}/*.jsonl"))
    valid_files = [f for f in all_files if os.path.getsize(f) > 0]
    train_files = valid_files[-WINDOW_SIZE:]

    if not train_files:
        raise Exception("No valid training data found!")

    print(f"   Training on {len(train_files)} files.")

    cmd = [
              "python3", PATHS["train"],
              "--data"] + train_files + [
              "--epochs", str(TRAIN_EPOCHS),
              "--resume", PATHS["champion_pt"],
              "--run-name", "candidate",
              "--output-dir", PATHS["model_dir"],
              "--lr", "0.0001"
          ]

    run_command(cmd)

def export_model(pt_path, onnx_path):
    cmd = [
        "python3", PATHS["export"],
        pt_path,
        onnx_path
    ]
    run_command(cmd)

def get_latest_candidate_pt():
    """
    Finds the best available candidate file.
    If 'best.pt' is missing (because loss didn't improve), returns the last epoch.
    """
    if os.path.exists(PATHS["candidate_pt"]):
        return PATHS["candidate_pt"]

    pattern = f"{PATHS['candidate_dir']}/epoch_*.pt"
    epoch_files = glob.glob(pattern)

    if not epoch_files:
        raise FileNotFoundError(f"No candidate models found in {PATHS['candidate_dir']}")

    latest_file = max(epoch_files, key=lambda f: int(re.search(r"epoch_(\d+).pt", f).group(1)))

    print(f"   [!] 'best.pt' not found (loss didn't improve). Using latest epoch: {os.path.basename(latest_file)}")
    return latest_file

def evaluate_student(iter_num):
    print(f">>> [Iter {iter_num}] EVALUATING (Candidate vs Champion)")

    actual_candidate_pt = get_latest_candidate_pt()

    export_model(actual_candidate_pt, PATHS["candidate_onnx"])

    PATHS["current_actual_candidate_pt"] = actual_candidate_pt

    cmd_candidate = f"{PATHS['engine']} {PATHS['candidate_onnx']}"
    cmd_champion  = f"{PATHS['engine']} {PATHS['best_onnx']}"

    cmd = [
        PATHS["arbiter"],
        cmd_candidate,
        cmd_champion,
        str(EVAL_GAMES // 2),
        "6",
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

def promote_student(iter_num):
    print(f">>> PROMOTING CANDIDATE TO CHAMPION")

    archive_name = f"champion_iter_{iter_num-1}.pt"
    shutil.copy(PATHS["champion_pt"], f"{PATHS['model_dir']}/archive/{archive_name}")

    source_pt = PATHS.get("current_actual_candidate_pt", PATHS["candidate_pt"])
    shutil.copy(source_pt, PATHS["champion_pt"])

    shutil.copy(PATHS["candidate_onnx"], PATHS["best_onnx"])

    print(f"   Champion updated. Old champion archived to {archive_name}")

def main():
    try:
        initialize_workspace()
    except Exception as e:
        print(f"[FATAL] Could not initialize workspace: {e}")
        return

    for i in range(ITERATION_START, ITERATIONS + 1):
        start_time = time.time()
        print(f"\n{'='*60}\nSTARTING ITERATION {i}\n{'='*60}")

        try:
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