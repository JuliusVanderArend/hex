import os
import shutil
import subprocess
import glob
import re
import time

# --- CONFIGURATION ---
ITERATION_START = 1
ITERATIONS = 100

# Data Generation
GAMES_PER_ITER = 250   # Games to generate per loop
MCTS_SIMS_GEN = 200    # Fast simulations for generation

# Training
TRAIN_EPOCHS = 2
WINDOW_SIZE = 15       # Train on last 15 generations of data (Replay Buffer)

# Evaluation
EVAL_GAMES = 40        # Tournament size (Student vs Teacher)
EVAL_SIMS = 400        # Stronger simulations for evaluation
WIN_THRESHOLD = 0.55   # Student must win 55% to be promoted

# Paths
PATHS = {
    "self_play": "./cmake-build-release/SelfPlay",
    "arbiter":   "./cmake-build-release/Arbiter",
    "train":     "../../../hex_nn/train.py",
    "export":    "../../../hex_nn/export_onnx.py",
    "model_dir": "models",
    "data_dir":  "data",
    "best_onnx": "models/best.onnx",
    "cand_pt":   "models/candidate/best.pt", # train.py saves here usually
    "cand_onnx": "models/candidate.onnx"
}

def run_command(cmd, log_file=None):
    """Helper to run shell commands and log output."""
    print(f"   [CMD] {' '.join(cmd)}")
    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, check=True)

def generate_data(iter_num):
    print(f"\n>>> [Iter {iter_num}] GENERATING DATA")
    output_file = f"{PATHS['data_dir']}/gen_{iter_num:03d}.jsonl"

    # NOTE: self_play.cpp usually loads a hardcoded model path or takes an arg.
    # Ensure 'models/best.onnx' exists before running this.

    cmd = [
        PATHS["self_play"],
        "agent",              # Mode
        str(GAMES_PER_ITER),  # Games
        str(MCTS_SIMS_GEN),   # Sims
        output_file,          # Output JSONL
        "0"                   # Save SGF? (0 = No, save disk space)
    ]
    run_command(cmd, log_file=f"logs/gen_{iter_num}.log")

def train_student(iter_num):
    print(f">>> [Iter {iter_num}] TRAINING STUDENT")

    # 1. Select Replay Buffer (Last N files)
    all_files = sorted(glob.glob(f"{PATHS['data_dir']}/*.jsonl"))
    train_files = all_files[-WINDOW_SIZE:]

    if not train_files:
        raise Exception("No training data found!")

    # 2. Run train.py
    # We resume from the previous best checkpoint to save time, or train from scratch?
    # Usually better to resume from the previous best.pt if available.
    cmd = [
              "python3", PATHS["train"],
              "--data"] + train_files + [
              "--epochs", str(TRAIN_EPOCHS),
              "--run-name", "candidate",
              "--output-dir", PATHS["model_dir"]
          ]

    # Optional: Resume from previous best to maintain knowledge
    # if iter_num > 1:
    #     cmd += ["--resume", f"{PATHS['model_dir']}/best_checkpoint.pt"]

    run_command(cmd)

def export_student():
    print(f">>> EXPORTING CANDIDATE TO ONNX")
    cmd = [
        "python3", PATHS["export"],
        PATHS["cand_pt"],
        PATHS["cand_onnx"]
    ]
    run_command(cmd)

def evaluate_student(iter_num):
    print(f">>> [Iter {iter_num}] EVALUATING (Student vs Teacher)")

    # Arbiters usually take commands to run engines.
    # We need a wrapper script that loads a specific ONNX file.
    # Assuming "./Group49 <onnx_path>" runs your engine.

    cmd_student = f"./Group49 {PATHS['cand_onnx']}"
    cmd_teacher = f"./Group49 {PATHS['best_onnx']}"

    cmd = [
        PATHS["arbiter"],
        cmd_student,
        cmd_teacher,
        str(EVAL_GAMES // 2), # Pairs (so x2 games)
        "6",                  # Threads
        str(EVAL_SIMS)
    ]

    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse Win Rate (Arbiter output: "Agent A Wins: X")
    match = re.search(r"Agent A Wins: (\d+)", result.stdout)
    if not match:
        print("   [!] Error parsing Arbiter output. Assuming 0 wins.")
        print(result.stdout)
        return False

    wins = int(match.group(1))
    win_rate = wins / EVAL_GAMES
    print(f"   Results: Student won {wins}/{EVAL_GAMES} ({win_rate*100:.1f}%)")

    return win_rate >= WIN_THRESHOLD

def promote_student(iter_num):
    print(f">>> PROMOTING STUDENT TO TEACHER")
    # Archive old teacher
    shutil.copy(PATHS["best_onnx"], f"{PATHS['model_dir']}/archive/best_{iter_num-1}.onnx")
    # Replace teacher
    shutil.copy(PATHS["cand_onnx"], PATHS["best_onnx"])
    # Also save the .pt for resuming training later
    shutil.copy(PATHS["cand_pt"], f"{PATHS['model_dir']}/best_checkpoint.pt")

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"{PATHS['model_dir']}/archive", exist_ok=True)

    for i in range(ITERATION_START, ITERATIONS + 1):
        start_time = time.time()

        try:
            generate_data(i)
            train_student(i)
            export_student()

            if evaluate_student(i):
                promote_student(i)
            else:
                print(f">>> REJECTED. Keeping current Teacher.")

        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            break

        print(f"--- Iteration {i} finished in {(time.time()-start_time)/60:.1f} mins ---\n")

if __name__ == "__main__":
    main()