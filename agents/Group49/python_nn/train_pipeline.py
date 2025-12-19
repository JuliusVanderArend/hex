#!/usr/bin/env python3
"""
Training pipeline for Hex neural network.

This script orchestrates the full teacher-student training loop:
1. Generate training data from MoHex self-play (teacher)
2. Train the CNN on that data (student learns from teacher)
3. Export the trained model to ONNX for C++ runtime

Usage:
    # Full pipeline with MoHex:
    python train_pipeline.py --engine /path/to/mohex --games 1000 --epochs 50

    # Testing without MoHex (using random moves):
    python train_pipeline.py --mock --games 100 --epochs 10

    # Skip data generation (use existing data):
    python train_pipeline.py --data data/existing.jsonl --epochs 50

    # Resume training from checkpoint:
    python train_pipeline.py --data data/mohex.jsonl --resume checkpoints/run_001/best.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.hex_nn.config import defaults
from src.hex_nn.dataset import HexDataset
from src.hex_nn.model import HexPolicyValueNet
from src.hex_nn.game_runner import GameRunner, play_mock_games


def get_best_device() -> torch.device:
    """Get the best available device for training."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_loss: float,
) -> None:
    """Save a training checkpoint."""
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(payload, path)


def generate_data(
    engine_path: str | None,
    num_games: int,
    output_path: Path,
    use_mock: bool = False,
    engine_args: list[str] | None = None,
) -> bool:
    """
    Generate training data from self-play.

    Args:
        engine_path: Path to GTP engine (MoHex)
        num_games: Number of games to generate
        output_path: Where to save the JSONL data
        use_mock: Use random engine for testing
        engine_args: Arguments for the engine

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("STEP 1: Generating Training Data")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_mock:
        print(f"Using mock random engine (for testing)")
        print(f"Generating {num_games} games...")
        completed = play_mock_games(num_games, output_path)
    else:
        if not engine_path:
            print("Error: Must specify engine path or use --mock")
            return False

        engine = Path(engine_path)
        if not engine.exists():
            print(f"Error: Engine not found at {engine}")
            return False

        print(f"Using engine: {engine}")
        print(f"Generating {num_games} games...")

        runner = GameRunner(
            engine_path=engine,
            engine_args=engine_args or [],
        )
        completed = runner.play_games(num_games, output_path)

    return completed > 0


def train_model(
    data_paths: list[Path],
    run_name: str,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_blocks: int = 5,
    num_channels: int = 6,
    resume_checkpoint: str | None = None,
) -> Path:
    """
    Train the neural network.

    Args:
        data_paths: Paths to JSONL training data
        run_name: Name for this training run
        output_dir: Directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        num_blocks: Number of residual blocks
        num_channels: Number of input channels
        resume_checkpoint: Optional checkpoint to resume from

    Returns:
        Path to the best checkpoint
    """
    print("\n" + "=" * 60)
    print("STEP 2: Training Neural Network")
    print("=" * 60)

    dataset = HexDataset([str(p) for p in data_paths])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = get_best_device()
    print(f"Device: {device}")
    print(f"Training on {len(dataset)} samples")
    print(f"Batches per epoch: {len(loader)}")

    model = HexPolicyValueNet(
        input_channels=num_channels,
        num_blocks=num_blocks,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    start_epoch = 0
    best_loss = float("inf")

    if resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}")
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        best_loss = state.get("best_loss", best_loss)

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for states, policies, values in loader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()

            policy_logits, value_pred = model(states)
            policy_loss = criterion_policy(policy_logits, policies.argmax(dim=1))
            value_loss = criterion_value(value_pred, values)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            batch_size_actual = states.size(0)
            total_loss += loss.item() * batch_size_actual
            total_policy_loss += policy_loss.item() * batch_size_actual
            total_value_loss += value_loss.item() * batch_size_actual

        avg_loss = total_loss / len(dataset)
        avg_policy_loss = total_policy_loss / len(dataset)
        avg_value_loss = total_value_loss / len(dataset)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={avg_loss:.4f} (policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f})"
        )

        # Save checkpoints
        checkpoint_path = run_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = run_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, best_loss)
            print(f"  -> New best model saved!")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return run_dir / "best.pt"


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    num_channels: int = 6,
    num_blocks: int = 5,
) -> bool:
    """
    Export trained model to ONNX format.

    Args:
        checkpoint_path: Path to the checkpoint to export
        output_path: Where to save the ONNX model
        num_channels: Number of input channels
        num_blocks: Number of residual blocks

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("STEP 3: Exporting to ONNX")
    print("=" * 60)

    from src.hex_nn.model import HexPolicyValueNet

    class InferenceWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            policy_logits, value = self.model(x)
            policy_probs = torch.softmax(policy_logits, dim=1)
            return policy_probs, value

    device = torch.device("cpu")
    model = HexPolicyValueNet(input_channels=num_channels, num_blocks=num_blocks)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    export_model = InferenceWrapper(model)
    dummy = torch.zeros(1, num_channels, 11, 11, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes=None,
        opset_version=18,
    )

    print(f"Exported ONNX model to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Training pipeline for Hex neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with MoHex:
  python train_pipeline.py --engine /path/to/mohex --games 1000

  # Testing without MoHex:
  python train_pipeline.py --mock --games 100 --epochs 10

  # Use existing data:
  python train_pipeline.py --data data/mohex.jsonl --epochs 50
        """,
    )

    data_group = parser.add_argument_group("Data Generation")
    data_group.add_argument(
        "--engine",
        type=str,
        help="Path to GTP engine (MoHex)",
    )
    data_group.add_argument(
        "--engine-args",
        nargs="*",
        default=[],
        help="Arguments for the engine",
    )
    data_group.add_argument(
        "--mock",
        action="store_true",
        help="Use mock random engine for testing",
    )
    data_group.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)",
    )
    data_group.add_argument(
        "--data",
        type=str,
        nargs="+",
        help="Use existing data files (skip generation)",
    )

    # Training options
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    train_group.add_argument(
        "--num-blocks",
        type=int,
        default=5,
        help="Number of residual blocks (default: 5)",
    )
    train_group.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)",
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)",
    )
    output_group.add_argument(
        "--onnx-output",
        type=str,
        default=None,
        help="Path for ONNX export (default: models/<run_name>.onnx)",
    )
    output_group.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export",
    )

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Hex Neural Network Training Pipeline")
    print("=" * 60)
    print(f"Run name: {args.run_name}")

    if args.data:
        data_paths = [Path(p) for p in args.data]
        print(f"\nUsing existing data: {data_paths}")
    else:
        data_path = Path("data") / f"{args.run_name}.jsonl"
        success = generate_data(
            engine_path=args.engine,
            num_games=args.games,
            output_path=data_path,
            use_mock=args.mock,
            engine_args=args.engine_args,
        )
        if not success:
            print("Data generation failed!")
            return 1
        data_paths = [data_path]

    best_checkpoint = train_model(
        data_paths=data_paths,
        run_name=args.run_name,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_blocks=args.num_blocks,
        resume_checkpoint=args.resume,
    )

    if not args.skip_export:
        onnx_path = (
            Path(args.onnx_output)
            if args.onnx_output
            else Path("models") / f"{args.run_name}.onnx"
        )
        export_onnx(
            checkpoint_path=best_checkpoint,
            output_path=onnx_path,
            num_blocks=args.num_blocks,
        )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Checkpoint: {best_checkpoint}")
    if not args.skip_export:
        print(f"ONNX model: {onnx_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
