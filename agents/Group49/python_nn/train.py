from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from hex_nn.config import defaults, TrainConfig
from hex_nn.dataset import HexDataset
from hex_nn.model import HexPolicyValueNet


def parse_args() -> argparse.Namespace:
    cfg = defaults()
    parser = argparse.ArgumentParser(
        description="Train Hex policy/value network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--data", nargs="+", help="JSONL files (default: all .jsonl in data-dir)")
    parser.add_argument("--data-dir", default=cfg.data_dir, help="Directory to scan for .jsonl files")
    # Run
    parser.add_argument("--run-name", default=cfg.run_name)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--resume", type=str, default=cfg.resume_checkpoint, help="Resume from checkpoint")
    # Model
    parser.add_argument("--num-blocks", type=int, default=cfg.num_blocks, help="ResNet blocks")
    parser.add_argument("--channels", type=int, default=cfg.num_channels, help="ResNet filters")
    # Training
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    # Performance
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers, help="DataLoader workers")
    return parser.parse_args()


def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_loss: float) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(payload, path)

def get_best_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"  Model:       {args.num_blocks} blocks, {args.channels} filters")
    print(f"  Training:    {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  Performance: {args.num_workers} workers")
    print(f"  Output:      {args.output_dir}/{args.run_name}")
    print("=" * 60)

    if args.data:
        data_files = [Path(p) for p in args.data]
    else:
        data_dir = Path(args.data_dir)
        data_files = sorted(data_dir.glob("*.jsonl"))
        if not data_files:
            print(f"No .jsonl files found in {data_dir}")
            return

    print(f"\nLoading data from {len(data_files)} file(s):")
    for f in data_files:
        print(f"  - {f}")

    full_dataset = HexDataset(data_files)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    device = get_best_device()
    print(f"using device: {device.type}")
    print(f"train: {train_size} samples, {len(train_loader)} batches")
    print(f"val:   {val_size} samples, {len(val_loader)} batches")
    model = HexPolicyValueNet(channels=args.channels, num_blocks=args.num_blocks).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = 0 # start_epoch = state["epoch"] + 1
        best_loss = float("inf")# best_loss = state.get("best_loss", best_loss)

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_samples = 0

        for states, policies, values in train_loader:
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

            batch_size = states.size(0)
            train_loss += loss.item() * batch_size
            train_policy_loss += policy_loss.item() * batch_size
            train_value_loss += value_loss.item() * batch_size
            train_correct += (policy_logits.argmax(dim=1) == policies.argmax(dim=1)).sum().item()
            train_samples += batch_size

        model.eval()
        val_loss = 0.0
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for states, policies, values in val_loader:
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                policy_logits, value_pred = model(states)
                policy_loss = criterion_policy(policy_logits, policies.argmax(dim=1))
                value_loss = criterion_value(value_pred, values)
                loss = policy_loss + value_loss

                batch_size = states.size(0)
                val_loss += loss.item() * batch_size
                val_policy_loss += policy_loss.item() * batch_size
                val_value_loss += value_loss.item() * batch_size
                val_correct += (policy_logits.argmax(dim=1) == policies.argmax(dim=1)).sum().item()
                val_samples += batch_size

        epoch_time = time.time() - epoch_start

        train_loss /= train_samples
        train_policy_loss /= train_samples
        train_value_loss /= train_samples
        train_acc = 100.0 * train_correct / train_samples

        val_loss /= val_samples
        val_policy_loss /= val_samples
        val_value_loss /= val_samples
        val_acc = 100.0 * val_correct / val_samples

        scheduler.step(val_loss)

        epochs_remaining = args.epochs - (epoch + 1)
        eta_seconds = epochs_remaining * epoch_time
        if eta_seconds >= 3600:
            eta_str = f"{eta_seconds / 3600:.1f}h"
        elif eta_seconds >= 60:
            eta_str = f"{eta_seconds / 60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"

        is_best = val_loss < best_loss
        best_marker = " [new best]" if is_best else ""
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"val: loss={val_loss:.4f} acc={val_acc:.1f}% | "
            f"{epoch_time:.1f}s | ETA: {eta_str}{best_marker}"
        )

        checkpoint_path = output_dir / f"epoch_{epoch+1}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss)
        if is_best:
            best_loss = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_loss)

    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
