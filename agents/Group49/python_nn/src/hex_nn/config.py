from __future__ import annotations

from dataclasses import dataclass


def defaults() -> "TrainConfig":
    return TrainConfig()


@dataclass
class TrainConfig:
    """
    Training configuration with cross-platform defaults.
    Works on both Apple Silicon (MPS) and NVIDIA GPUs (CUDA).

    Tune for your hardware:
    - batch_size: Increase if you have more VRAM (try 1024, 2048)
    - num_workers: Set to ~50-70% of your CPU threads
    """
    # Run settings
    run_name: str = "run_001"
    output_dir: str = "checkpoints"
    data_dir: str = "data"
    resume_checkpoint: str | None = None

    # Model architecture
    num_blocks: int = 5
    num_channels: int = 128  # Filter count in ResNet

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    num_workers: int = 10  # ~70% of 14 threads
    pin_memory: bool = True  # Speeds up CUDA, no effect on MPS
