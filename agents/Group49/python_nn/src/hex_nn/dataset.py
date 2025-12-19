from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import BOARD_AREA, BOARD_SIZE, NUM_CHANNELS
from .io import RawSample, load_selfplay_file


def _build_planes(sample: RawSample) -> np.ndarray:
    planes = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    planes[0] = sample.red
    planes[1] = sample.blue

    planes[2] = sample.turn

    planes[3] = sample.last_move
    planes[4] = sample.conn_start
    planes[5] = sample.conn_end

    return planes


def _normalize_policy(policy: np.ndarray) -> np.ndarray:
    total = policy.sum()
    if total > 0:
        return policy / total
    return policy


def _record_from_raw(sample: RawSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = _build_planes(sample)
    policy = _normalize_policy(sample.policy.astype(np.float32))
    value = np.array(sample.value, dtype=np.float32)
    return state, policy, value


class HexDataset(Dataset):
    """Simple in-memory dataset backed by JSONL self-play logs."""

    def __init__(self, paths: Sequence[Path | str]):
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.values: List[np.ndarray] = []

        for path in paths:
            for raw in load_selfplay_file(path):
                state, policy, value = _record_from_raw(raw)
                self.states.append(state)
                self.policies.append(policy)
                self.values.append(value)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        state = torch.from_numpy(self.states[idx])
        policy = torch.from_numpy(self.policies[idx])
        value = torch.from_numpy(self.values[idx])
        return state, policy, value