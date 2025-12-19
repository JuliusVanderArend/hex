from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .constants import BOARD_AREA, BOARD_SIZE


@dataclass(frozen=True)
class RawSample:
    """Single row from the self-play exporter."""

    player: int
    red: np.ndarray
    blue: np.ndarray
    turn: np.ndarray
    last_move: np.ndarray
    conn_start: np.ndarray
    conn_end: np.ndarray
    policy: np.ndarray
    value: float


def _as_plane(values: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size != BOARD_AREA:
        raise ValueError(f"Plane has {arr.size} cells, expected {BOARD_AREA}")
    return arr.reshape(BOARD_SIZE, BOARD_SIZE)


def _as_policy(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size != BOARD_AREA:
        raise ValueError(f"Policy length {arr.size} does not match {BOARD_AREA}")
    return arr


def parse_line(line: str) -> RawSample:
    payload = json.loads(line)
    return RawSample(
        player=int(payload["player"]),
        red=_as_plane(payload["red"]),
        blue=_as_plane(payload["blue"]),
        turn=_as_plane(payload["turn"]),
        last_move=_as_plane(payload["last_move"]),
        conn_start=_as_plane(payload["conn_start"]),
        conn_end=_as_plane(payload["conn_end"]),
        policy=_as_policy(payload["policy"]),
        value=float(payload["value"]),
    )


def load_selfplay_file(path: Path | str) -> Iterator[RawSample]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue

            try:
                yield parse_line(line)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[Warning] Skipping corrupt line {i + 1} in {resolved.name}: {e}")
                continue