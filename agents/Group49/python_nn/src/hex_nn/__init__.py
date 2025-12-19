"""hex_nn package."""

try:
    from .dataset import HexDataset  # noqa: F401
except Exception:  # pragma: no cover - torch may be missing during tooling scripts
    HexDataset = None  # type: ignore

from .io import load_selfplay_file  # noqa: F401
from .gtp_client import GTPClient, vertex_to_coords, coords_to_vertex  # noqa: F401
from .game_runner import GameRunner, play_mock_games  # noqa: F401
