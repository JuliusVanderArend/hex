"""
Game runner for generating training data from MoHex vs MoHex self-play.

Plays games using GTP protocol and records positions for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .constants import BOARD_SIZE, BOARD_AREA
from .gtp_client import GTPClient, vertex_to_coords, move_to_index


# Player constants
BLACK = 0  # First player, connects top-bottom (like RED in hex framework)
WHITE = 1  # Second player, connects left-right (like BLUE in hex framework)


@dataclass
class Position:
    """A single game position with the move that was played."""
    player: int  # 0 = BLACK (to move), 1 = WHITE (to move)
    red: np.ndarray  # BLACK stones (11x11)
    blue: np.ndarray  # WHITE stones (11x11, transposed to match C++)
    turn_plane: np.ndarray  # All 1s if BLACK to move, all 0s if WHITE
    last_move: np.ndarray  # One-hot for previous move location
    move_played: Tuple[int, int]  # The move that was actually played (row, col)
    # Connectivity planes for current player's stones
    conn_start: np.ndarray = field(default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32))
    conn_end: np.ndarray = field(default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32))


@dataclass
class GameRecord:
    """Complete record of a game."""
    positions: List[Position]
    winner: int  # 0 = BLACK won, 1 = WHITE won
    num_moves: int


class HexBoard:
    """Simple Hex board state tracker."""

    def __init__(self, size: int = BOARD_SIZE):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0=empty, 1=BLACK, 2=WHITE
        self.last_move: Optional[Tuple[int, int]] = None
        self.move_history: List[Tuple[int, Tuple[int, int]]] = []  # (player, (row, col))
        self.swapped = False  # Track if swap rule was used

    def play(self, player: int, row: int, col: int) -> bool:
        """
        Play a move.

        Args:
            player: 0 for BLACK, 1 for WHITE
            row, col: Board coordinates (0-indexed)

        Returns:
            True if move was valid and played, False otherwise
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row, col] != 0:
            return False

        self.board[row, col] = player + 1  # 1 for BLACK, 2 for WHITE
        self.last_move = (row, col)
        self.move_history.append((player, (row, col)))
        return True

    def get_red_plane(self) -> np.ndarray:
        """Get BLACK stones as a float plane."""
        return (self.board == 1).astype(np.float32)

    def get_blue_plane(self) -> np.ndarray:
        """Get WHITE stones as a float plane (transposed to match C++ representation)."""
        return (self.board == 2).astype(np.float32).T

    def get_last_move_plane(self) -> np.ndarray:
        """Get one-hot plane for last move."""
        plane = np.zeros((self.size, self.size), dtype=np.float32)
        if self.last_move is not None:
            plane[self.last_move[0], self.last_move[1]] = 1.0
        return plane

    def get_turn_plane(self, player: int) -> np.ndarray:
        """Get turn indicator plane (all 1s if BLACK to move, all 0s if WHITE)."""
        if player == BLACK:
            return np.ones((self.size, self.size), dtype=np.float32)
        else:
            return np.zeros((self.size, self.size), dtype=np.float32)

    def get_connectivity_planes(self, player: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute connectivity planes for the given player.

        Returns two planes:
        - conn_start: cells connected to start edge (top for BLACK, left for WHITE)
        - conn_end: cells connected to end edge (bottom for BLACK, right for WHITE)

        Uses flood-fill from edges to find connected components.
        """
        stone_value = player + 1  # 1 for BLACK, 2 for WHITE
        conn_start = np.zeros((self.size, self.size), dtype=np.float32)
        conn_end = np.zeros((self.size, self.size), dtype=np.float32)

        # Hex grid neighbors (6 directions)
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        def flood_fill(start_cells: List[Tuple[int, int]], result: np.ndarray) -> None:
            """Flood fill from start cells through connected stones."""
            visited = np.zeros((self.size, self.size), dtype=bool)
            stack = []

            # Initialize with start edge cells that have player's stones
            for r, c in start_cells:
                if self.board[r, c] == stone_value:
                    stack.append((r, c))
                    visited[r, c] = True
                    result[r, c] = 1.0

            while stack:
                r, c = stack.pop()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if not visited[nr, nc] and self.board[nr, nc] == stone_value:
                            visited[nr, nc] = True
                            result[nr, nc] = 1.0
                            stack.append((nr, nc))

        if player == BLACK:
            # BLACK connects top (row 0) to bottom (row size-1)
            start_cells = [(0, c) for c in range(self.size)]
            end_cells = [(self.size - 1, c) for c in range(self.size)]
        else:
            # WHITE connects left (col 0) to right (col size-1)
            start_cells = [(r, 0) for r in range(self.size)]
            end_cells = [(r, self.size - 1) for r in range(self.size)]

        flood_fill(start_cells, conn_start)
        flood_fill(end_cells, conn_end)

        return conn_start, conn_end

    def is_winner(self, player: int) -> bool:
        """
        Check if the given player has won.

        BLACK (player 0) wins by connecting top to bottom.
        WHITE (player 1) wins by connecting left to right.
        """
        stone_value = player + 1
        visited = np.zeros((self.size, self.size), dtype=bool)

        # Directions for hex grid neighbors
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        def dfs(row: int, col: int) -> bool:
            if visited[row, col]:
                return False
            if self.board[row, col] != stone_value:
                return False

            visited[row, col] = True

            # Check win condition
            if player == BLACK and row == self.size - 1:  # Bottom row
                return True
            if player == WHITE and col == self.size - 1:  # Right column
                return True

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if dfs(nr, nc):
                        return True
            return False

        # Start from appropriate edge
        if player == BLACK:  # Start from top row
            for col in range(self.size):
                if dfs(0, col):
                    return True
        else:  # Start from left column
            for row in range(self.size):
                if dfs(row, 0):
                    return True

        return False

    def swap(self) -> None:
        """
        Execute swap rule: second player takes first player's stone.

        In Hex swap rule, WHITE (second player) can choose to swap after BLACK's
        first move. This means WHITE takes over BLACK's stone and BLACK must
        play as if they were WHITE.

        We implement this by swapping the stone colors on the board.
        """
        # Swap all stones: 1 (BLACK) becomes 2 (WHITE) and vice versa
        new_board = np.zeros_like(self.board)
        new_board[self.board == 1] = 2  # BLACK -> WHITE
        new_board[self.board == 2] = 1  # WHITE -> BLACK
        self.board = new_board
        self.swapped = True
        # last_move stays the same (the position is still relevant)

    def copy(self) -> "HexBoard":
        """Create a copy of the board."""
        new_board = HexBoard(self.size)
        new_board.board = self.board.copy()
        new_board.last_move = self.last_move
        new_board.move_history = self.move_history.copy()
        new_board.swapped = self.swapped
        return new_board


class GameRunner:
    """
    Runs games between GTP engines and records training data.

    For MoHex vs MoHex self-play, you can use the same engine path for both.
    """

    def __init__(
        self,
        engine_path: str | Path,
        engine_args: Optional[List[str]] = None,
        board_size: int = BOARD_SIZE,
        init_commands: Optional[List[str]] = None,
        skip_gtp_i: bool = False,
    ):
        """
        Initialize the game runner.

        Args:
            engine_path: Path to the GTP engine (MoHex)
            engine_args: Optional arguments for the engine
            board_size: Size of the Hex board (default 11)
            init_commands: Optional list of GTP commands (e.g.,
                "param_mohex max_time 0.5") that are sent after
                boardsize/clear to configure the engine before every game
            skip_gtp_i: Whether to treat the column letter 'i' as skipped
                (some GTP engines omit it to avoid confusion with 'j').
        """
        self.engine_path = Path(engine_path)
        self.engine_args = engine_args or []
        self.board_size = board_size
        self.init_commands = init_commands or []
        self.skip_gtp_i = skip_gtp_i

    def play_game(self) -> Optional[GameRecord]:
        """
        Play a single game of MoHex vs MoHex.

        Returns:
            GameRecord with all positions and outcome, or None if game failed
        """
        positions: List[Position] = []
        board = HexBoard(self.board_size)
        current_player = BLACK

        try:
            with GTPClient(self.engine_path, self.engine_args) as engine:
                engine.set_boardsize(self.board_size)
                engine.clear_board()

                for command in self.init_commands:
                    response = engine.run_command(command)
                    if not response.success:
                        print(f"Engine init command failed: {command} -> {response.response}")
                        return None

                while True:
                    # Record position BEFORE the move
                    conn_start, conn_end = board.get_connectivity_planes(current_player)
                    position = Position(
                        player=current_player,
                        red=board.get_red_plane(),
                        blue=board.get_blue_plane(),
                        turn_plane=board.get_turn_plane(current_player),
                        last_move=board.get_last_move_plane(),
                        move_played=(-1, -1),  # Will be set after move
                        conn_start=conn_start,
                        conn_end=conn_end,
                    )

                    # Get move from engine
                    color = "black" if current_player == BLACK else "white"
                    response = engine.genmove(color)

                    if not response.success:
                        print(f"Engine error: {response.response}")
                        return None

                    move_str = response.response.strip().lower()

                    if move_str == "resign":
                        winner = 1 - current_player
                        break

                    if move_str in ("swap", "swap-pieces"):
                        board.swap()
                        current_player = 1 - current_player
                        print(f"  [SWAP executed]")
                        continue

                    # Parse and play the move
                    row, col = vertex_to_coords(move_str, skip_i=self.skip_gtp_i)
                    position.move_played = (row, col)
                    positions.append(position)

                    if not board.play(current_player, row, col):
                        print(f"Invalid move from engine: {move_str}")
                        return None

                    # Check for winner
                    if board.is_winner(current_player):
                        winner = current_player
                        break

                    # Switch player
                    current_player = 1 - current_player

                return GameRecord(
                    positions=positions,
                    winner=winner,
                    num_moves=len(positions),
                )

        except Exception as e:
            print(f"Error during game: {e}")
            return None

    def play_games(self, num_games: int, output_path: str | Path) -> int:
        """
        Play multiple games and save training data to JSONL.

        Args:
            num_games: Number of games to play
            output_path: Path to output JSONL file

        Returns:
            Number of games successfully completed
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        completed = 0
        total_positions = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for game_num in range(num_games):
                print(f"Playing game {game_num + 1}/{num_games}...", end=" ")

                record = self.play_game()
                if record is None:
                    print("FAILED")
                    continue

                for pos in record.positions:
                    value = 1.0 if pos.player == record.winner else -1.0

                    policy = np.zeros(BOARD_AREA, dtype=np.float32)
                    move_idx = move_to_index(pos.move_played[0], pos.move_played[1])
                    policy[move_idx] = 1.0

                    sample = {
                        "player": pos.player,
                        "red": pos.red.flatten().tolist(),
                        "blue": pos.blue.flatten().tolist(),
                        "turn": pos.turn_plane.flatten().tolist(),
                        "last_move": pos.last_move.flatten().tolist(),
                        "conn_start": pos.conn_start.flatten().tolist(),
                        "conn_end": pos.conn_end.flatten().tolist(),
                        "policy": policy.tolist(),
                        "value": value,
                    }

                    f.write(json.dumps(sample) + "\n")

                completed += 1
                total_positions += record.num_moves
                winner_str = "BLACK" if record.winner == BLACK else "WHITE"
                print(f"OK ({record.num_moves} moves, {winner_str} wins)")

        print(f"\nCompleted {completed}/{num_games} games")
        print(f"Total positions: {total_positions}")
        print(f"Output: {output_path}")

        return completed


class MockGTPEngine:
    """
    A mock GTP engine for testing without MoHex.

    Plays random legal moves.
    """

    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)

    def genmove(self, player: int) -> Tuple[int, int]:
        """Generate a random legal move."""
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return (-1, -1)  # No moves

        idx = np.random.randint(len(empty))
        row, col = empty[idx]
        self.board[row, col] = player + 1
        return (int(row), int(col))


def play_mock_games(num_games: int, output_path: str | Path) -> int:
    """
    Play games using mock random engine (for testing without MoHex).

    Args:
        num_games: Number of games to play
        output_path: Path to output JSONL file

    Returns:
        Number of games completed
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    total_positions = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for game_num in range(num_games):
            positions: List[Position] = []
            board = HexBoard(BOARD_SIZE)
            mock_engine = MockGTPEngine(BOARD_SIZE)
            current_player = BLACK

            while True:
                conn_start, conn_end = board.get_connectivity_planes(current_player)
                position = Position(
                    player=current_player,
                    red=board.get_red_plane(),
                    blue=board.get_blue_plane(),
                    turn_plane=board.get_turn_plane(current_player),
                    last_move=board.get_last_move_plane(),
                    move_played=(-1, -1),
                    conn_start=conn_start,
                    conn_end=conn_end,
                )

                row, col = mock_engine.genmove(current_player)
                if row == -1:
                    break  # No legal moves

                position.move_played = (row, col)
                positions.append(position)
                board.play(current_player, row, col)

                if board.is_winner(current_player):
                    winner = current_player
                    break

                current_player = 1 - current_player

            for pos in positions:
                value = 1.0 if pos.player == winner else -1.0
                policy = np.zeros(BOARD_AREA, dtype=np.float32)
                move_idx = move_to_index(pos.move_played[0], pos.move_played[1])
                policy[move_idx] = 1.0

                sample = {
                    "player": pos.player,
                    "red": pos.red.flatten().tolist(),
                    "blue": pos.blue.flatten().tolist(),
                    "turn": pos.turn_plane.flatten().tolist(),
                    "last_move": pos.last_move.flatten().tolist(),
                    "conn_start": pos.conn_start.flatten().tolist(),
                    "conn_end": pos.conn_end.flatten().tolist(),
                    "policy": policy.tolist(),
                    "value": value,
                }

                f.write(json.dumps(sample) + "\n")

            completed += 1
            total_positions += len(positions)

            if (game_num + 1) % 100 == 0:
                print(f"Completed {game_num + 1}/{num_games} games...")

    print(f"\nCompleted {completed}/{num_games} games")
    print(f"Total positions: {total_positions}")
    print(f"Output: {output_path}")

    return completed
