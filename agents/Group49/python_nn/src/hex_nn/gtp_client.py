"""GTP (Go Text Protocol) client for communicating with Hex engines like MoHex."""

from __future__ import annotations

import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .constants import BOARD_SIZE


@dataclass
class GTPResponse:
    """Response from a GTP command."""
    success: bool
    response: str


class GTPClient:
    """
    Client for communicating with a GTP-compatible Hex engine.

    GTP (Go Text Protocol) is a text-based protocol originally designed for Go
    but also used by Hex engines like MoHex.

    Usage:
        client = GTPClient("/path/to/mohex")
        client.start()
        client.set_boardsize(11)
        client.clear_board()
        move = client.genmove("black")  # Returns e.g. "f6"
        client.play("white", "e5")
        client.quit()
    """

    def __init__(self, engine_path: str | Path, args: Optional[list[str]] = None):
        """
        Initialize the GTP client.

        Args:
            engine_path: Path to the GTP engine executable
            args: Optional command-line arguments for the engine
        """
        self.engine_path = Path(engine_path)
        self.args = args or []
        self.process: Optional[subprocess.Popen] = None
        self._command_id = 0

    def start(self) -> None:
        """Start the engine subprocess."""
        if self.process is not None:
            raise RuntimeError("Engine already started")

        cmd = [str(self.engine_path)] + self.args
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

    def _send_command(self, command: str) -> GTPResponse:
        """
        Send a command to the engine and wait for response.

        GTP responses start with '=' for success or '?' for failure,
        followed by an optional id, then the response text.
        Response ends with two newlines.
        """
        if self.process is None:
            raise RuntimeError("Engine not started. Call start() first.")

        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()

        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "\n" and response_lines:
                break
            response_lines.append(line.rstrip("\n"))

        full_response = "\n".join(response_lines).strip()

        if not full_response:
            return GTPResponse(success=False, response="Empty response")

        success = full_response[0] == "="
        response_text = re.sub(r"^[=?]\d*\s*", "", full_response)

        return GTPResponse(success=success, response=response_text)

    def run_command(self, command: str) -> GTPResponse:
        """Send an arbitrary GTP command (e.g., parameter tweaks)."""
        return self._send_command(command)

    def set_boardsize(self, size: int = BOARD_SIZE) -> GTPResponse:
        """Set the board size."""
        return self._send_command(f"boardsize {size}")

    def clear_board(self) -> GTPResponse:
        """Clear the board for a new game."""
        return self._send_command("clear_board")

    def play(self, color: str, vertex: str) -> GTPResponse:
        """
        Play a move on the board.

        Args:
            color: "black" or "white" (MoHex convention)
            vertex: Board position like "a1", "f6", "k11"
        """
        return self._send_command(f"play {color} {vertex}")

    def genmove(self, color: str) -> GTPResponse:
        """
        Generate and play a move for the given color.

        Args:
            color: "black" or "white"

        Returns:
            GTPResponse with the move vertex (e.g., "f6") or "resign"
        """
        return self._send_command(f"genmove {color}")

    def undo(self) -> GTPResponse:
        """Undo the last move."""
        return self._send_command("undo")

    def showboard(self) -> GTPResponse:
        """Get a text representation of the current board."""
        return self._send_command("showboard")

    def quit(self) -> None:
        """Quit the engine and terminate the subprocess."""
        if self.process is not None:
            try:
                self._send_command("quit")
            except:
                pass
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None

    def __enter__(self) -> "GTPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quit()


def vertex_to_coords(vertex: str, skip_i: bool = False) -> Tuple[int, int]:
    """
    Convert GTP vertex (e.g., "f6") to board coordinates (row, col).

    GTP uses letters a-k for columns (skipping 'i') and 1-11 for rows.
    Note: Some implementations use 'i', some skip it. MoHex typically skips 'i'.

    Returns:
        (row, col) tuple, 0-indexed
    """
    vertex = vertex.lower().strip()

    if vertex in ("pass", "resign"):
        return (-1, -1)

    col_char = vertex[0]
    row_num = int(vertex[1:])

    col = ord(col_char) - ord('a')
    if skip_i and col_char > 'i':
        col -= 1

    row = row_num - 1

    return (row, col)


def coords_to_vertex(row: int, col: int) -> str:
    """
    Convert board coordinates (row, col) to GTP vertex (e.g., "f6").

    Args:
        row: 0-indexed row
        col: 0-indexed column

    Returns:
        GTP vertex string like "a1", "f6", etc.
    """
    if col >= 8:  # >= 'i' position
        col_char = chr(ord('a') + col + 1)
    else:
        col_char = chr(ord('a') + col)

    row_num = row + 1

    return f"{col_char}{row_num}"


def move_to_index(row: int, col: int, board_size: int = BOARD_SIZE) -> int:
    """Convert (row, col) to flat index 0-120."""
    return row * board_size + col


def index_to_coords(index: int, board_size: int = BOARD_SIZE) -> Tuple[int, int]:
    """Convert flat index 0-120 to (row, col)."""
    row = index // board_size
    col = index % board_size
    return (row, col)
