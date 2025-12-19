import os
from subprocess import Popen, PIPE
from copy import deepcopy

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class CppAgentWrapper(AgentBase):
    """
    Python wrapper for the C++ Hex agent.
    Communicates via stdin/stdout using the coursework protocol.
    """

    def __deepcopy__(self, memo):
        """
        Required because the game engine may deepcopy agents.
        We must NOT deepcopy the subprocess.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.__dict__ = {
            k: deepcopy(v, memo)
            for k, v in self.__dict__.items()
            if k != "agent_process"
        }
        result.agent_process = None
        return result

    def __init__(self, colour: Colour):
        super().__init__(colour)

        # --- FIX: LOCATE DEPENDENCIES ---
        # Get the absolute path to this script's directory (agents/Group49/)
        cwd = os.path.dirname(os.path.abspath(__file__))

        # --- FIX: UPDATE LIBRARY PATH ---
        # Create a copy of the environment variables
        env = os.environ.copy()
        # Prepend the current directory to LD_LIBRARY_PATH
        # This tells the loader (and plugins) to look for libcudnn.so.9 HERE.
        env["LD_LIBRARY_PATH"] = f"{cwd}:{env.get('LD_LIBRARY_PATH', '')}"

        # Launch the compiled C++ agent
        self.agent_process = Popen(
            [
                "./cpp_agent",            # Now relative to cwd
                colour.get_char(colour),  # "R" or "B"
                "11",
            ],
            stdin=PIPE,
            stdout=PIPE,
            text=True,
            cwd=cwd,   # CRITICAL: Run inside agents/Group49 so it finds models/libs
            env=env    # CRITICAL: Pass the environment with LD_LIBRARY_PATH
        )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Called by the Hex engine every turn.
        Sends COMMAND;MOVE;BOARD;TURN; to the C++ agent
        and expects "x,y" or "-1,-1" in return.
        """

        # ---- serialize board ----
        board_strings = []
        for row in board.tiles:
            s = ""
            for tile in row:
                if tile.colour is None:
                    s += "0"
                else:
                    s += tile.colour.get_char(tile.colour)
            board_strings.append(s)

        board_string = ",".join(board_strings)

        # ---- build command ----
        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.is_swap():
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        # ---- send to C++ agent ----
        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        # ---- read response ----
        response = self.agent_process.stdout.readline().strip()
        x, y = response.split(",")

        return Move(int(x), int(y))
