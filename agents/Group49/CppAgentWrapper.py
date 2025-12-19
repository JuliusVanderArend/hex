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

        # Get the directory where this script is located
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(agent_dir, "lib")

        # Set LD_LIBRARY_PATH for ONNX Runtime
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = lib_path

        # Get plain "R" or "B" (not ANSI colored)
        colour_char = "R" if colour == Colour.RED else "B"

        # Launch the compiled C++ agent
        self.agent_process = Popen(
            [
                os.path.join(agent_dir, "Group49"),
                colour_char,
                "11",
            ],
            stdin=PIPE,
            stdout=PIPE,
            text=True,
            env=env,
            cwd=agent_dir,
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
                elif tile.colour == Colour.RED:
                    s += "R"
                else:
                    s += "B"
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
