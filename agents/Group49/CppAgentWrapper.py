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

        cwd = os.path.dirname(os.path.abspath(__file__))

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{cwd}:{env.get('LD_LIBRARY_PATH', '')}"

        self.agent_process = Popen(
            [
                "./cpp_agent",            # Now relative to cwd
                colour.get_char(colour),  # "R" or "B"
                "11",
            ],
            stdin=PIPE,
            stdout=PIPE,
            text=True,
            cwd=cwd,
            env=env
        )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Called by the Hex engine every turn.
        Sends COMMAND;MOVE;BOARD;TURN; to C++ agent
        and expects "x,y" or "-1,-1" in return.
        """

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

        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.is_swap():
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().strip()
        x, y = response.split(",")

        return Move(int(x), int(y))
