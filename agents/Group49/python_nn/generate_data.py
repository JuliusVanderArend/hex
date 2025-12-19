#!/usr/bin/env python3
"""
Generate training data from MoHex vs MoHex self-play.

Usage:
    # With MoHex binary:
    python generate_data.py --engine /path/to/mohex --games 1000 --output data/mohex_selfplay.jsonl

    # For testing without MoHex (random moves):
    python generate_data.py --mock --games 100 --output data/mock_selfplay.jsonl
"""

import argparse
from pathlib import Path

from src.hex_nn.game_runner import GameRunner, play_mock_games


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from Hex self-play"
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="Path to GTP engine (e.g., MoHex)",
    )
    parser.add_argument(
        "--engine-arg",
        "--engine-args",
        dest="engine_args",
        action="append",
        default=[],
        help=(
            "Extra argument passed to the engine executable. "
            "Use the flag multiple times for multiple args."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/selfplay.jsonl",
        help="Output JSONL file path (default: data/selfplay.jsonl)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock random engine for testing (no MoHex required)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=11,
        help="Board size (default: 11)",
    )
    parser.add_argument(
        "--gtp-command",
        action="append",
        default=[],
        help=(
            "Optional GTP command to run after boardsize/clear to tune the engine. "
            "Repeat the flag for multiple commands (e.g., "
            "--gtp-command 'param_mohex max_time 0.5')."
        ),
    )
    parser.add_argument(
        "--gtp-skip-i",
        action="store_true",
        help=(
            "If set, treat the letter 'i' as skipped in GTP coordinates (some engines "
            "use a b c d e f g h j k ...). MoHex emits the literal 'i', so leave this off "
            "unless your engine omits it."
        ),
    )

    args = parser.parse_args()

    if args.mock:
        print("Using mock random engine for testing")
        print(f"Generating {args.games} games...")
        play_mock_games(args.games, args.output)
    elif args.engine:
        engine_path = Path(args.engine)
        if not engine_path.exists():
            print(f"Error: Engine not found at {engine_path}")
            print("Use --mock for testing without an engine")
            return 1

        print(f"Using engine: {engine_path}")
        print(f"Generating {args.games} games...")

        runner = GameRunner(
            engine_path=engine_path,
            engine_args=args.engine_args,
            board_size=args.board_size,
            init_commands=args.gtp_command,
            skip_gtp_i=args.gtp_skip_i,
        )
        runner.play_games(args.games, args.output)
    else:
        print("Error: Must specify --engine or --mock")
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
