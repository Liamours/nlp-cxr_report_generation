"""
Orchestrator: preprocess both datasets then run count verification.

Usage:
    uv run python src/script/run_preprocess.py [--dataset chexpertplus|mimiccxr|all]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocess import chexpertplus, mimiccxr
from src.util.count import run_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["chexpertplus", "mimiccxr", "all"], default="all"
    )
    args = parser.parse_args()

    if args.dataset in ("chexpertplus", "all"):
        print("\n[1/2] Preprocessing CheXpert+...")
        chexpertplus.run()

    if args.dataset in ("mimiccxr", "all"):
        print("\n[2/2] Preprocessing MIMIC-CXR...")
        mimiccxr.run()

    print("\n[Verification] Counting and checking against paper values...")
    run_all()


if __name__ == "__main__":
    main()
