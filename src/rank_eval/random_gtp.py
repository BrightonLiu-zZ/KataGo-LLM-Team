#!/usr/bin/env python3
"""Uniform-random legal-move GTP bot (bottom rung of the 9x9 ladder).

Plays a random legal move that is not an own true eye; passes when none is
left. ``--seed`` for reproducibility.
"""
import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_gtp import LLMEngine, gtp_loop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    engine = LLMEngine(lambda msgs: [], max_attempts=1, eye_guard=True, valid_list="legal")
    engine.NAME = "random-9x9"

    def genmove(color):
        legal = [p for p in engine.board.valid_coords(color) if not engine.board.is_own_true_eye(color, p)]
        mv = rng.choice(legal) if legal else "pass"
        engine.board.play(color, mv)
        return mv

    engine.genmove = genmove
    gtp_loop(engine)


if __name__ == "__main__":
    main()
