#!/usr/bin/env python3
"""Minimal two-engine GTP match runner (9x9) with a GnuGo referee.

A dependency-free stand-in for gogui-twogtp (which needs Java) so the LLM
engine can be exercised end-to-end and, later, run against the fixed-strength
ladder in docs/RANK-EVAL-PLAN.md §3.B.

    python src/rank_eval/twogtp.py \
        --black "python src/rank_eval/llm_gtp.py --model go-v5c" \
        --white "$HOME/.local/bin/gnugo --mode gtp --level 10 --chinese-rules --capture-all-dead" \
        --games 4 --komi 7.0 --alternate --sgf-dir tmp/twogtp --jobs 2

Game end: two consecutive passes, a resign, or --max-moves. Scoring: the
referee (GnuGo by default) replays the game and answers ``final_score``
(area scoring with --chinese-rules). Results are appended as JSONL.

``--black-ending/--white-ending "<gtp cmd>"`` attaches an *ending bot* to a
side exactly like gtp2ogs' ``ending_bot`` (pass when it passes, resign after
``--allowed-resigns`` consecutive resign votes, consulted after
``--ending-ratio``*81 moves) so local matches predict OGS behaviour.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

DEFAULT_REFEREE = os.path.expanduser("~/.local/bin/gnugo") + " --mode gtp --chinese-rules --capture-all-dead --level 10"


class GTPEngine:
    def __init__(self, command: str, name: str) -> None:
        self.name = name
        self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lock = threading.Lock()

    def send(self, cmd: str) -> str:
        with self.lock:
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
            lines = []
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    raise RuntimeError(f"{self.name}: engine died on '{cmd}'")
                if line.strip() == "" and lines:
                    break
                if line.strip() == "" and not lines:
                    continue
                lines.append(line.rstrip("\n"))
            resp = "\n".join(lines)
            if resp.startswith("?"):
                raise RuntimeError(f"{self.name}: '{cmd}' -> {resp}")
            return resp[1:].strip() if resp.startswith("=") else resp

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def sgf_of(moves: List[Tuple[str, str]], komi: float, pb: str, pw: str, result: str) -> str:
    cols = "ABCDEFGHJ"

    def pt(m: str) -> str:
        if m.lower() == "pass":
            return ""
        c = cols.index(m[0].upper())
        r = int(m[1:]) - 1
        return "abcdefghi"[c] + "abcdefghi"[8 - r]

    body = "".join(f";{c}[{pt(m)}]" for c, m in moves)
    return f"(;GM[1]FF[4]SZ[9]KM[{komi}]DT[{_dt.date.today().isoformat()}]PB[{pb}]PW[{pw}]RE[{result}]{body})\n"


def play_game(idx: int, black_cmd: str, white_cmd: str, black_name: str, white_name: str,
              komi: float, max_moves: int, referee_cmd: str, sgf_dir: Optional[str],
              black_ending: Optional[str] = None, white_ending: Optional[str] = None,
              ending_ratio: float = 0.35, allowed_resigns: int = 3) -> dict:
    """One game. ``*_ending`` = optional GTP command of an *ending bot* for that
    side, mirroring gtp2ogs: after ceil(81*ending_ratio) moves it is asked for
    a move too; if it says pass we pass, if it says resign ``allowed_resigns``
    times in a row we resign; otherwise the main engine's move is played."""
    t0 = time.time()
    black = GTPEngine(black_cmd, black_name)
    white = GTPEngine(white_cmd, white_name)
    ref = GTPEngine(referee_cmd, "referee")
    enders = {"B": GTPEngine(black_ending, "end-B") if black_ending else None,
              "W": GTPEngine(white_ending, "end-W") if white_ending else None}
    engines = [black, white, ref] + [e for e in enders.values() if e]
    moves: List[Tuple[str, str]] = []
    result = "?"
    reason = ""
    overrides = {"B": 0, "W": 0}
    resign_votes = {"B": 0, "W": 0}
    start_checking = int(-(-81 * ending_ratio // 1))
    try:
        for e in engines:
            e.send("boardsize 9")
            e.send("clear_board")
            e.send(f"komi {komi}")
        passes = 0
        color = "B"
        while len(moves) < max_moves:
            me, other = (black, white) if color == "B" else (white, black)
            full = "black" if color == "B" else "white"
            mv = me.send(f"genmove {full}").strip()
            ender = enders[color]
            if ender is not None and len(moves) >= start_checking:
                emv = ender.send(f"genmove {full}").strip().lower()
                if emv == "resign":
                    resign_votes[color] += 1
                else:
                    resign_votes[color] = 0
                    ender.send("undo")  # the ender only *votes*; the real move is replayed below
                if emv == "pass" and mv.lower() != "pass":
                    mv = "pass"
                    overrides[color] += 1
                    me.send("undo")
                    me.send(f"play {full} pass")
                elif resign_votes[color] >= allowed_resigns:
                    mv = "resign"
                    overrides[color] += 1
            if mv.lower() == "resign":
                result = ("W" if color == "B" else "B") + "+Resign"
                reason = "resign"
                break
            moves.append((color, mv))
            other.send(f"play {full} {mv}")
            ref.send(f"play {full} {mv}")
            for e in enders.values():  # every ending bot tracks the whole game
                if e is not None:
                    e.send(f"play {full} {mv}")
            passes = passes + 1 if mv.lower() == "pass" else 0
            if passes >= 2:
                reason = "two_passes"
                break
            color = "W" if color == "B" else "B"
        else:
            reason = "max_moves"
        if result == "?":
            result = ref.send("final_score").strip()
    except Exception as e:  # engine crash etc.
        result = "?"
        reason = f"error: {e}"
    finally:
        for e in engines:
            e.close()
    rec = {"game": idx, "black": black_name, "white": white_name, "komi": komi, "result": result,
           "moves": len(moves), "reason": reason, "seconds": round(time.time() - t0, 1),
           "black_passes": sum(1 for c, m in moves if c == "B" and m.lower() == "pass"),
           "white_passes": sum(1 for c, m in moves if c == "W" and m.lower() == "pass"),
           "ending_overrides": overrides}
    if sgf_dir:
        os.makedirs(sgf_dir, exist_ok=True)
        with open(os.path.join(sgf_dir, f"game{idx:04d}_{black_name}_vs_{white_name}.sgf"), "w") as f:
            f.write(sgf_of(moves, komi, black_name, white_name, result))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--black", required=True, help="engine command (player A)")
    ap.add_argument("--white", required=True, help="engine command (player B)")
    ap.add_argument("--black-name", default="A")
    ap.add_argument("--white-name", default="B")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--komi", type=float, default=7.0)
    ap.add_argument("--alternate", action="store_true", help="swap colours every game")
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--referee", default=DEFAULT_REFEREE)
    ap.add_argument("--sgf-dir", default=None)
    ap.add_argument("--results", default=None, help="JSONL results file (append)")
    ap.add_argument("--black-ending", default=None, help="ending-bot GTP command for player A (see play_game)")
    ap.add_argument("--white-ending", default=None, help="ending-bot GTP command for player B")
    ap.add_argument("--ending-ratio", type=float, default=0.35)
    ap.add_argument("--allowed-resigns", type=int, default=3)
    args = ap.parse_args()

    jobs = []
    for i in range(args.games):
        swap = args.alternate and i % 2 == 1
        b_cmd, w_cmd = (args.white, args.black) if swap else (args.black, args.white)
        b_name, w_name = (args.white_name, args.black_name) if swap else (args.black_name, args.white_name)
        b_end, w_end = (args.white_ending, args.black_ending) if swap else (args.black_ending, args.white_ending)
        jobs.append((i, b_cmd, w_cmd, b_name, w_name, b_end, w_end))

    wins = {args.black_name: 0, args.white_name: 0}
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(play_game, i, b, w, bn, wn, args.komi, args.max_moves, args.referee, args.sgf_dir,
                          be, we, args.ending_ratio, args.allowed_resigns)
                for i, b, w, bn, wn, be, we in jobs]
        for fut in cf.as_completed(futs):
            rec = fut.result()
            winner = rec["black"] if rec["result"].startswith("B") else rec["white"] if rec["result"].startswith("W") else None
            if winner:
                wins[winner] += 1
            print(json.dumps(rec), flush=True)
            if args.results:
                with open(args.results, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    total = sum(wins.values())
    print(f"# {args.black_name} {wins[args.black_name]} - {wins[args.white_name]} {args.white_name} "
          f"({total} decided of {args.games})", file=sys.stderr)


if __name__ == "__main__":
    main()
