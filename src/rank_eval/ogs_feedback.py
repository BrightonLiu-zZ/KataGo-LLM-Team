#!/usr/bin/env python3
"""Collect the humans' chat messages (feedback) out of the gtp2ogs logs.

The bots ask their opponent for feedback when the game ends (``farewell`` in
config/ogs/bot-common.json5).  gtp2ogs writes every message the *other* player
sends -- including ones typed after the game is over, which arrive as
``lateChatReceivedInGame`` notifications -- to its logfile as::

    Aug 18 13:08:35   Game.ts:123   [game 1234567] Game chat from alice: nice shape

...but only when ``log_game_chat: true``.  This script turns those lines into a
JSONL record per message plus a readable digest, so the feedback can be read in
one place and quoted in the rank report.

    python src/rank_eval/ogs_feedback.py                 # digest of everything
    python src/rank_eval/ogs_feedback.py --since 2026-08-19
    python src/rank_eval/ogs_feedback.py --json          # JSONL to stdout

Game results (``Game over.   Result: B+R  W``) are picked up too, so each
message can be shown next to how that game ended.
"""
import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LOG_DIR = os.path.join(REPO, "src", "rank_eval", "logs", "ogs")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS = r"(?P<ts>[A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d)"
CHAT_RE = re.compile(TS + r"\s+\S+\s+\[game (?P<game>\d+)\] Game chat from (?P<user>.+?): (?P<body>.*)$")
RESULT_RE = re.compile(TS + r"\s+\S+\s+\[game (?P<game>\d+)\] Game over\.\s+Result: (?P<result>\S+)\s+(?P<winloss>[WL])")
BOT_RE = re.compile(r"(?:console|gtp2ogs)-(?:ogs-)?(?P<bot>[a-z0-9]+)\.log$")


def parse_ts(raw: str, year: int) -> str:
    """gtp2ogs timestamps carry no year; assume ``year`` (default: this one)."""
    try:
        dt = _dt.datetime.strptime(f"{raw} {year}", "%b %d %H:%M:%S %Y")
    except ValueError:
        return raw
    return dt.isoformat(sep=" ")


def collect(log_dir: str, year: int):
    """Returns (messages, results) parsed from every log file in ``log_dir``."""
    messages, results = [], {}
    seen = set()
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        m = BOT_RE.search(os.path.basename(path))
        bot = m.group("bot") if m else "?"
        with open(path, errors="replace") as f:
            for raw in f:
                line = ANSI_RE.sub("", raw).rstrip("\n")
                hit = CHAT_RE.search(line)
                if hit:
                    rec = {"ts": parse_ts(hit.group("ts"), year), "bot": bot,
                           "game_id": int(hit.group("game")), "user": hit.group("user"),
                           "body": hit.group("body").strip(),
                           "url": f"https://online-go.com/game/{hit.group('game')}"}
                    key = (rec["game_id"], rec["user"], rec["body"], rec["ts"])
                    if key not in seen:          # console-*.log duplicates gtp2ogs-*.log
                        seen.add(key)
                        messages.append(rec)
                    continue
                hit = RESULT_RE.search(line)
                if hit:
                    results[int(hit.group("game"))] = {
                        "result": hit.group("result"),
                        "bot_won": hit.group("winloss") == "W",
                        "ts": parse_ts(hit.group("ts"), year), "bot": bot}
    messages.sort(key=lambda r: (r["ts"], r["game_id"]))
    return messages, results


def digest(messages, results) -> str:
    if not messages:
        return "no opponent chat found (games with no talkative humans, or log_game_chat is off)"
    out, last = [], None
    for m in messages:
        if m["game_id"] != last:
            last = m["game_id"]
            res = results.get(m["game_id"])
            tail = f"  ({res['result']}, bot {'won' if res['bot_won'] else 'lost'})" if res else ""
            out.append(f"\n=== game {m['game_id']}  vs bot {m['bot']}{tail}\n    {m['url']}")
        out.append(f"  [{m['ts']}] {m['user']}: {m['body']}")
    users = {m["user"] for m in messages}
    games = {m["game_id"] for m in messages}
    out.append(f"\n{len(messages)} messages from {len(users)} players across {len(games)} games")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--since", default=None, help="ISO date/time, e.g. 2026-08-19")
    ap.add_argument("--year", type=int, default=_dt.date.today().year,
                    help="year to stamp on the log's year-less timestamps")
    ap.add_argument("--json", action="store_true", help="print JSONL instead of the digest")
    ap.add_argument("--out", default=None, help="also append the JSONL records to this file")
    args = ap.parse_args(argv)

    messages, results = collect(args.log_dir, args.year)
    if args.since:
        messages = [m for m in messages if m["ts"] >= args.since]
    if args.out:
        with open(args.out, "w") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if args.json:
        for m in messages:
            print(json.dumps(m, ensure_ascii=False))
    else:
        print(digest(messages, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
