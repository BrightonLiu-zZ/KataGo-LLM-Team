#!/usr/bin/env python3
"""Print OGS 9x9 ratings / ranks for player ids.

    python src/rank_eval/ogs_rating.py v5c=123456 base=234567 v4=345678
    python src/rank_eval/ogs_rating.py 605979            # any id, e.g. amybot-beginner

Uses https://online-go.com/termination-api/player/{id} — the same endpoint the
site uses; ``ratings`` has keys overall / 9x9 / live-9x9 / blitz-9x9 /
correspondence-9x9 (and 13x13, 19x19). Rank = ln(rating/525) * 23.15
(0 = 30k, 20 = 10k, 30 = 1d, 38 = 9d) — OGS's own formula.
"""
import json
import math
import sys
import urllib.request

CATS = ["overall", "9x9", "live-9x9", "blitz-9x9", "correspondence-9x9"]


def rank_str(rating: float) -> str:
    r = math.log(rating / 525.0) * 23.15
    return f"{30 - r:.1f}k" if r < 30 else f"{r - 30 + 1:.1f}d"


def fetch(pid: str) -> dict:
    req = urllib.request.Request(f"https://online-go.com/termination-api/player/{pid}",
                                 headers={"User-Agent": "katago-llm-rank-eval/0.1 (research bot admin)"})
    with urllib.request.urlopen(req, timeout=30) as f:
        return json.load(f)


def main(argv):
    if not argv:
        print(__doc__)
        return
    for arg in argv:
        label, _, pid = arg.rpartition("=")
        pid = pid or arg
        try:
            d = fetch(pid)
        except Exception as e:
            print(f"{label or pid}: fetch failed: {e}")
            continue
        head = f"{label or ''} {d.get('username', '?')} (id {pid}, bot={d.get('is_bot')})"
        print(head)
        ratings = d.get("ratings", {})
        for cat in CATS:
            r = ratings.get(cat)
            if not r:
                continue
            print(f"  {cat:20s} {r['rating']:7.1f} ± {r['deviation']:5.1f}   {rank_str(r['rating']):>6s}")


if __name__ == "__main__":
    main(sys.argv[1:])
