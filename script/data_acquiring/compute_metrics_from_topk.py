import json
from pathlib import Path
from statistics import pstdev

PATH_TOPK = Path(r"D:\katago_old\lizzie\json_output_with_top_k.jsonl")
PATH_METRICS = Path(r"D:\katago_old\lizzie\metrics.jsonl")

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    n_in, n_out, n_skip = 0, 0, 0
    with PATH_TOPK.open("r", encoding="utf-8") as fin, \
         PATH_METRICS.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                js = json.loads(line)
                # Skip mid-search snapshots if any
                if js.get("isDuringSearch") is True:
                    continue

                mid = js.get("moveInfos") or []
                if not mid:
                    n_skip += 1
                    continue

                # Collect winrates and scoreLead; keep index of best by winrate
                wrs = []
                sls = []
                for mi in mid[:10]:  # limit to top-k ~10
                    wr = to_float(mi.get("winrate"))
                    sl = to_float(mi.get("scoreLead"))
                    if wr is None:
                        continue
                    wrs.append(wr)
                    sls.append(sl)

                if not wrs:
                    n_skip += 1
                    continue

                # Sort desc by winrate to get best/2nd
                wrs_sorted = sorted(wrs, reverse=True)
                wr_best = wrs_sorted[0]
                wr_2nd  = wrs_sorted[1] if len(wrs_sorted) >= 2 else wr_best
                gap12   = wr_best - wr_2nd

                # Range and std over available wrs
                rng_topk = max(wrs) - min(wrs)
                std_topk = pstdev(wrs) if len(wrs) > 1 else 0.0

                # sl_best: take scoreLead of best move (align by index of best)
                # We need the index of best move in original list (first occurrence)
                best_idx = None
                best_val = -1.0
                for idx, mi in enumerate(mid[:10]):
                    wr = to_float(mi.get("winrate"))
                    if wr is None:
                        continue
                    if wr > best_val:
                        best_val = wr
                        best_idx = idx
                sl_best = None
                if best_idx is not None:
                    sl_best = to_float(mid[:10][best_idx].get("scoreLead"))

                rec = {
                    "id": js.get("id"),
                    "wr_best": wr_best,
                    "wr_2nd": wr_2nd,
                    "gap12": gap12,
                    "rng_topk": rng_topk,
                    "std_topk": std_topk,
                    "sl_best": sl_best,
                    "n_cands": len(wrs)
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
            except Exception:
                n_skip += 1

    print(f"read={n_in}, wrote metrics={n_out}, skipped={n_skip} -> {PATH_METRICS}")

if __name__ == "__main__":
    main()
