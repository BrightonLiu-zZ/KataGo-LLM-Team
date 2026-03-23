"""
Analyze the root_winrate distribution in the training data.
Answers: what fraction of positions are lopsided (one side clearly winning)?
"""
import json
import sys

DATA_FILE = "data/training_ready_data_v3.jsonl"

def main():
    winrates = []
    score_leads = []
    missing = 0

    print(f"Reading {DATA_FILE} ...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            wr = rec.get("root_winrate")
            sl = rec.get("root_scoreLead")
            if wr is not None:
                winrates.append(float(wr))
            else:
                missing += 1
            if sl is not None:
                score_leads.append(float(sl))

    N = len(winrates)
    print(f"\nTotal records: {N + missing}  (with root_winrate: {N}, missing: {missing})")

    if not winrates:
        print("No root_winrate data found.")
        return

    winrates.sort()

    # --- Percentiles ---
    def pct(arr, p):
        idx = int(len(arr) * p / 100)
        return arr[min(idx, len(arr) - 1)]

    print("\n=== root_winrate distribution ===")
    print(f"  Min:    {winrates[0]:.6f}")
    print(f"  P5:     {pct(winrates, 5):.6f}")
    print(f"  P10:    {pct(winrates, 10):.6f}")
    print(f"  P25:    {pct(winrates, 25):.6f}")
    print(f"  Median: {pct(winrates, 50):.6f}")
    print(f"  P75:    {pct(winrates, 75):.6f}")
    print(f"  P90:    {pct(winrates, 90):.6f}")
    print(f"  P95:    {pct(winrates, 95):.6f}")
    print(f"  Max:    {winrates[-1]:.6f}")
    print(f"  Mean:   {sum(winrates)/N:.6f}")

    # --- Lopsided-position analysis ---
    # "Lopsided" = one side has >95% winrate (or <5%)
    lopsided_95 = sum(1 for w in winrates if w > 0.95 or w < 0.05)
    lopsided_90 = sum(1 for w in winrates if w > 0.90 or w < 0.10)
    lopsided_80 = sum(1 for w in winrates if w > 0.80 or w < 0.20)
    close = sum(1 for w in winrates if 0.35 <= w <= 0.65)

    print("\n=== Position balance breakdown ===")
    print(f"  Lopsided (>95% or <5%):  {lopsided_95:>6d}  ({100*lopsided_95/N:.1f}%)")
    print(f"  Lopsided (>90% or <10%): {lopsided_90:>6d}  ({100*lopsided_90/N:.1f}%)")
    print(f"  Lopsided (>80% or <20%): {lopsided_80:>6d}  ({100*lopsided_80/N:.1f}%)")
    print(f"  Close    (35%-65%):       {close:>6d}  ({100*close/N:.1f}%)")

    # --- Histogram (text-based) ---
    bins = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.001]
    labels = [
        "0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.35",
        "0.35-0.50", "0.50-0.65", "0.65-0.80", "0.80-0.90",
        "0.90-0.95", "0.95-1.00"
    ]
    counts = [0] * (len(bins) - 1)
    for w in winrates:
        for i in range(len(bins) - 1):
            if bins[i] <= w < bins[i + 1]:
                counts[i] += 1
                break

    max_count = max(counts) if counts else 1
    bar_width = 40

    print("\n=== Histogram of root_winrate ===")
    for label, count in zip(labels, counts):
        bar = "█" * int(bar_width * count / max_count)
        print(f"  {label} | {bar} {count} ({100*count/N:.1f}%)")

    # --- scoreLead summary ---
    if score_leads:
        score_leads.sort()
        M = len(score_leads)
        print(f"\n=== root_scoreLead distribution ===")
        print(f"  Min:    {score_leads[0]:.2f}")
        print(f"  P10:    {pct(score_leads, 10):.2f}")
        print(f"  Median: {pct(score_leads, 50):.2f}")
        print(f"  P90:    {pct(score_leads, 90):.2f}")
        print(f"  Max:    {score_leads[-1]:.2f}")
        print(f"  Mean:   {sum(score_leads)/M:.2f}")

    # --- KataGo eval coverage check (sample 100 records) ---
    print("\n=== KataGo eval coverage (sampled from first 100 records) ===")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        sample_stats = {"has_wr": 0, "no_wr": 0, "has_sl": 0, "no_sl": 0, "total_moves": 0}
        for i, line in enumerate(f):
            if i >= 100:
                break
            rec = json.loads(line)
            evals = rec.get("katago_evals", {})
            for coord, metrics in evals.items():
                if not isinstance(metrics, dict):
                    continue
                if metrics.get("policy", -1) < 0:
                    continue  # occupied square
                sample_stats["total_moves"] += 1
                if metrics.get("winrate") is not None:
                    sample_stats["has_wr"] += 1
                else:
                    sample_stats["no_wr"] += 1
                if metrics.get("scoreLead") is not None:
                    sample_stats["has_sl"] += 1
                else:
                    sample_stats["no_sl"] += 1

        total = sample_stats["total_moves"]
        print(f"  Total legal-move entries sampled: {total}")
        print(f"  With winrate:    {sample_stats['has_wr']} ({100*sample_stats['has_wr']/total:.1f}%)")
        print(f"  Without winrate: {sample_stats['no_wr']} ({100*sample_stats['no_wr']/total:.1f}%)")
        print(f"  With scoreLead:    {sample_stats['has_sl']} ({100*sample_stats['has_sl']/total:.1f}%)")
        print(f"  Without scoreLead: {sample_stats['no_sl']} ({100*sample_stats['no_sl']/total:.1f}%)")


if __name__ == "__main__":
    main()
