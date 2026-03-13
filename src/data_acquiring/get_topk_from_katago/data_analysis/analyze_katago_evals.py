#!/usr/bin/env python3
"""
Analyze katago_evals fields (policy, winrate, scoreLead) in a JSONL file.
Produces a summary table and histograms for non-null values.
"""

import json
import sys
import os
from collections import defaultdict
import statistics

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not found — histograms will be skipped. Install with: pip install matplotlib")

JSONL_PATH = "data/json_output_with_topk_TEST.jsonl"
FIELDS = ["policy", "winrate", "scoreLead"]
OUTPUT_DIR = "."


def load_values(jsonl_path):
    """Extract non-null values for each field from all katago_evals entries."""
    values = defaultdict(list)
    total_records = 0
    total_eval_entries = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_records += 1
            katago_evals = record.get("katago_evals", {})
            if not isinstance(katago_evals, dict):
                continue
            for move_key, move_data in katago_evals.items():
                total_eval_entries += 1
                if not isinstance(move_data, dict):
                    continue
                for field in FIELDS:
                    val = move_data.get(field)
                    if val is not None:
                        values[field].append(val)

    return values, total_records, total_eval_entries


def summary_stats(vals):
    """Compute summary statistics for a list of numeric values."""
    if not vals:
        return {}
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    median = statistics.median(sorted_vals)
    stdev = statistics.stdev(sorted_vals) if n > 1 else 0.0
    min_v = sorted_vals[0]
    max_v = sorted_vals[-1]
    q1 = statistics.median(sorted_vals[:n // 2]) if n >= 2 else sorted_vals[0]
    q3 = statistics.median(sorted_vals[(n + 1) // 2:]) if n >= 2 else sorted_vals[0]
    return {
        "count (non-null)": n,
        "mean": mean,
        "median": median,
        "std": stdev,
        "min": min_v,
        "Q1 (25%)": q1,
        "Q3 (75%)": q3,
        "max": max_v,
    }


def print_summary_table(values, total_records, total_eval_entries):
    """Print a formatted summary table to stdout."""
    print("=" * 70)
    print(f"File: {JSONL_PATH}")
    print(f"Total game records: {total_records}")
    print(f"Total katago_evals move entries: {total_eval_entries}")
    print("=" * 70)

    all_stats = {}
    for field in FIELDS:
        all_stats[field] = summary_stats(values[field])

    # Determine column widths
    stat_keys = list(next(iter(all_stats.values())).keys())
    col_w = 18
    label_w = 20

    header = f"{'Statistic':<{label_w}}" + "".join(f"{f:>{col_w}}" for f in FIELDS)
    print(header)
    print("-" * (label_w + col_w * len(FIELDS)))

    for stat in stat_keys:
        row = f"{stat:<{label_w}}"
        for field in FIELDS:
            v = all_stats[field].get(stat, "N/A")
            if isinstance(v, int):
                row += f"{v:>{col_w},}"
            elif isinstance(v, float):
                row += f"{v:>{col_w}.6f}"
            else:
                row += f"{str(v):>{col_w}}"
        print(row)

    print("=" * 70)
    return all_stats


def plot_histograms(values, all_stats, 
                    #output_path="katago_evals_histograms.png"
                    output_path='src/data_acquiring/get_topk_from_katago/data_analysis/katago_evals_histograms.png'
                    ):
    """Generate a 3-panel histogram figure."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("KataGo Evals — Distribution of Non-Null Values", fontsize=14, fontweight="bold")

    colors = ["steelblue", "darkorange", "seagreen"]

    for ax, field, color in zip(axes, FIELDS, colors):
        vals = values[field]
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(field)
            continue

        stats = all_stats[field]
        n = stats["count (non-null)"]
        mean = stats["mean"]
        median = stats["median"]

        ax.hist(vals, bins=50, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean:.4f}")
        ax.axvline(median, color="gold", linestyle="-.", linewidth=1.5, label=f"median={median:.4f}")

        ax.set_title(f"{field}  (n={n:,})", fontsize=12)
        ax.set_xlabel(field)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, output_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nHistograms saved to: {save_path}")
    plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else JSONL_PATH

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    print(f"\nLoading {path} ...")
    values, total_records, total_eval_entries = load_values(path)

    print()
    all_stats = print_summary_table(values, total_records, total_eval_entries)

    plot_histograms(values, all_stats)


if __name__ == "__main__":
    main()
