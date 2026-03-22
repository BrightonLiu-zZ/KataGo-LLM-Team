"""
Prepare v4 training dataset from the existing v3 augmented data.

Changes from v3 → v4:
  1. Restore 'Valid empty coordinates: [...]' list (stripped by augmentation)
  2. Add instruction line after 'Player to move:'
  3. Downsample lopsided positions (root_winrate >0.95 or <0.05) to keep 30%
  4. Shuffle

Does NOT re-augment — works on existing augmented data.
"""
import json
import random
import re
import os
import sys

INPUT_FILE = "data/training_data_augmented_shuffled_v3.jsonl"
OUTPUT_FILE = "data/training_data_v4.jsonl"

VALID_COLS = "ABCDEFGHJ"
LOPSIDED_KEEP_RATIO = 0.30
LOPSIDED_UPPER = 0.95
LOPSIDED_LOWER = 0.05

SEED = 42


def extract_empty_coords_from_board(prompt_text: str) -> list:
    """Parse the ASCII board embedded in the prompt and return sorted empty coordinates."""
    empty = []
    for line in prompt_text.split("\n"):
        m = re.match(r"^\s*([1-9])\s+((?:[.XO]\s+)*[.XO])\s*$", line)
        if not m:
            continue
        row_num = int(m.group(1))
        cells = m.group(2).split()
        for col_idx, cell in enumerate(cells):
            if col_idx < 9 and cell == ".":
                empty.append(f"{VALID_COLS[col_idx]}{row_num}")
    empty.sort(key=lambda c: (int(c[1:]), VALID_COLS.index(c[0])))
    return empty


def rebuild_prompt(prompt_text: str) -> str:
    """Insert instruction line and valid-coordinates list after 'Player to move:' line."""
    empty_coords = extract_empty_coords_from_board(prompt_text)
    coords_str = ", ".join(empty_coords)

    lines = prompt_text.split("\n")
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if "Player to move:" in line and not inserted:
            inserted = True
            out.append("Analyze the board, then pick a move and explain your reasoning.")
            out.append("")
            out.append(f"Valid empty coordinates: [{coords_str}]")
    return "\n".join(out)


def is_lopsided(root_winrate) -> bool:
    if root_winrate is None:
        return False
    wr = float(root_winrate)
    return wr > LOPSIDED_UPPER or wr < LOPSIDED_LOWER


def main():
    random.seed(SEED)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Lopsided threshold: >{LOPSIDED_UPPER} or <{LOPSIDED_LOWER}")
    print(f"Lopsided keep ratio: {LOPSIDED_KEEP_RATIO}")
    print()

    kept_lines = []
    total = 0
    n_lopsided = 0
    n_lopsided_kept = 0
    n_balanced = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            rwr = rec.get("root_winrate")

            if is_lopsided(rwr):
                n_lopsided += 1
                if random.random() > LOPSIDED_KEEP_RATIO:
                    continue
                n_lopsided_kept += 1
            else:
                n_balanced += 1

            rec["user_prompt"] = rebuild_prompt(rec["user_prompt"])
            kept_lines.append(json.dumps(rec, ensure_ascii=False))

            if (i + 1) % 50000 == 0:
                print(f"  Scanned {i+1} records, kept {len(kept_lines)} ...")

    print(f"\nScanned {total} records total.")
    print(f"  Lopsided: {n_lopsided}  (kept {n_lopsided_kept}, dropped {n_lopsided - n_lopsided_kept})")
    print(f"  Balanced: {n_balanced}  (kept all)")
    print(f"  Total kept: {len(kept_lines)}")

    print(f"\nShuffling {len(kept_lines)} records ...")
    random.shuffle(kept_lines)

    print(f"Writing to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    print(f"Done. Output: {OUTPUT_FILE} ({len(kept_lines)} records)")

    # Sanity check: verify a few records
    print("\n=== Sanity check (first 2 records) ===")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for j in range(2):
            rec = json.loads(f.readline())
            prompt = rec["user_prompt"]
            has_valid = "Valid empty coordinates:" in prompt
            has_instr = "Analyze the board" in prompt
            has_rwr = rec.get("root_winrate") is not None
            print(f"  Record {j+1}: valid_coords={has_valid}, instruction={has_instr}, root_wr={has_rwr}")
            # Show tail of prompt
            tail = prompt.split("Player to move:")[1] if "Player to move:" in prompt else "(no player line?)"
            print(f"    Prompt tail: ...Player to move:{tail[:200]}")
            print()


if __name__ == "__main__":
    main()
