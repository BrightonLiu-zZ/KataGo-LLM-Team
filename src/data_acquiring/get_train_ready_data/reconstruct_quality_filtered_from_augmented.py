"""
Reconstruct data/training_data_quality_filtered_v3.jsonl from the augmented dataset.

Why this script exists
----------------------
The 2026-05-06 incident destroyed every intermediate file in data/ (see
docs/RECOVERY-2026-05-06.md). The only dataset that survived was
`training_data_augmented_v3.jsonl`, which had been archived to HuggingFace
(brightonliuzZ/katago-grpo-datasets).

That file is a 16x augmentation of `training_data_quality_filtered_v3.jsonl`,
produced by `data_augmentation.py`: 8 D4 symmetries x 2 colour assignments. The
augmenter tags every derived record with

    new_rec['original_id'] = f"{orig_id}_aug{op}" + ("_inv" if color_flip else "")

so the identity transform (op == 0, colour NOT flipped) carries the suffix
`_aug0` and is byte-for-byte equivalent to its source record apart from that
suffix and a re-joined prompt string. Selecting those records and stripping the
suffix therefore recovers the quality-filtered dataset exactly.

What this does NOT recover
--------------------------
`training_ready_data_v3.jsonl` (18,493 records) and
`json_output_with_topk_official.jsonl` (the raw KataGo per-move analysis) are
gone for good. `data_quality_cut.py` kept only positions with move number >= 4
(parsed from the `_pos<N>` suffix of `original_id`), so the ~1,618 dropped
opening positions are not present in the augmented file and cannot be restored.

Usage
-----
    python src/data_acquiring/get_train_ready_data/reconstruct_quality_filtered_from_augmented.py
"""

import json
import os

INPUT_FILE = "data/training_data_augmented_v3.jsonl"
OUTPUT_FILE = "data/training_data_quality_filtered_v3.jsonl"

# The suffix data_augmentation.py appends for op=0 without a colour flip.
IDENTITY_SUFFIX = "_aug0"


def reconstruct() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("Download it first:")
        print("  hf download brightonliuzZ/katago-grpo-datasets "
              "training_data_augmented_v3.jsonl --repo-type=dataset --local-dir ./data")
        return

    total = 0
    kept = 0
    seen_ids = set()
    repeated_ids = set()

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)

            aug_id = record.get("original_id", "")
            # Identity transform only: op == 0 and no colour flip.
            # "_aug0_inv" is the colour-flipped twin and must be excluded.
            if not aug_id.endswith(IDENTITY_SUFFIX):
                continue

            source_id = aug_id[: -len(IDENTITY_SUFFIX)]
            # Do NOT de-duplicate. The source dataset genuinely contains a
            # repeated original_id, so dropping the second copy would make the
            # reconstruction one record short of the real file.
            if source_id in seen_ids:
                repeated_ids.add(source_id)
            seen_ids.add(source_id)

            record["original_id"] = source_id
            fout.write(json.dumps(record) + "\n")
            kept += 1

    print(f"Scanned {total} augmented records.")
    print(f"Recovered {kept} source records -> {OUTPUT_FILE}")
    if repeated_ids:
        print(f"Note: {len(repeated_ids)} original_id value(s) appear more than "
              f"once in the source data and were kept as-is: "
              f"{sorted(repeated_ids)}")
    if total and kept:
        print(f"Augmentation factor observed: {total / kept:.2f}x (expected 16.00x)")


if __name__ == "__main__":
    reconstruct()
