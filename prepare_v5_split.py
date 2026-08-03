"""
Build the exp05 (v5) train / held-out eval split from the v4 dataset.

Leakage rule: the v4 dataset is augmented (each source position appears in up
to 16 symmetry/colour variants, sharing the source-game hash prefix of
`original_id`). Splitting row-wise would leak mirror images of eval positions
into training, so the split is by SOURCE GAME:

  - held-out games are chosen deterministically (seed 42) until their identity
    records (`_aug0`, no `_inv`) reach ~EVAL_TARGET positions;
  - the eval set contains only those identity records (one canonical
    orientation per position);
  - the train set drops ALL variants of the held-out games.

Run from the repository root:

    python prepare_v5_split.py
"""
import json
import random
from collections import defaultdict

INPUT_FILE = "data/training_data_v4.jsonl"
TRAIN_OUT = "data/training_data_v5_train.jsonl"
EVAL_OUT = "data/eval_positions_v5.jsonl"
EVAL_TARGET = 500
SEED = 42


def game_id(original_id: str) -> str:
    return original_id.split("_pos")[0]


def is_identity(original_id: str) -> bool:
    return original_id.endswith("_aug0")


def main():
    print(f"Reading {INPUT_FILE} ...")
    identity_count = defaultdict(int)
    total = 0
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            if is_identity(rec["original_id"]):
                identity_count[game_id(rec["original_id"])] += 1

    games = sorted(identity_count)
    print(f"{total} rows, {len(games)} source games, "
          f"{sum(identity_count.values())} identity records")

    rng = random.Random(SEED)
    rng.shuffle(games)
    heldout, n_eval = set(), 0
    for g in games:
        if n_eval >= EVAL_TARGET:
            break
        heldout.add(g)
        n_eval += identity_count[g]
    print(f"held out {len(heldout)} games -> {n_eval} eval positions")

    n_train = n_eval_written = n_dropped = 0
    with open(INPUT_FILE, encoding="utf-8") as fin, \
         open(TRAIN_OUT, "w", encoding="utf-8") as ftrain, \
         open(EVAL_OUT, "w", encoding="utf-8") as feval:
        for line in fin:
            rec = json.loads(line)
            gid = game_id(rec["original_id"])
            if gid in heldout:
                if is_identity(rec["original_id"]):
                    feval.write(line)
                    n_eval_written += 1
                else:
                    n_dropped += 1  # augmented variant of a held-out game
            else:
                ftrain.write(line)
                n_train += 1

    print(f"train: {n_train} rows -> {TRAIN_OUT}")
    print(f"eval:  {n_eval_written} rows -> {EVAL_OUT}")
    print(f"dropped {n_dropped} augmented variants of held-out games "
          f"(leakage prevention)")


if __name__ == "__main__":
    main()
