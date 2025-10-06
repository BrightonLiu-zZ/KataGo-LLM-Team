# -*- coding: utf-8 -*-
"""
Merge decisions.jsonl into your Chinese training jsonl, keep only 'keep=True',
and attach Chinese metadata fields:
- 样本权重
- 是否一边倒的局面
- 是否关键局面

Input:
  training_data_zh.jsonl (built earlier)
  decisions.jsonl        (from label_and_select.py)
Output:
  training_data_zh.filtered.jsonl
"""

import json
from pathlib import Path

# ---- fixed paths ----
PATH_TRAIN_ZH = Path(r"D:\katago_old\lizzie\training_data_zh.jsonl")
PATH_DECISIONS = Path(r"D:\katago_old\lizzie\decisions.jsonl")
PATH_OUT = Path(r"D:\katago_old\lizzie\training_data_zh.filtered.jsonl")

def load_decisions(path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            m[d["id"]] = d
    return m

def main():
    decisions = load_decisions(PATH_DECISIONS)
    n_in = n_out = 0

    with PATH_TRAIN_ZH.open("r", encoding="utf-8") as fin, \
         PATH_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            js = json.loads(line)

            # '局面编号' is your Chinese id key
            rid = js.get("局面编号")
            if rid is None or rid not in decisions:
                continue

            dec = decisions[rid]
            if not dec.get("keep", False):
                continue  # drop

            # Attach Chinese metadata fields
            js["样本权重"] = float(dec.get("weight", 1.0))
            js["是否一边倒的局面"] = bool(dec.get("is_blowout", False))
            js["是否关键局面"] = bool(dec.get("is_key", False))

            fout.write(json.dumps(js, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read={n_in}, wrote(filtered)={n_out} -> {PATH_OUT}")

if __name__ == "__main__":
    main()
