import json, random
from pathlib import Path

# ---- fixed paths ----
PATH_METRICS   = Path(r"D:\katago_old\lizzie\metrics.jsonl")
PATH_DECISIONS = Path(r"D:\katago_old\lizzie\decisions.jsonl")
PATH_KEEPIDS   = Path(r"D:\katago_old\lizzie\keep_ids.txt")

# ---- thresholds (feel free to tweak) ----
WR_BLOWOUT_HI = 0.98   # D1
WR_BLOWOUT_LO = 0.02
SL_EXTREME    = 15.0   # D2
RANGE_FLAT    = 0.02   # D3
STD_FLAT      = 0.02   # D4
WR_BIG_LEAD   = 0.90   # used by D3/D4
WR_BIG_TRAIL  = 0.10

GAP12_KEY     = 0.15   # K1
RANGE_KEY     = 0.10   # K2
WR_BAL_LO     = 0.35   # K2 window
WR_BAL_HI     = 0.65

KEEP_PROB_BLOWOUT = 0.08  # keep 5~10% blowouts
WEIGHT_NORMAL     = 1.0
WEIGHT_KEY        = 1.5
WEIGHT_BLOWOUT    = 0.2   # if you prefer downweight instead of drop, we still add weight

RANDOM_SEED = 42

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    rng = random.Random(RANDOM_SEED)
    n_in = n_keep = 0
    with PATH_METRICS.open("r", encoding="utf-8") as fin, \
         PATH_DECISIONS.open("w", encoding="utf-8") as fdec, \
         PATH_KEEPIDS.open("w", encoding="utf-8") as fids:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            js = json.loads(line)

            wr_best  = to_float(js.get("wr_best"))
            wr_2nd   = to_float(js.get("wr_2nd"))
            gap12    = to_float(js.get("gap12"))
            rng_topk = to_float(js.get("rng_topk"))
            std_topk = to_float(js.get("std_topk"))
            sl_best  = to_float(js.get("sl_best"))

            # ---- D-rules: blowout ----
            is_blowout = False
            if wr_best is not None and (wr_best >= WR_BLOWOUT_HI or wr_best <= WR_BLOWOUT_LO):
                is_blowout = True  # D1
            if sl_best is not None and abs(sl_best) >= SL_EXTREME:
                is_blowout = True  # D2
            if (rng_topk is not None and rng_topk <= RANGE_FLAT) and \
               (wr_best is not None and (wr_best >= WR_BIG_LEAD or wr_best <= WR_BIG_TRAIL)):
                is_blowout = True  # D3
            if (std_topk is not None and std_topk <= STD_FLAT) and \
               (wr_best is not None and (wr_best >= WR_BIG_LEAD or wr_best <= WR_BIG_TRAIL)):
                is_blowout = True  # D4

            # ---- K-rules: key positions ----
            is_key = False
            if gap12 is not None and gap12 >= GAP12_KEY:
                is_key = True  # K1
            if (rng_topk is not None and rng_topk >= RANGE_KEY) and \
               (wr_best is not None and WR_BAL_LO <= wr_best <= WR_BAL_HI):
                is_key = True  # K2

            # ---- keep / weight policy ----
            keep = True
            weight = WEIGHT_NORMAL

            if is_key:
                weight = WEIGHT_KEY

            if is_blowout and not is_key:
                # keep only small fraction; otherwise drop
                if rng.random() < KEEP_PROB_BLOWOUT:
                    keep = True
                    weight = WEIGHT_BLOWOUT
                else:
                    keep = False

            out = {
                "id": js.get("id"),
                "is_blowout": is_blowout,
                "is_key": is_key,
                "keep": keep,
                "weight": weight,
                # keep metrics for debugging/tuning
                "wr_best": wr_best,
                "wr_2nd": wr_2nd,
                "gap12": gap12,
                "rng_topk": rng_topk,
                "std_topk": std_topk,
                "sl_best": sl_best
            }
            fdec.write(json.dumps(out, ensure_ascii=False) + "\n")
            if keep:
                fids.write(str(js.get("id")) + "\n")
                n_keep += 1

    print(f"metrics read={n_in}, kept={n_keep}, decisions -> {PATH_DECISIONS}, ids -> {PATH_KEEPIDS}")

if __name__ == "__main__":
    main()
