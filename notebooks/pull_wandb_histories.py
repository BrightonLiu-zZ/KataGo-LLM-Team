"""Re-pull full W&B histories for exp01-04 into notebooks/data/*.parquet.

Run from the repo root:  python notebooks/pull_wandb_histories.py
Requires W&B credentials in ~/.netrc and `pip install wandb pandas pyarrow`.
NOTE: run it from a directory whose cwd does not contain the local `wandb/`
run directory on sys.path shadowing the wandb package (repo root is fine
because wandb/ has no __init__.py, but keep this in mind if imports fail).
"""
import json
import os

import pandas as pd
import wandb

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

ENTITY_PROJECT = "brighton_zz-uc-san-diego/huggingface"
RUNS = {
    "exp01": ("vs89r50o", "hardy-river-3"),
    "exp02": ("h0va9zlw", "true-darkness-4"),
    "exp03": ("jwcpfh4t", "drawn-meadow-5"),
    "exp04": ("cjteb9re", "fiery-firebrand-6"),
}

api = wandb.Api(timeout=60)
configs = {}

for exp, (rid, name) in RUNS.items():
    run = api.run(f"{ENTITY_PROJECT}/{rid}")
    keys = [k for k in run.summary.keys()
            if k.startswith("train") and "profiling" not in k]
    rows = list(run.scan_history(keys=keys + ["_step"]))
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/{exp}_{name}.parquet")
    print(f"{exp} ({name}): {len(df)} rows saved")
    configs[exp] = {
        "wandb_name": name, "wandb_id": rid, "state": run.state,
        "created": str(run.createdAt),
        **{k: v for k, v in run.config.items() if not isinstance(v, (dict, list))},
    }

with open(f"{OUT}/run_configs.json", "w") as f:
    json.dump(configs, f, indent=2, default=str)
print("run_configs.json saved")
