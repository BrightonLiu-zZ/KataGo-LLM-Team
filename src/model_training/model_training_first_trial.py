# Copyright 2020-2025 The HuggingFace Team. 
# Modified for: KataGo × LLM (Pure Offline GRPO) training
#
# This script turns the generic GRPO example into a Go-specific,
# fully-offline training loop using precomputed KataGo evaluations.
#
# -------------------------
# How to run (example)
# -------------------------
# 1) Prepare an offline dataset file (JSONL or Parquet) with columns:
#    - "user_prompt": string. Text/ASCII board + task instruction for the model.
#    - "board_key": string. Unique key for the position (hash of SGF or similar).
#    - "katago_all": dict (move -> winrate float 0~1). Precomputed by your offline pipeline.
#    - "katago_best": float 0~1. Best winrate for current position (optional but recommended).
#    NOTE: Keep your move labels consistent with what the model will output in "MOVE: <...>".
#
# 2) Set environment variables to point at your offline data file:
#    export OFFLINE_DATASET=/path/to/offline_go_train.jsonl   # JSONL with one JSON object per line
#
# 3) Launch training with Qwen2.5-7B-Instruct + LoRA + GRPO:
#    accelerate launch \
#      --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
#      this_script.py \
#      --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
#      --output_dir grpo-qwen2.5-7b-go-offline \
#      --learning_rate 1e-5 \
#      --torch_dtype bfloat16 \
#      --gradient_checkpointing \
#      --max_prompt_length 1024 \
#      --max_completion_length 512 \
#      --use_peft \
#      --lora_target_modules "q_proj","k_proj","v_proj","o_proj" \
#      --per_device_train_batch_size 1 \
#      --gradient_accumulation_steps 4 \
#      --num_generations 4 \
#      --logging_steps 10
#
# -------------------------
# Dependencies (install)
# -------------------------
# pip install "trl @ git+https://github.com/huggingface/trl.git"
# pip install peft datasets accelerate
# (optional) pip install trackio
#
# Notes:
# - We removed math-related deps (math-verify, latex2sympy2_extended) and all image code.
# - We do NOT call KataGo online during training — rewards read from offline columns only.
# - Rewards are easy to change in `go_offline_reward()` below (e.g., use wr, or wr-best).
# - This script expects the model to output:
#     <think>...</think>
#     MOVE: C4
#     EXPLAIN: ...
#   Keep your system prompt strict so we can parse MOVE reliably.

import os
import re
import json
from typing import List, Dict, Any

import torch
from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward  # optional; we also implement our own small format check

# Optional: enable basic logging in a Hugging Face Space (harmless otherwise)
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# -------------------------
# Configurable constants
# -------------------------

# Strict system prompt to force well-structured output.
SYSTEM_PROMPT = (
    "You are a Go (Weiqi) assistant for 9x9 boards. "
    "Always first reason inside <think>...</think>, then output the final move and a short explanation.\n\n"
    "STRICT OUTPUT FORMAT (do not add extra text before or after):\n"
    "<think>your chain-of-thought (brief, bullet-style)</think>\n"
    "MOVE: <coordinate like C4>\n"
    "EXPLAIN: <one sentence why this move>\n"
)

# Regex used to extract the MOVE line; keep the label exactly "MOVE:".
# NOTE: We don't enforce a specific coordinate system here; we just read the string after "MOVE:".
MOVE_PAT = re.compile(r"(?i)^\s*MOVE:\s*([A-Za-z0-9]+)\s*$", re.M)

# Reward shaping weights (you can tweak safely)
FORMAT_BONUS = 0.03        # tiny bonus if <think>...</think> appears
MISSING_MOVE_PENALTY = -0.2  # penalty if we fail to parse a MOVE line
UNKNOWN_MOVE_PENALTY = -0.1  # penalty if move is not found in katago_all dict

# -------------------------
# Utility functions
# -------------------------

def extract_move(text: str) -> str:
    """
    Extract the move string from the model output. 
    We accept anything that appears after "MOVE:"; coordinate validation is not performed here.
    Return None if we can't find a MOVE line.
    """
    m = MOVE_PAT.search(text)
    return m.group(1).strip().upper() if m else None


def ensure_dict(obj: Any) -> Dict[str, float]:
    """
    HF 'datasets' may load a JSON field as dict already; if it's a string, parse it.
    The dict must be: { "C4": 0.53, "D3": 0.41, ... } — winrates in [0,1].
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# -------------------------
# Reward function (pure offline)
# -------------------------

def go_offline_reward(
    completions: List[List[Dict[str, str]]],
    katago_all: List[Any],
    katago_best: List[Any] = None,
    **kwargs,
) -> List[float]:
    """
    Offline reward function for Go.
    Inputs (per-sample):
      - completions: TRL hands us a list of generated candidates; we only need the text of the first (best) for reward.
      - katago_all: dict (move -> winrate) for the current position, fully offline precomputed.
      - katago_best: scalar best winrate for the position (optional, else we use max(katago_all.values())).
    Output:
      - rewards: list[float] with one reward per sample.
    Recommended reward:
      wr(move) - best_wr + format_bonus
      (wr, best_wr are in [0,1]; difference in [-1,1]; bonus is small)
    """
    rewards: List[float] = []
    contents = [c[0]["content"] for c in completions]  # TRL packs candidates as [[{"role":..., "content": ...}], ...]

    for idx, content in enumerate(contents):
        # (1) Parse MOVE and format bonus
        move = extract_move(content)
        has_think = ("<think>" in content) and ("</think>" in content)
        fmt_bonus = FORMAT_BONUS if has_think else 0.0

        # (2) Read katago tables from dataset
        table = ensure_dict(katago_all[idx])
        if not table:
            # This sample has no offline data — give small negative signal to quickly skip it
            rewards.append(-0.05)
            continue

        # Compute best winrate (from column if provided, else from dict)
        if katago_best is not None and len(katago_best) > idx and katago_best[idx] is not None:
            try:
                best_wr = float(katago_best[idx])
            except Exception:
                best_wr = max(table.values()) if len(table) else 0.0
        else:
            best_wr = max(table.values()) if len(table) else 0.0

        if move is None:
            # No MOVE parsed — penalize a bit but not too harsh (model will quickly learn to include it)
            rewards.append(MISSING_MOVE_PENALTY)
            continue

        # (3) Look up the move's offline winrate
        wr = table.get(move)
        if wr is None:
            # Unknown move (not in offline dict) — slight penalty
            rewards.append(UNKNOWN_MOVE_PENALTY)
            continue

        # (4) Compose final reward
        #     Main term: advantage over best (often negative unless equal to best);
        #     If you prefer absolute strength, you can use just wr instead.
        advantage = float(wr) - float(best_wr)
        reward_val = advantage + fmt_bonus
        rewards.append(float(reward_val))

    return rewards


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    # Parse standard TRL args (ScriptArguments, GRPOConfig, ModelConfig)
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # -------------------------
    # Model & dtype / quant
    # -------------------------
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # -------------------------
    # Load OFFLINE dataset
    # -------------------------
    # Expect a single file with all samples; we split into train/test via train_test_split.
    # Supported formats: JSONL (recommended) or Parquet — choose via file extension.
    offline_path = os.environ.get("OFFLINE_DATASET", "data/offline_go_train.jsonl")
    if not os.path.exists(offline_path):
        raise FileNotFoundError(
            f"OFFLINE_DATASET not found: {offline_path}\n"
            "Please export OFFLINE_DATASET=/path/to/offline_go_train.jsonl"
        )

    if offline_path.endswith(".jsonl") or offline_path.endswith(".json"):
        dataset = load_dataset("json", data_files=offline_path, split="train")
    elif offline_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=offline_path, split="train")
    else:
        raise ValueError("Unsupported dataset file type. Use .jsonl/.json or .parquet")

    # Minimal validation of columns; you can add more checks here.
    required_cols = ["user_prompt", "board_key", "katago_all"]
    for col in required_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset missing required column: {col}")

    # Split (hold out a tiny test set if eval is enabled)
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    # Wrap prompts into Chat format (system + user) expected by TRL/Chat models
    def build_chat(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["user_prompt"]},
        ]
        example["prompt"] = prompt
        return example

    train_dataset = train_dataset.map(build_chat)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(build_chat)

    # -------------------------
    # Reward functions
    # -------------------------
    # You can include `think_format_reward` as a tiny, general formatting reward.
    # But the main signal comes from `go_offline_reward` which reads offline KataGo winrates.
    reward_functions = [go_offline_reward]  # or [think_format_reward, go_offline_reward]

    # -------------------------
    # Trainer (GRPO)
    # -------------------------
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,     # e.g., "Qwen/Qwen2.5-7B-Instruct"
        args=training_args,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),  # LoRA/QLoRA config from CLI
    )

    # Kick off training
    trainer.train()

    # Save artifacts
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        # Reuse script_args.dataset_name if you want; otherwise, you can pass a custom string.
        trainer.push_to_hub(dataset_name=getattr(script_args, "dataset_name", "offline_go"))
