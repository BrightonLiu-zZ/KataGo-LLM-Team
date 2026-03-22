"""
GRPO training script v4 — Qwen3-8B for 9×9 Go.

Changes from v3:
  - System prompt: removed <think> scaffold; model outputs REASONING + MOVE
  - Reward Layer 2 redesigned:
      R_policy:  log-scaled (smooth for non-top moves)
      R_rank:    percentile rank by scoreLead (works in lopsided positions)
      Adaptive weighting via root_winrate closeness
  - Valid-coordinates list restored in prompts (via prepare_v4_dataset.py)
  - Lopsided positions downsampled to 30% in dataset
  - max_completion_length reduced to 512
  - Fresh start from base model (no checkpoint)
"""
import math
import os
import random
import re
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ================= 1. Configuration =================

DATASET_PATH = "data/training_data_v4.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "runs/Qwen3-8B-GRPO-Go-Pro-v4"

VALID_COLS = set("ABCDEFGHJ")
VALID_ROWS = set(str(i) for i in range(1, 10))

SYSTEM_PROMPT = (
    "You are a 9x9 Go (Weiqi) player. "
    "Board notation: '.' = empty, 'X' = Black, 'O' = White. "
    "Columns: A-J (no I). Rows: 1 (bottom) to 9 (top). "
    "You MUST only play on an empty '.' intersection listed in the valid coordinates.\n\n"
    "Respond in exactly this format:\n"
    "REASONING: [2-4 sentences analyzing the position]\n"
    "MOVE: [coordinate, e.g. D4]"
)

# ================= 2. Utilities =================

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS":
        return True
    if len(coord) < 2:
        return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)


def extract_move(text: str) -> Optional[str]:
    match = re.search(r"MOVE:\s*([A-HJ-Z][1-9]|PASS)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None


def check_spatial_legality(prompt_text: str, move_str: str) -> float:
    """Returns 0.0 if move is on an empty '.', else a negative penalty."""
    if move_str == "PASS":
        return 0.0

    col_char, row_char = move_str[0], move_str[1]

    lines = prompt_text.split("\n")
    board_lines = []
    for line in lines:
        if re.match(r"^\s*[1-9]\s+([.XO]\s+)*[.XO]", line):
            board_lines.append(line)

    if len(board_lines) != 9:
        return -0.1

    row_idx = 9 - int(row_char)
    col_mapping = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
        "F": 5, "G": 6, "H": 7, "J": 8,
    }
    col_idx = col_mapping.get(col_char)
    if col_idx is None:
        return -0.5

    row_elements = board_lines[row_idx].strip().split()
    if len(row_elements) >= 10:
        target_spot = row_elements[col_idx + 1]
        if target_spot != ".":
            return -0.8

    return 0.0


# ================= 3. Reward Components =================

def _compute_r_policy_log(move: str, evals: dict) -> float:
    """
    Log-scaled policy reward ∈ [0, 1].
      policy=1e-4 → 0.0
      policy=1e-3 → 0.25
      policy=1e-2 → 0.50
      policy=1e-1 → 0.75
      policy=1.0  → 1.0
    """
    if move == "PASS":
        entry = evals.get("pass") or evals.get("PASS") or {}
    else:
        entry = evals.get(move, {})

    p = entry.get("policy")
    if p is None or p <= 0:
        return 0.0
    return max(0.0, min(1.0, (math.log10(max(p, 1e-6)) + 4.0) / 4.0))


def _compute_r_rank(move: str, evals: dict) -> float:
    """
    Percentile rank by scoreLead among all evaluated legal moves.
      Best move  → 1.0
      Median     → 0.5
      Worst move → 0.0
    Moves not in the eval dict get 0.0 (below all evaluated moves).
    """
    scored = []
    for coord, metrics in evals.items():
        if not isinstance(metrics, dict):
            continue
        sl = metrics.get("scoreLead")
        pol = metrics.get("policy", -1)
        if sl is not None and pol >= 0:
            scored.append((coord.upper(), float(sl)))

    if not scored:
        return 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    n = len(scored)

    move_upper = move.upper() if move != "PASS" else "PASS"
    for rank, (coord, _) in enumerate(scored):
        if coord == move_upper:
            return 1.0 - rank / max(n - 1, 1)

    return 0.0


# ================= 4. Reward Functions =================

# --- Layer 1: Format & Spatial Legality (gate) ---

def format_and_legality_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Gate layer with partial credit.
      0.0   valid move on empty square
     -0.5   MOVE: found but coord invalid or on occupied stone
     -1.0   no MOVE: found (truncated / wrong format)
    """
    rewards = []
    for prompt, content in zip(prompts, completions):
        move_str = extract_move(content)
        if not move_str:
            rewards.append(-1.0)
        elif check_spatial_legality(prompt, move_str) < 0:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


# --- Layer 2: KataGo Composite (core — redesigned) ---

def katago_composite_reward_func(
    prompts, completions, katago_evals, root_winrate, **kwargs
) -> List[float]:
    """
    Core reward using log-scaled policy + percentile rank.

    Adaptive weighting based on position closeness (root_winrate):
      Close game  → heavier rank weight  (getting the right move matters)
      Lopsided    → heavier policy weight (just play natural moves)

    Returns ∈ [0, 1] for valid moves; 0.0 for invalid (gate already penalized).
    """
    rewards = []

    for prompt, content, evals, rwr in zip(
        prompts, completions, katago_evals, root_winrate
    ):
        if not isinstance(evals, dict):
            evals = {}

        move = extract_move(content)
        if not move or check_spatial_legality(prompt, move) < 0:
            rewards.append(0.0)
            continue

        r_policy = _compute_r_policy_log(move, evals)
        r_rank = _compute_r_rank(move, evals)

        rwr_val = float(rwr) if rwr is not None else 0.5
        closeness = 1.0 - 2.0 * abs(rwr_val - 0.5)  # 0=lopsided, 1=even

        w_rank = 0.5 + 0.3 * closeness    # 0.5 (lopsided) → 0.8 (close)
        w_policy = 1.0 - w_rank            # 0.5 (lopsided) → 0.2 (close)

        composite = w_policy * r_policy + w_rank * r_rank
        rewards.append(composite)

    return rewards


# --- Layer 3: Logging (non-scoring) ---

LOG_FILE = "src/model_training/logs/grpo_training_rollouts_v4.jsonl"


def logging_reward_func(
    prompts, completions, katago_evals, root_winrate, **kwargs
) -> List[float]:
    zero_rewards = [0.0] * len(completions)

    if random.random() < 1.0:
        idx = random.randint(0, len(completions) - 1)
        prompt_text = prompts[idx]
        content = completions[idx]
        evals = katago_evals[idx] if isinstance(katago_evals[idx], dict) else {}
        rwr = root_winrate[idx] if root_winrate else None

        move = extract_move(content)

        gate = 0.0
        if not move:
            gate = -1.0
        elif check_spatial_legality(prompt_text, move) < 0:
            gate = -0.5

        r_policy = _compute_r_policy_log(move, evals) if move and gate == 0.0 else 0.0
        r_rank = _compute_r_rank(move, evals) if move and gate == 0.0 else 0.0

        rwr_val = float(rwr) if rwr is not None else 0.5
        closeness = 1.0 - 2.0 * abs(rwr_val - 0.5)
        w_rank = 0.5 + 0.3 * closeness
        w_policy = 1.0 - w_rank
        composite = w_policy * r_policy + w_rank * r_rank if gate == 0.0 else 0.0

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "board_state_prompt": prompt_text,
            "model_generation": content,
            "extracted_move": move,
            "root_winrate": rwr_val,
            "closeness": round(closeness, 4),
            "katago_evals_sample": {k: v for k, v in list(evals.items())[:5]},
            "reward_breakdown": {
                "gate": gate,
                "r_policy_log": round(r_policy, 6),
                "r_rank": round(r_rank, 6),
                "w_policy": round(w_policy, 4),
                "w_rank": round(w_rank, 4),
                "composite": round(composite, 6),
                "total": round(gate + composite, 6),
            },
        }

        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return zero_rewards


# ================= 5. Main =================

def main():
    print(f"Loading dataset from {DATASET_PATH} ...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        print("Run prepare_v4_dataset.py first.")
        return

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        formatted_prompts = []
        for user_p in examples["user_prompt"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}

    dataset = dataset.map(preprocess_function, batched=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.02,
        lr_scheduler_type="constant_with_warmup",
        logging_steps=10,
        bf16=True,
        per_device_train_batch_size=16,
        num_generations=8,
        gradient_accumulation_steps=4,
        max_completion_length=512,
        save_steps=500,
        max_steps=270000,
        report_to="wandb",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            format_and_legality_reward_func,
            katago_composite_reward_func,
            logging_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print(f"Starting GRPO training (v4) — model: {MODEL_NAME} ...")
    print(f"  Dataset: {DATASET_PATH}  ({len(dataset)} examples)")
    print(f"  vLLM: {training_args.use_vllm}")
    print(f"  Batch: {training_args.per_device_train_batch_size}  "
          f"Generations: {training_args.num_generations}")
    print(f"  max_completion_length: {training_args.max_completion_length}")

    trainer.train()

    print(f"Training done. Saving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
