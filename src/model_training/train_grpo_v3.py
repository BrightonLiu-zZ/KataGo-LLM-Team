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

# ================= 1. 配置区域 (Configuration) =================
DATASET_PATH = "data/training_data_augmented_shuffled_v3.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "runs/Qwen3-8B-GRPO-Go-Pro-v3"

# 9x9 围棋坐标系 (不包含 'I')
VALID_COLS = set("ABCDEFGHJ")
VALID_ROWS = set(str(i) for i in range(1, 10))

SYSTEM_PROMPT = (
    "You are a 9x9 Go (Weiqi) player. "
    "Board notation: '.' = empty, 'X' = Black, 'O' = White. "
    "Columns: A-J (no I). Rows: 1 (bottom) to 9 (top). "
    "You MUST only play on an empty '.' intersection.\n\n"
    "<think>\n"
    "Step 1: Read the opponent's last move. What does it threaten — a cut, a capture, territory expansion?\n"
    "Step 2: Identify 2-3 candidate responses. For each, describe what it connects, cuts, or defends.\n"
    "Step 3: Pick your move. Read the board at that coordinate and confirm it shows '.' (empty).\n"
    "</think>\n"
    "MOVE: [Coordinate]\n"
    "EXPLAIN: [2-4 sentences]"
)

# ================= 2. 核心辅助函数 (Utilities) =================

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS": return True
    if len(coord) < 2: return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)

def extract_move(text: str) -> Optional[str]:
    match = re.search(r"MOVE:\s*([A-HJ-Z][1-9]|PASS)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None

def extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def check_spatial_legality(prompt_text: str, move_str: str) -> float:
    """
    Returns 0.0 if move is on an empty '.', else a negative penalty.
    """
    if move_str == "PASS":
        return 0.0

    col_char, row_char = move_str[0], move_str[1]

    lines = prompt_text.split('\n')
    board_lines = []
    for line in lines:
        if re.match(r"^\s*[1-9]\s+([.XO]\s+)*[.XO]", line):
            board_lines.append(line)

    if len(board_lines) != 9:
        return -0.1

    row_idx = 9 - int(row_char)
    col_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}
    col_idx = col_mapping.get(col_char)

    if col_idx is None:
        return -0.5

    row_elements = board_lines[row_idx].strip().split()
    if len(row_elements) >= 10:
        target_spot = row_elements[col_idx + 1]
        if target_spot != '.':
            return -0.8

    return 0.0


# ================= 3. 奖励函数群 (Reward Engineering) =================

# ---------- 辅助：从 katago_evals 计算各分量 ----------

def _compute_r_policy(move: str, evals: dict) -> float:
    """
    R_policy = prior probability in [0, 1].
    If move not in evals or has no policy, return 0.
    """
    if move == "PASS":
        entry = evals.get("pass") or evals.get("PASS") or {}
    else:
        entry = evals.get(move, {})
    p = entry.get("policy")
    if p is None or p < 0:   # negative policy = KataGo "forbidden" sentinel
        return 0.0
    return float(p)


def _compute_r_score(move: str, evals: dict) -> float:
    """
    R_score ∈ [-1, +1].
    Formula: ((max(ΔS, -8) + 8) / 4) - 1
      ΔS = 0  → +1.0   (perfect match with best move)
      ΔS = -4 →  0.0
      ΔS ≤ -8 → -1.0   (terrible or missing data)
    """
    valid_scores = [v["scoreLead"] for v in evals.values()
                    if isinstance(v, dict) and v.get("scoreLead") is not None]
    if not valid_scores:
        return -1.0

    best_score = max(valid_scores)

    if move == "PASS":
        entry = evals.get("pass") or evals.get("PASS") or {}
    else:
        entry = evals.get(move, {})

    chosen_score = entry.get("scoreLead")
    if chosen_score is None:
        # Move not analyzed by KataGo — impute worst case
        return -1.0

    delta_s = float(chosen_score) - float(best_score)
    r = (max(delta_s, -8.0) + 8.0) / 4.0 - 1.0
    return max(-1.0, min(1.0, r))


def _compute_r_winrate(move: str, evals: dict) -> float:
    """
    R_winrate ∈ [-1, +1].
    Formula: 1.0 + 2 × ΔW   where ΔW = chosen_wr - best_wr ∈ [-1, 0]
      ΔW =  0   → +1.0  (perfect)
      ΔW = -0.5 →  0.0
      ΔW = -1.0 → -1.0
    Moves without winrate data → -1.0 (imputed worst case).
    """
    valid_wrs = [v["winrate"] for v in evals.values()
                 if isinstance(v, dict) and v.get("winrate") is not None]
    if not valid_wrs:
        return -1.0

    best_wr = max(valid_wrs)

    if move == "PASS":
        entry = evals.get("pass") or evals.get("PASS") or {}
    else:
        entry = evals.get(move, {})

    chosen_wr = entry.get("winrate")
    if chosen_wr is None:
        return -1.0

    delta_w = float(chosen_wr) - float(best_wr)
    r = 1.0 + 2.0 * delta_w
    return max(-1.0, min(1.0, r))


# --- A. 格式与二维空间感知奖励 (Layer 1: gate) ---

def format_and_legality_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Gate layer with partial credit to bootstrap format learning.
    Returns:
      0.0   valid move on an empty square
     -0.5   MOVE: extracted but coord is invalid or on an occupied stone
     -1.0   no MOVE: found at all (truncated / wrong format)
    Total range [-1, 0] — keeps overall reward within [-1, +1].
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


# --- B. KataGo 复合奖励：策略直觉 + 目差 + 胜率 (Layer 2: core) ---

def katago_composite_reward_func(prompts, completions, katago_evals, **kwargs) -> List[float]:
    """
    Core reward.  Returns 0 for invalid moves; otherwise:
      0.5 × R_policy + 0.3 × R_score + 0.2 × R_winrate
    All three components ∈ [-1, +1], so total ∈ [-0.5, +1.0].

    Missing winrate/scoreLead → imputed to worst-case (-1.0) for that component.
    """
    rewards = []

    for prompt, content, evals in zip(prompts, completions, katago_evals):
        if not isinstance(evals, dict):
            evals = {}

        move = extract_move(content)

        # Gate: invalid move gets 0 here (format func already penalized it)
        if not move or check_spatial_legality(prompt, move) < 0:
            rewards.append(0.0)
            continue

        r_policy  = _compute_r_policy(move, evals)
        r_score   = _compute_r_score(move, evals)
        r_winrate = _compute_r_winrate(move, evals)

        composite = 0.5 * r_policy + 0.3 * r_score + 0.2 * r_winrate
        rewards.append(composite)

    return rewards


# --- C. 言行一致性与思考质量奖励 (Layer 3) ---

def thinking_quality_reward_func(prompts, completions, katago_evals, **kwargs) -> List[float]:
    """
    Consistency + length reward.
    Gate: 0 if move is invalid / on occupied stone.
    """
    rewards = []
    for prompt, content, evals in zip(prompts, completions, katago_evals):
        if not isinstance(evals, dict):
            evals = {}

        score = 0.0
        move_str = extract_move(content)
        think_text = extract_think_content(content)

        if not move_str or check_spatial_legality(prompt, move_str) < 0:
            rewards.append(0.0)
            continue

        # 言行一致性检查
        if move_str == "PASS":
            mentioned = [m.upper() for m in re.findall(r'\bpass\b', think_text, re.IGNORECASE)]
        else:
            mentioned = [m.upper() for m in re.findall(r'\b([A-HJ-Z][1-9])\b', think_text, re.IGNORECASE)]

        is_consistent = move_str in mentioned

        if not is_consistent:
            score -= 0.5
        else:
            score += 0.2

            length = len(think_text)
            if length == 0:
                score -= 0.5
            elif 100 <= length <= 1200:
                score += 0.1
            elif length > 1200:
                score -= 0.1

            # Bonus: move appears in KataGo's analyzed candidates
            if move_str in evals or move_str.lower() in evals:
                score += 0.2

        rewards.append(score)
    return rewards


# --- D. 日志记录 (non-scoring) ---

LOG_FILE = "src/model_training/logs/grpo_training_rollouts_v3.jsonl"

def logging_reward_func(prompts, completions, katago_evals, **kwargs) -> List[float]:
    zero_rewards = [0.0] * len(completions)

    if random.random() < 1.0:
        idx = random.randint(0, len(completions) - 1)
        prompt_text = prompts[idx]
        content = completions[idx]
        evals = katago_evals[idx] if isinstance(katago_evals[idx], dict) else {}

        move = extract_move(content)

        layer1 = 0.2 if "<think>" in content and "</think>" in content else 0.0
        if not move:
            layer1 -= 1.0
        else:
            layer1 += check_spatial_legality(prompt_text, move)

        composite = 0.0
        if move and check_spatial_legality(prompt_text, move) >= 0:
            composite = (0.5 * _compute_r_policy(move, evals)
                         + 0.3 * _compute_r_score(move, evals)
                         + 0.2 * _compute_r_winrate(move, evals))

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "board_state_prompt": prompt_text,
            "model_generation": content,
            "extracted_move": move,
            "katago_evals_sample": {k: v for k, v in list(evals.items())[:5]},
            "reward_breakdown": {
                "layer1_format_and_space": layer1,
                "layer2_katago_composite": composite,
                "r_policy":  _compute_r_policy(move, evals) if move else None,
                "r_score":   _compute_r_score(move, evals) if move else None,
                "r_winrate": _compute_r_winrate(move, evals) if move else None,
            }
        }

        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return zero_rewards


# ================= 4. 主程序 (Main) =================

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        formatted_prompts = []
        for user_p in examples["user_prompt"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_p}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}

    dataset = dataset.map(preprocess_function, batched=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
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
        warmup_ratio=0.02,           # short warmup — training runs long
        lr_scheduler_type="constant_with_warmup",  # flat LR after warmup; safe for open-ended runs
        logging_steps=10,
        bf16=True,
        per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        num_generations=8,
        gradient_accumulation_steps=4,
        max_completion_length=1024,
        save_steps=500,              # checkpoint every ~500 steps; terminate manually when ready
        max_steps=270000,            # safety ceiling (~weeks of training); stop manually via Ctrl+C
        report_to="wandb",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            format_and_legality_reward_func,   # gate: format + spatial legality
            katago_composite_reward_func,       # core: policy + score + winrate
            # thinking_quality_reward_func,    # disabled: model handles format natively
            logging_reward_func,               # non-scoring: logging
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print(f"Starting GRPO training (v3) — model: {MODEL_NAME}...")
    print(f"  Dataset: {DATASET_PATH}  ({len(dataset)} examples)")
    print(f"  vLLM: {training_args.use_vllm}")
    print(f"  Batch: {training_args.per_device_train_batch_size}  Generations: {training_args.num_generations}")

    trainer.train()

    print(f"Training done. Saving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
