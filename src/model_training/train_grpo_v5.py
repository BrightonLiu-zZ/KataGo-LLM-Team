"""
GRPO training script v5 (exp05) — Qwen3-8B for 9×9 Go.

Everything that changed vs v4, and why, is recorded in
docs/adr/0006-entropy-preserving-grpo-config-v5.md. Summary:

  Diagnosed from exp04 (W&B fiery-firebrand-6; see notebooks/exp01-04_analysis.ipynb):
    - policy entropy collapse (0.55 -> 0.31) == completion-length collapse (90 -> 43 tok)
    - up to 61% of groups with zero reward std -> zero gradient
    - whole run inside LR warmup (warmup_ratio 0.02 x max_steps 270,000)
    - reasoning text never rewarded -> templated "decorative CoT"

  Fixes in this script:
    - reward: + reasoning layer (consistency / length window / group dedup), see reward_v5.py
    - sampling: temperature 1.15
    - loss: epsilon_high 0.28 (DAPO clip-higher), scale_rewards "batch",
      adaptive entropy regularization toward exp04's healthy early entropy (~0.5)
    - schedule: lr 2e-6, warmup_steps=100 (absolute), max_steps=10,000 (real horizon)
    - batch: num_generations 16, grad_accum 8 (128 completions = 8 groups/step)
    - throughput: vLLM colocate + sleep mode (fallback: set USE_VLLM = False)
    - eval: leakage-safe held-out set every 500 steps (prepare_v5_split.py)

Requires TRL >= 1.8 (adaptive entropy). The script fails fast with a clear
message if the installed GRPOConfig does not support an agreed-upon knob.
"""
import os
import json
import random
from dataclasses import fields as dc_fields
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
import trl
from trl import GRPOTrainer, GRPOConfig

from reward_v5 import (
    extract_move,
    check_spatial_legality,
    extract_reasoning,
    composite_core,
    compute_r_policy_log,
    compute_r_rank,
    reasoning_quality_rewards,
)

# ================= 1. Configuration =================

TRAIN_DATASET_PATH = "data/training_data_v5_train.jsonl"
EVAL_DATASET_PATH = "data/eval_positions_v5.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "runs/Qwen3-8B-GRPO-Go-Pro-v5"
LOG_FILE = "src/model_training/logs/grpo_training_rollouts_v5.jsonl"

USE_VLLM = True  # flip to False if the vLLM smoke test fails on this machine

SYSTEM_PROMPT = (
    "You are a 9x9 Go (Weiqi) player. "
    "Board notation: '.' = empty, 'X' = Black, 'O' = White. "
    "Columns: A-J (no I). Rows: 1 (bottom) to 9 (top). "
    "You MUST only play on an empty '.' intersection listed in the valid coordinates.\n\n"
    "Respond in exactly this format:\n"
    "REASONING: [2-4 sentences analyzing the position]\n"
    "MOVE: [coordinate, e.g. D4]"
)


# ================= 2. Reward functions (TRL signature) =================

def format_and_legality_reward_func(prompts, completions, **kwargs):
    """Gate layer, unchanged from v4: -1 / -0.5 / 0."""
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


def katago_composite_reward_func(prompts, completions, katago_evals, root_winrate, **kwargs):
    """Core layer, unchanged from v4: adaptive log-policy + scoreLead rank."""
    rewards = []
    for prompt, content, evals, rwr in zip(prompts, completions, katago_evals, root_winrate):
        if not isinstance(evals, dict):
            evals = {}
        move = extract_move(content)
        if not move or check_spatial_legality(prompt, move) < 0:
            rewards.append(0.0)
            continue
        rewards.append(composite_core(move, evals, rwr))
    return rewards


def reasoning_quality_reward_func(prompts, completions, **kwargs):
    """v5 reasoning layer: consistency + length window + group dedup.
    Range [-0.2, +0.05]; 0.0 whenever the gate already failed."""
    return reasoning_quality_rewards(prompts, completions)


def logging_reward_func(prompts, completions, katago_evals, root_winrate, **kwargs):
    """Non-scoring: samples one completion per batch into the rollout log."""
    zero_rewards = [0.0] * len(completions)

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

    core = composite_core(move, evals, rwr) if gate == 0.0 else 0.0
    reasoning = extract_reasoning(content)
    aux = reasoning_quality_rewards([prompt_text], [content])[0]

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "board_state_prompt": prompt_text,
        "model_generation": content,
        "extracted_move": move,
        "reasoning_chars": len(reasoning),
        "root_winrate": float(rwr) if rwr is not None else None,
        "reward_breakdown": {
            "gate": gate,
            "core": round(core, 6),
            "r_policy_log": round(compute_r_policy_log(move, evals), 6) if move else 0.0,
            "r_rank": round(compute_r_rank(move, evals), 6) if move else 0.0,
            "reasoning_aux": round(aux, 6),
            "total": round(gate + core + aux, 6),
        },
    }
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return zero_rewards


# ================= 3. Config with fail-fast compatibility check =================

# Every knob here was an explicit exp05 decision (ADR 0006). If the installed
# TRL doesn't know one of them, upgrading TRL is the fix — not silently
# dropping the knob.
REQUIRED_KNOBS = {
    "epsilon_high", "scale_rewards", "temperature",
    "use_adaptive_entropy", "entropy_target",
}

CONFIG = dict(
    output_dir=OUTPUT_DIR,
    run_name="exp05-grpo-v5",
    # -- schedule: real horizon, absolute warmup (exp04 never left warmup)
    learning_rate=2e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=100,
    max_steps=10000,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    # -- batch: 16 gens x 8 groups = 128 completions per optimizer step
    per_device_train_batch_size=16,
    num_generations=16,
    gradient_accumulation_steps=8,
    max_completion_length=512,
    # -- entropy preservation package
    temperature=1.15,               # ProRL: rollout diversity
    epsilon=0.2,
    epsilon_high=0.28,              # DAPO clip-higher
    scale_rewards="batch",          # Lite PPO; removes per-group difficulty bias
    loss_type="dapo",               # explicit (was the silent default in v4)
    beta=0.0,                       # explicit; KL + ref resets reserved for exp06
    use_adaptive_entropy=True,      # Skywork-OR1 adaptive entropy control
    entropy_target=0.5,             # exp04's healthy early entropy was ~0.55
    entropy_coef=0.01,
    entropy_coef_delta=0.005,
    # -- eval on the leakage-safe held-out split
    eval_strategy="steps",
    eval_steps=500,
    per_device_eval_batch_size=16,
    # -- logging / checkpoints
    logging_steps=10,
    bf16=True,
    save_steps=500,
    save_total_limit=6,             # /home has ~58G free; ~525MB per checkpoint
    report_to="wandb",
    # -- throughput
    use_vllm=USE_VLLM,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.2,
    vllm_enable_sleep_mode=True,
)


def build_config():
    valid = {f.name for f in dc_fields(GRPOConfig)}
    unknown = set(CONFIG) - valid
    missing_required = unknown & REQUIRED_KNOBS
    if missing_required:
        raise RuntimeError(
            f"Installed TRL {trl.__version__} does not support required exp05 "
            f"knobs: {sorted(missing_required)}. Upgrade TRL (>=1.8; the "
            f"Dockerfile pins a known-good version) instead of dropping them."
        )
    if unknown:
        print(f"[warn] dropping GRPOConfig kwargs unknown to TRL {trl.__version__}: "
              f"{sorted(unknown)}")
    return GRPOConfig(**{k: v for k, v in CONFIG.items() if k in valid})


# ================= 4. Main =================

def main():
    for path, hint in [
        (TRAIN_DATASET_PATH, "Run prepare_v5_split.py first."),
        (EVAL_DATASET_PATH, "Run prepare_v5_split.py first."),
    ]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. {hint}")
            return

    print(f"TRL version: {trl.__version__}")
    train_dataset = load_dataset("json", data_files=TRAIN_DATASET_PATH, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_DATASET_PATH, split="train")

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
                enable_thinking=False,   # ADR 0003
            )
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

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

    training_args = build_config()

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            format_and_legality_reward_func,
            katago_composite_reward_func,
            reasoning_quality_reward_func,
            logging_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    print(f"Starting GRPO training (v5 / exp05) — model: {MODEL_NAME}")
    print(f"  Train: {TRAIN_DATASET_PATH} ({len(train_dataset)} examples)")
    print(f"  Eval:  {EVAL_DATASET_PATH} ({len(eval_dataset)} examples)")
    print(f"  vLLM: {training_args.use_vllm}  "
          f"Batch: {training_args.per_device_train_batch_size}  "
          f"Generations: {training_args.num_generations}  "
          f"Accum: {training_args.gradient_accumulation_steps}")

    trainer.train()

    print(f"Training done. Saving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
