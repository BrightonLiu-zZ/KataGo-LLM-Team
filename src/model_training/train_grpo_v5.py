"""
GRPO training script v5 (exp05 -> exp05b) — Qwen3-8B for 9×9 Go.

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
    - loss: epsilon_high 0.28 (DAPO clip-higher), scale_rewards "group",
      adaptive entropy regularization toward exp04's healthy early entropy (~0.5)

  exp05b (ADR 0006 Amendment 2): exp05 flatlined at base level for ~1490 steps —
  effective step size was ~5x too small. Two knobs changed, nothing else:
  scale_rewards "batch" -> "group" (restores ~3x advantage magnitude) and
  vllm_importance_sampling_mode "sequence_mask" -> "token_truncate" (removes
  the ~x0.59 sequence-product tax from trainer-vs-vLLM kernel logp mismatch).
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

# W&B destination. Pinned explicitly because the credential in ~/.netrc may
# default to a different entity than the one holding exp01-04 (that is exactly
# what happened on the first exp05 launch: the run landed in `model-as-memory`
# and was invisible next to fiery-firebrand-6). setdefault, so an env var set
# outside still wins.
os.environ.setdefault("WANDB_ENTITY", "brighton_zz-uc-san-diego")
os.environ.setdefault("WANDB_PROJECT", "huggingface")

TRAIN_DATASET_PATH = "data/training_data_v5_train.jsonl"
EVAL_DATASET_PATH = "data/eval_positions_v5.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "runs/Qwen3-8B-GRPO-Go-Pro-v5c"
LOG_FILE = "src/model_training/logs/grpo_training_rollouts_v5c.jsonl"

USE_VLLM = True  # flip to False if the vLLM smoke test fails on this machine

# Resume support. exp05c died at step 1616 to a CPython refcount crash inside a
# C extension (`none_dealloc`), which is random and can recur on a 10,000-step
# run. Set RESUME_FROM=auto to continue from the newest checkpoint in
# OUTPUT_DIR, or to an explicit checkpoint path. Pair it with
# WANDB_RUN_ID=<id> WANDB_RESUME=allow to keep one continuous W&B curve.
_resume_env = os.environ.get("RESUME_FROM", "").strip()
RESUME_FROM = (
    True if _resume_env.lower() in {"auto", "1", "true"}
    else (_resume_env or None)
)

if USE_VLLM:
    # vLLM's bundled flash-attention has PTX-only sm_120 kernels, which the
    # host's CUDA 12.8 driver cannot JIT (needs >=12.9). FlashInfer instead
    # JIT-compiles SASS with the container's CUDA 12.9 toolchain, which the
    # driver loads without version constraints. TRL 1.9.2 exposes no engine
    # kwarg for the backend, so inject it into the LLM class TRL instantiates.
    import trl.generation.vllm_generation as _vllm_gen
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _OrigLLM = _vllm_gen.LLM

    def _llm_with_flashinfer(*args, **kwargs):
        kwargs.setdefault("attention_backend", AttentionBackendEnum.FLASHINFER)
        llm = _OrigLLM(*args, **kwargs)
        # Part of the TRL #5312 fix below: generate() used to call
        # collective_rpc("reload_weights"), which with no arguments reloads
        # the BASE model from disk — clobbering the trained weights that
        # sync_weights() had just pushed. Neuter that RPC; the generate
        # wrapper below re-pushes trainer weights instead.
        _orig_rpc = llm.collective_rpc

        def _rpc_no_reload(method, *a, **k):
            if method == "reload_weights":
                return None
            return _orig_rpc(method, *a, **k)

        llm.collective_rpc = _rpc_no_reload
        return llm

    _vllm_gen.LLM = _llm_with_flashinfer

    # --- Fix for TRL #5312 (upstream PR #5313, merged 2026-07-28 but not in
    # any release as of trl 1.9.2). With vllm_enable_sleep_mode=True the
    # per-step order was: sync_weights() pushes trained weights -> generate()
    # calls reload_weights, wiping them back to base -> every rollout AND
    # every eval ran on the frozen base policy (the exp05/exp05b flatline;
    # see ADR 0006 Amendments 2-3). Mirror the upstream fix: never reload
    # from disk; re-push trainer weights whenever the engine's level-2 sleep
    # has discarded them since the last sync.
    _VG = _vllm_gen.VLLMGeneration
    _orig_sync = _VG.sync_weights
    _orig_generate = _VG.generate

    def _sync_and_mark(self, *a, **k):
        out = _orig_sync(self, *a, **k)
        self._weights_need_push = False
        return out

    def _generate_fixed(self, *args, **kwargs):
        if self.mode == "colocate" and self.enable_sleep_mode:
            # True initially (engine sleeps right after init) and after every
            # generate(); False right after a sync_weights().
            if getattr(self, "_weights_need_push", True):
                self.sync_weights()
        out = _orig_generate(self, *args, **kwargs)
        if self.mode == "colocate" and self.enable_sleep_mode:
            self._weights_need_push = True  # generate() ends with sleep(level=2)
        return out

    _VG.sync_weights = _sync_and_mark
    _VG.generate = _generate_fixed

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
    "vllm_importance_sampling_mode",
}

CONFIG = dict(
    output_dir=OUTPUT_DIR,
    run_name="exp05c-grpo-v5c",
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
    scale_rewards="group",          # exp05b: "batch" divided advantages by the
                                    # batch-wide std (~0.4) instead of per-group
                                    # std, shrinking gradients ~3x -> flatline.
                                    # See ADR 0006 Amendment 2.
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
    save_steps=200,
    save_total_limit=6,             # /home has ~58G free; ~525MB per checkpoint
    report_to="wandb",
    # -- throughput
    use_vllm=USE_VLLM,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.2,
    # Without a cap vLLM sizes its KV cache for Qwen3's native 40,960-token
    # window and refuses to start inside the 0.2 memory budget. Measured need:
    # prompts <=518 tokens + 512 completion = 1,030.
    vllm_max_model_length=1280,
    vllm_enable_sleep_mode=False,
    # exp05b: TRL's default "sequence_mask" multiplies each sequence's loss by
    # exp(sum of per-token trainer-vs-vLLM logp diffs); ~-0.006/token kernel
    # mismatch x ~90 tokens = a flat x0.59 tax on every gradient (the constant
    # 0.59 IS-ratio on W&B in exp05). Token-level truncation keeps the
    # correction (~0.994/token) without the sequence-product amplification.
    vllm_importance_sampling_mode="token_truncate",
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

    if RESUME_FROM:
        print(f"  Resuming from: {RESUME_FROM}")
    trainer.train(resume_from_checkpoint=RESUME_FROM)

    print(f"Training done. Saving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
