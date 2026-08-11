# CLAUDE.md — KataGo LLM workspace

Single source of truth for this repo. Prefer this over stale comments in random files.

**Companion docs:**

- [`CONTEXT.md`](./CONTEXT.md) — glossary. What the project's terms mean, and which
  synonyms to avoid. Read it before naming anything new.
- [`docs/adr/`](./docs/adr/) — why the hard-to-reverse decisions were made the way
  they were.
- [`docs/RECOVERY-2026-05-06.md`](./docs/RECOVERY-2026-05-06.md) — the 2026-05-06
  data-loss incident: what was lost permanently, what was rebuilt and how.

---

## What this repo is

- **Goal:** Train LLMs to play **9×9 Go** with **GRPO** (Group Relative Policy Optimization), using **KataGo** as the teacher.
- **Current training stack:** **Qwen/Qwen3-8B** — active experiment is **exp05c** (**`train_grpo_v5.py`** launched via `run_with_restart.sh`, datasets from `prepare_v5_split.py`, rationale in ADR 0006 + Amendments 2–4; exp05 and exp05b were both stopped after diagnosed bugs, see Pitfalls). Last completed run: exp04 (`train_grpo_v4.py`, `data/training_data_v4.jsonl`).
- **Experiment ↔ W&B mapping and the exp04 post-mortem:** `notebooks/exp01-04_analysis.ipynb` (executed; data cached in `notebooks/data/`).
- **Edge deployment (proven):** v4 checkpoint → GGUF Q4_K_M → **LM Studio** → **C++ GTP proxy** → **Lizzie** (Go GUI). Also supports loading the **un-fine-tuned base model** for comparison.
- **Legacy:** `train_grpo_v3.py` (v3 dataset), `train_grpo.py` (Qwen2.5-7B v1).

---

## Git conventions

- **Committing and pushing is governed by the `commit-and-push` skill**
  (`.claude/skills/commit-and-push/SKILL.md` — deliberately **not** in version
  control, since `.claude/` is gitignored; recreate it on a new machine) — it
  is the authority on batching
  (one commit per file), message style (**no `feat:`/`chore:` prefixes**), what
  may be touched (only files already changed in `git status`), and the push
  flow (commit everything, push once at the end). Follow it whenever asked to
  commit, push, or land the current changes.
- **Never add `Co-Authored-By: Claude` (or any AI attribution) to commit messages.** Commits are authored solely by the human developer.
- Run commands from **repository root** unless a script says otherwise.
- **Board:** columns **A–J** (skip **I**), rows **1–9** (1 = bottom). Moves: `C4`, `PASS`.
- Keep **`VALID_COLS = set("ABCDEFGHJ")`** consistent everywhere it appears.
- **Qwen3 chat:** always use **`enable_thinking=False`** in `apply_chat_template()` for training and interactive tests — native thinking mode blows the token budget before `MOVE:`.

---

## Environment

**Docker (recommended):**
```bash
# Production (RTX Ada, vLLM, Flash Attention 2)
docker build -f Dockerfile -t katago-llm .
docker run --gpus all -it -v $(pwd):/workspace -w /workspace katago-llm bash

# Local dev (4-bit quant, no vLLM)
docker build -f Dockerfile.local -t katago-llm-local .
docker run --gpus all -it -v $(pwd):/workspace -w /workspace katago-llm-local bash
```

**Conda (manual):**
```bash
conda create -n katago-llm python=3.10 -y && conda activate katago-llm
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl bitsandbytes pandas scipy wandb sgfmill
```

**KataGo:** binary `KataGo_engine/katago` (`chmod +x`), weights `KataGo_engine/KataGo18b9x9.gz`. Paths in `src/data_acquiring/get_topk_from_katago/run_katago_analysis.py`.

The binary is CUDA-linked and needs `libcublas.so.12` + `libcudnn.so.9`, which the
bare host does not have — running it directly fails with a shared-library error.
It has always been run inside the CUDA container. Rebuild that image from
`Dockerfile` if it is missing. The weights currently in place are
`kata1-b18c384nbt-s9996604416-d4316597426` re-downloaded from katagotraining.org;
the original net file was lost and its exact identity is unknown, so KataGo evals
generated from here on will not be bit-identical to the v3 corpus.

---

## Data pipeline (v3 JSONL → v4 training set)

Typical v3 flow (each step feeds the next; edit paths in scripts as needed):

1. `filter_9x9.py` → filtered SGFs
2. `sgf_to_jsonl.py` → `data/json_output.jsonl`
3. `run_katago_analysis.py` → enriched JSONL with KataGo evals
4. `prepare_training_data.py` (v3) → e.g. `data/training_ready_data_v3.jsonl`
5. `data_quality_cut.py` → `data/training_data_quality_filtered_v3.jsonl`
6. `data_augmentation.py` → `data/training_data_augmented_v3.jsonl` (D4 × color)
7. Shuffle → `data/training_data_augmented_shuffled_v3.jsonl`

**v4 dataset (do not re-augment):** from the **already augmented** shuffled v3 file, run:

```bash
python prepare_v4_dataset.py
```

This builds **`data/training_data_v4.jsonl`**: restores **valid empty coordinates** in each prompt (stripped by augmentation), **downsamples lopsided** positions (root winrate above 0.95 or below 0.05) to **30%**, keeps balanced positions fully, then shuffles (113,404 rows).

**Where the chain is broken.** Steps 1–4 cannot be re-run against the original
corpus: `data/json_output_with_topk_official.jsonl` (18,493 positions, ~48
GPU-hours of KataGo analysis) and `data/training_ready_data_v3.jsonl` were
destroyed on 2026-05-06 and exist nowhere. The pipeline is intact from step 6
onward because `training_data_augmented_v3.jsonl` had been archived to HF.

Step 5's output was rebuilt by inverting step 6 — the augmenter tags identity
records with `_aug0`, so they can be selected and un-suffixed:

```bash
python src/data_acquiring/get_train_ready_data/reconstruct_quality_filtered_from_augmented.py
```

Yields 16,875 records, matching the original count exactly. To rebuild the corpus
from scratch you need the source SGFs (last known location: a Windows machine,
`D:\katago_old\9x9_online_go_game`) and a full re-run of steps 1–4.

---

## Training

**Active — v5c / exp05c (running, W&B `lc4hyp23`):** the first run of this
generation that actually learns — core reward 0.401 → 0.492 over 1,500 steps
(~1.5× exp04's rate) with entropy held at 0.52 and length at 88 tokens, and
held-out eval core 0.4853 → 0.5059. Two dead runs preceded it, each
root-caused in ADR 0006:
- exp05 (W&B `tijdgbfj`, ~1490 steps): ~5× effective-step-size deficit
  (`scale_rewards="batch"` ~3× + TRL `sequence_mask` vLLM-IS tax ×0.59) —
  fixed by exp05b's two knobs (Amendment 2).
- exp05b (~2000 steps): TRL 1.9.2 upstream bug ([trl#5312]) — with
  `vllm_enable_sleep_mode=True`, `generate()` calls
  `collective_rpc("reload_weights")` *after* `sync_weights()`, reloading the
  **base model from disk** every step; all rollouts and evals in exp05/05b
  came from the frozen base policy. `train_grpo_v5.py` now monkey-patches
  the fix (mirrors unreleased TRL PR #5313; remove when a TRL release ships
  it), validated by a checkpoint-initialized smoke (Amendment 3).

Watch on W&B: `importance_sampling_ratio/mean` ~0.99 and *stable* (exp05b
decayed 1.0→0.66 = sync broken again) — currently 0.992; core-reward slope —
holding; `frac_reward_zero_std` — **at ~23%, above the <15% target**, flat
since step 750; sustained growth past ~30% means exp04's collapse is back.
```bash
python prepare_v5_split.py                       # once: leakage-safe train/eval split

# Long runs: always launch through the wrapper. exp05c hit a random C-extension
# crash (`none_dealloc`) at step 1616; the wrapper resumes from the newest
# checkpoint and refuses to restart-loop on a genuinely broken config.
WANDB_RUN_ID=lc4hyp23 WANDB_RESUME=allow \
  bash src/model_training/run_with_restart.sh

# Bare invocation (smoke tests, fresh experiments):
python src/model_training/train_grpo_v5.py       # RESUME_FROM=auto to continue
```
- **Datasets:** `data/training_data_v5_train.jsonl` (105,596 rows) + `data/eval_positions_v5.jsonl` (502 held-out positions, split by source game — row-wise splitting would leak augmentation variants)
- **Output:** `runs/Qwen3-8B-GRPO-Go-Pro-v5c` (the dead runs left `…-v5/checkpoint-{500,1000}` and `…-v5b/checkpoint-{500,1000,1500}`; kept as evidence)
- **Reward:** v4 gate + core unchanged, plus a **reasoning layer** (`reward_v5.py`): coordinate mention +0.05, length window down to −0.1, group dedup down to −0.1. CPU tests: `python src/model_training/test_reward_v5.py`
- **Config vs v4:** lr **2e-6** constant, **warmup_steps=100** (absolute), **max_steps=10000**, **num_generations=16**, grad_accum=8, **temperature=1.15**, **epsilon_high=0.28**, **scale_rewards="group"** (exp05b; exp05's `"batch"` shrank gradients ~3×), **vllm_importance_sampling_mode="token_truncate"** (exp05b; the `sequence_mask` default taxed every gradient ×0.59), adaptive entropy (target 0.5), eval every 500 steps, **`save_steps=200`** (must stay below the crash interval — see Pitfalls), `save_total_limit=6`, vLLM colocate with **`vllm_enable_sleep_mode=False`** (Amendment 5; `USE_VLLM=False` in the script to fall back entirely)
- **GPU footprint:** ~**88.7 GB of 97.9 GB (90.6%)**, steady state, since sleep mode was disabled. Shared server — coordinate before launching. If a co-tenant needs more than ~9 GB, try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first (torch reserves 85.4 GiB while allocating only 62.7 GiB), then lower `vllm_gpu_memory_utilization` (tight below 0.2: the 8B weights alone need ~16 GB)
- **Requires:** TRL ≥1.8 — Dockerfile pins `trl==1.9.2` + vLLM `0.25.1+cu129` (GitHub wheel) on a **CUDA 12.9** base. The 12.9/cu129 combination is forced by the host's 12.8 driver (570.x, no sudo): cu130 wheels won't initialize, vLLM's bundled flash-attention is PTX-only for sm_120 (driver can't JIT it) so the script injects `attention_backend=FLASHINFER`, and flashinfer's sm_120 JIT needs nvcc ≥12.9 in the container. The script fails fast if a required knob is missing. Smoke-tested 2026-08-04: ~40s/step, ~68 min per full eval, est. 5.5 days for 10k steps
- **Stopping rule:** eval reward flat for 4 consecutive evals (2,000 steps) → stop manually. Note this rule assumes "learned then plateaued" — a curve that is flat *from step 0* is a bug, not a plateau; triage it with the recipe under Pitfalls instead of waiting out 2,000 steps
- **Resume:** `RESUME_FROM=auto` (newest checkpoint in `OUTPUT_DIR`) or a checkpoint path; a crash costs up to `save_steps` = 500 steps ≈ 5 h
- **Why all of it:** `docs/adr/0006-entropy-preserving-grpo-config-v5.md`

**Previous — v4 (exp04, completed):**
```bash
python src/model_training/train_grpo_v4.py
```
- **Dataset:** `data/training_data_v4.jsonl`
- **Output:** `runs/Qwen3-8B-GRPO-Go-Pro-v4`
- **Format:** `REASONING:` + `MOVE:` (no `</think>` scaffold)
- **`max_completion_length`:** 512
- **Rewards:** gate (format + empty intersection) + **log-scaled policy** + **scoreLead rank** among legal moves, with **adaptive weights** from root winrate "closeness"
- **Logs:** `src/model_training/logs/grpo_training_rollouts_v4.jsonl`
- **LoRA:** `r=16`, `lora_alpha=32`, dropout `0.05`, all projection layers
- **Batch:** `per_device_train_batch_size=16`, `num_generations=8`, `gradient_accumulation_steps=4`
- **Checkpoints:** every 500 steps

**Legacy — v3:**
```bash
python src/model_training/train_grpo_v3.py
```
- Dataset: `data/training_data_augmented_shuffled_v3.jsonl`
- Output: `runs/Qwen3-8B-GRPO-Go-Pro-v3`
- Composite: `0.5×R_policy + 0.3×R_score + 0.2×R_winrate`; `max_completion_length` 1024

**Other:** `train_grpo.py` (7B v1), `train_grpo_laptop.py` (0.5B ≤12GB), `train_grpo_local.py` (smoke-test, 5 steps).

**VRAM:** Qwen3-8B GRPO needs **~68GB+** (RTX PRO 6000 98GB). v1 Qwen2.5-7B needs ≥48GB. Adjust batch/gen if OOM.

---

## Testing & interactive play

```bash
python src/model_training/test_reward_v5.py          # no GPU — v5 reward unit tests
python src/model_training/test_model_interactive.py  # must use enable_thinking=False + v4 system prompt if testing v4
```

---

## Reward stacks (short)

**v5 (`train_grpo_v5.py` + `reward_v5.py`):** v4 gate + core unchanged, plus
reasoning layer in `[−0.2, +0.05]`: coordinate mention (+0.05), length window
(<200 chars → down to −0.1), within-group bigram-overlap penalty (down to
−0.1; breaks same-move reward ties → fewer zero-variance groups). Logging v5.

**v4 (`train_grpo_v4.py`):**
1. **Gate:** no `MOVE:` → −1; illegal coord / occupied → −0.5; legal on `.` → 0.
2. **Core:** `w_policy × R_policy_log + w_rank × R_rank` (rank = percentile by **scoreLead** over KataGo-evaluated legal moves). Weights depend on **root_winrate** closeness to 0.5.
3. **Logging:** non-scoring; JSONL v4.

**v3 (`train_grpo_v3.py`):** gate + `0.5×policy + 0.3×score + 0.2×winrate` + logging v3.

**v1 (`train_grpo.py`):** format/tags + outcome regret (`chosen_wr − 1.5 × (best_wr − chosen_wr)`) + thinking-quality (50–800 chars) + logging.

---

## Edge deployment & Lizzie integration

### v4 (Qwen3-8B) — proven

Full pipeline: **LoRA merge → GGUF Q4_K_M → LM Studio → C++ GTP proxy → Lizzie**.

**Quantize (Linux/Docker):**
```bash
bash src/deployment/export_gguf_model_v4.sh
```
Merges LoRA from `runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-3000` into Qwen3-8B, converts to FP16 GGUF, quantizes to `qwen3-8b-go-v4-Q4_K_M.gguf` (~5GB).

**Build proxy (Windows, Developer Command Prompt):**
```bat
cd src\interception
cl.exe /std:c++17 /EHsc /utf-8 /W4 /O2 gtp_proxy_v4.cpp /Fe:gtp_proxy_v4.exe /link ws2_32.lib
```

**Run:** Load GGUF in LM Studio → Start Server (port 1234) → point Lizzie engine to `gtp_proxy_v4.exe`.

**Features:**
- Full 9×9 board state with Go capture logic
- ASCII board + move history + valid empty coords sent to LLM (matches training prompt format)
- Model reasoning (`REASONING: ... MOVE: ...`) printed to **stderr** → visible in Lizzie's GTP Console
- `<think>` tags stripped automatically; only user-facing reasoning shown
- Invalid move → auto-fallback to `pass`

### Base model comparison (Qwen3-8B un-fine-tuned)

`gtp_proxy_v4_base.cpp` / `gtp_proxy_v4_base.exe` — identical to v4 proxy but points at the **base Qwen3-8B** GGUF (download `Qwen3-8B-Q4_K_M.gguf` from [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) on HuggingFace).

```bat
cl.exe /std:c++17 /EHsc /utf-8 /W4 /O2 gtp_proxy_v4_base.cpp /Fe:gtp_proxy_v4_base.exe /link ws2_32.lib
```

LM Studio can load one model at a time on 8GB VRAM. Switch between fine-tuned and base by swapping models + Lizzie engine.

### Legacy v1 (Qwen2.5-7B) — proven

- Script: `src/deployment/legacy_7b/export_gguf_model.sh`
- Proxy: `gtp_proxy.cpp` / `gtp_proxy.exe`
- Model: `qwen-7b-go-Q4_K_M.gguf` (on HF: `brightonliuzZ/qwen2.5-7b-go`)

### LM Studio notes (v0.4.7+)

- **Model identifiers** use `publisher/model@quant` format (e.g. `qwen/qwen3-8b@q4_k_m`), **not** raw filenames. Custom/manually-loaded GGUFs may still use filenames. Check the error message if loading fails.
- **Thinking mode:** Qwen3 models default to thinking mode. Disable via: `max_tokens: 4096` (large enough for thinking + response), `/no_think` appended to system prompt, and `"chat_template_kwargs": {"enable_thinking": false}` in the API payload. All three are set in the v4 proxy source.

### Alternative inference backends

- **llama.cpp server:** `./llama.cpp/build/bin/llama-server -m <model>.gguf --port 1234` — same `/v1/chat/completions` API, no proxy changes needed.
- **Ollama:** install from [ollama.com](https://ollama.com/), create a `Modelfile`, change `LM_STUDIO_PORT` to `11434`.

See **`src/interception/GTP_PROXY_GUIDE.md`** for full setup guide.

---

## Paths to edit before runs

| Script / area | What to check |
|----------------|---------------|
| `run_katago_analysis.py` | `KATAGO_EXE`, `CONFIG_FILE`, `MODEL_FILE`, `MAX_LIMIT` (default 100) |
| `train_grpo_v5.py` | `TRAIN_DATASET_PATH`, `EVAL_DATASET_PATH`, `OUTPUT_DIR`, `USE_VLLM`; env `RESUME_FROM` |
| `run_with_restart.sh` | env `MAX_RESTARTS`, `MIN_HEALTHY_SECONDS`, `WANDB_RUN_ID` (reads `OUTPUT_DIR` from the trainer) |
| `prepare_v5_split.py` | `INPUT_FILE`, `EVAL_TARGET`, `SEED` |
| `train_grpo_v4.py` | `DATASET_PATH`, `OUTPUT_DIR` |
| `train_grpo_v3.py` | `DATASET_PATH`, `OUTPUT_DIR` |
| `train_grpo.py` | `DATASET_PATH`, `OUTPUT_DIR` |
| `filter_9x9.py` | `INPUT_DIR`, `OUTPUT_DIR` |
| `prepare_v4_dataset.py` | `INPUT_FILE`, `OUTPUT_FILE`, lopsided thresholds |
| `export_gguf_model_v4.sh` | `LORA_CKPT_PATH` (now `checkpoint-5000`; the published GGUF came from the lost `checkpoint-3000`), `BASE_MODEL` |
| `export_gguf_model.sh` | `LORA_CKPT_PATH` (defaults to `runs/Qwen2.5-7B-GRPO-Go-Pro-v2/checkpoint-1000`) |
| `gtp_proxy_v4.cpp` | `LM_STUDIO_MODEL`, `LM_STUDIO_HOST`, `LM_STUDIO_PORT` |
| `gtp_proxy_v4_base.cpp` | `LM_STUDIO_MODEL` (must match LM Studio's identifier for the base model) |

---

## Pitfalls (lessons learned)

- **Native Qwen3 thinking in GRPO:** disable with `enable_thinking=False` or generations truncate before `MOVE:`. In training, Qwen3's native mode can produce 500+ tokens of `<think>` before any useful output.
- **LM Studio thinking mode (inference):** even with `enable_thinking=False`, LM Studio may route Qwen3 output to `reasoning_content` instead of `content`, causing empty responses. Fix: set `max_tokens: 4096` and append `/no_think` to system prompt. See "LM Studio notes" section above.
- **LM Studio model identifiers (v0.4.7+):** models downloaded via LM Studio search use `publisher/model@quant` format, not filenames. Manually loaded GGUFs may differ.
- **Augmentation dropped valid-coordinate lists:** v4 dataset script restores them; training prompts must include them for legality.
- **Lopsided positions:** most training rows had root winrate near 0 or 1 — v4 downsamples those and uses rank-based core reward so signals aren't dominated by saturated winrate deltas.
- **GRPO:** `num_generations` must divide evenly into batch settings (common crash if misaligned).
- **Cold start:** binary gate only gives weak gradients; partial credit (−1 / −0.5 / 0) helps early training.
- **Windows compilation:** `cl.exe` requires **Developer Command Prompt for Visual Studio**, not regular PowerShell. Alternative: MinGW `g++ -std=c++17 -O2 -o proxy.exe proxy.cpp -lws2_32`.

### GRPO / TRL training — the flatline family

exp05, exp05b and exp05c between them burned ~5 days of GPU on runs whose
reward curves were flat. All three causes were invisible in the reward panels
alone. **Before committing to any multi-day run, do the pre-flight below.**

- **Pre-flight (30 minutes, catches all three):** run ~10 steps twice — once
  from the base model, once initialized from a *trained* checkpoint
  (`PeftModel.from_pretrained(base, ckpt, is_trainable=True)` passed as
  `model=`, with `peft_config=None`). If the two runs produce the same reward
  level, generation is not seeing the trained weights. Then check
  `sampling/importance_sampling_ratio/mean` ≈ 0.99 on step 1.
- **A flat reward curve does not mean "the config is wrong."** Triage in this
  order, all cheap and offline (`tmp/discriminate_v5.py` does 1–3):
  1. `‖lora_B‖_F` across consecutive checkpoints (`lora_B` starts at exactly
     0) — flat/tiny ⇒ the *trainer* is not learning; growing ⇒ it is, and the
     problem is downstream in generation.
  2. Rescore real rollout completions from the log under base vs
     base+checkpoint. Per-token IS ratio ≈ 1.0 ⇒ the policy never moved.
  3. `importance_sampling_ratio/mean` **trend**: a constant <1 value is a
     gradient tax; a monotonic decay (1.0 → 0.66) means trainer and sampler
     are diverging, i.e. **weight sync is broken**.
  4. Rollout log: ring distribution of moves, early window vs late. Frozen
     distribution + frozen text = generation from a frozen policy.
- **Silent gradient shrink (exp05, ~5× too slow):** `scale_rewards="batch"`
  divides advantages by the batch-wide std instead of the per-group std (~3×
  smaller), and TRL's default `vllm_importance_sampling_mode="sequence_mask"`
  multiplies each sequence's loss by exp(Σ per-token logp diff) — a ~×0.59 tax
  from harmless per-token kernel mismatch. **Use `scale_rewards="group"` and
  `vllm_importance_sampling_mode="token_truncate"`** unless you can defend
  otherwise; both are set in `train_grpo_v5.py`.
- **vLLM sleep-mode weight sync (exp05b, [trl#5312]):** with
  `vllm_enable_sleep_mode=True`, TRL ≤1.9.2 calls
  `collective_rpc("reload_weights")` inside `generate()` *after*
  `sync_weights()`; with no arguments that **reloads the base model from
  disk**, so every rollout and every eval comes from the frozen base policy.
  `train_grpo_v5.py` monkey-patches the fix (mirrors TRL PR #5313, merged
  2026-07-28, unreleased as of 1.9.2). **Delete the patch once a TRL release
  contains #5313** — check before upgrading TRL.
- **`frac_reward_zero_std` lies under `scale_rewards="batch"`:** TRL derives
  it from whichever std the scaling mode produced, so under `"batch"` it reads
  ~0 regardless. Only comparable across runs that use `"group"`.
- **Reward-scale changes make curves incomparable across experiments** (ADR
  0002); compare rank/eval metrics, not raw reward, when the stack changed.

### Container & CUDA environment

- **Never pipe `docker build` into `tail`/`head`.** The pipe swallows the exit
  code: a build that died on "no space left on device" reported success and
  the failure only surfaced hours later. Run it unpiped, or check
  `${PIPESTATUS[0]}`.
- **`docker exec` with a heredoc needs `-i`**, otherwise stdin is not
  forwarded and the script silently does nothing.
- **Blackwell (sm_120) on this host's 12.8 driver** needs the whole chain in
  the Dockerfile comments: cu129 wheels (not cu130), CUDA 12.9 base image (for
  flashinfer's JIT), and `attention_backend=FLASHINFER` injected into TRL's
  vLLM constructor. Misleading symptom to recognise: flashinfer reports
  "requires GPUs with sm75 or higher" when the real error is "SM 12.x requires
  CUDA >= 12.9".
- **vLLM sizes its KV cache for the model's native context** (40,960 tokens
  for Qwen3) and refuses to start inside a small memory budget — cap it with
  `vllm_max_model_length`.
- **transformers 5.x `apply_chat_template(tokenize=True)` returns a dict,**
  not a list of ids — `len()` on it gives 2 and silently corrupts any
  prompt-length measurement. Use `tokenize=False` then tokenize explicitly.

### Running long jobs

- **Random C-extension crashes are expected on multi-day runs.** exp05c died
  at step 1616 with `Fatal Python error: none_dealloc` (refcount bug in a C
  extension — vLLM/FlashInfer/Triton/numba are all loaded). Nothing to fix in
  the config. **Launch through `src/model_training/run_with_restart.sh`**,
  which resumes from the newest checkpoint, keeps the W&B curve continuous
  (`WANDB_RUN_ID=<id> WANDB_RESUME=allow`), and refuses to restart-loop on a
  genuinely broken config (3 fast failures ⇒ abort).
- **`save_steps` must stay below the observed crash interval.** exp05c's
  second crash came 3.6 h in while checkpoints were 5.3 h apart (500 steps ×
  38 s), so the restart rolled back from step 1834 to 1500 and was on course
  to repeat that forever — **progress can livelock with no error message**,
  and the wrapper's fast-failure guard cannot see it because each attempt runs
  for hours. Now `save_steps=200` (~2.1 h). `save_total_limit` caps the disk
  cost regardless of the interval, so shortening it is nearly free.
- **vLLM sleep mode is a liability on long runs** (disabled since ADR 0006
  Amendment 5). Sleep level 2 frees vLLM's weights, and every `wake_up()` must
  re-map ~16 GB of *physical* memory while torch holds most of the card
  (measured: 62.7 GiB allocated vs **85.4 GiB reserved** — a 22.7 GiB
  fragmentation gap). exp05c crash #2 was exactly that:
  `CUDA Error: out of memory at cumem_allocator.cpp:163` inside
  `sync_weights() → wake_up()`. Turning it off also makes the trl#5312 patch
  inert, since both it and TRL's buggy `reload_weights` sit behind the same
  `enable_sleep_mode` guard. Cost: 48.6 GB → 88.7 GB steady state.
- **A training script must support resume from day one.** `trainer.train()`
  with no arguments cannot continue a crashed run; `train_grpo_v5.py` takes
  `RESUME_FROM=auto` (or a checkpoint path). A crash costs up to `save_steps`
  of progress — at 500 steps × ~38s that is ~5 hours.
- **In tmux, detach with `Ctrl-B` then `D`.** Typing `exit` inside the window
  ends the shell, and the last window closing destroys the session (harmless
  for data — checkpoints and logs are on disk — but the scrollback with the
  crash traceback goes with it).
- **Smoke-test copies must redirect every output path**, including the final
  `trainer.save_model(OUTPUT_DIR)` — a sed-generated smoke copy once wrote
  into the real run directory. Also `PYTHONPATH=src/model_training` when
  running a copy from `tmp/`, or `reward_v5` will not import.

---

## HuggingFace repos

| Repo | Contents | Use case |
|------|----------|----------|
| [brightonliuzZ/qwen3-8b-go-v4](https://huggingface.co/brightonliuzZ/qwen3-8b-go-v4) | Full-precision merged model + `qwen3-8b-go-v4-Q4_K_M.gguf` | **Primary deployment model** |
| [brightonliuzZ/qwen2.5-7b-go](https://huggingface.co/brightonliuzZ/qwen2.5-7b-go) | Legacy 7B full-precision merged model + `qwen-7b-go-Q4_K_M.gguf` | Legacy deployment |
| [brightonliuzZ/grpo-go-checkpoints](https://huggingface.co/brightonliuzZ/grpo-go-checkpoints) | Raw LoRA checkpoints: `v1/checkpoint-800`, `v2/checkpoint-1000`, `v3/checkpoint-500` | Resuming / research |
| [brightonliuzZ/katago-grpo-datasets](https://huggingface.co/datasets/brightonliuzZ/katago-grpo-datasets) | `training_data_augmented_v3.jsonl` | Data / research |
| [brightonliuzZ/qwen2.5-7b-and-qwen3-8b-go-GGUF-legacy](https://huggingface.co/brightonliuzZ/qwen2.5-7b-and-qwen3-8b-go-GGUF-legacy) | Older GGUF releases | Historical reference |

### What are the non-GGUF files in a model repo?

Each model repo (`qwen3-8b-go-v4`, `qwen2.5-7b-go`) contains two categories:

**For most users — just download the `.gguf` file** and load it in LM Studio.

**Full-precision files (for fine-tuning / re-quantizing only):**

| File(s) | Purpose |
|---------|---------|
| `model-000XX-of-000XX.safetensors` | Model weights shards (~5 GB each). Needed for training or re-quantizing. |
| `config.json` | Architecture config. Required to load in Python (`transformers`). |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | Tokenizer — text ↔ token IDs. Required for training and Python inference. |
| `generation_config.json` | Default generation params (temperature, max tokens, etc.). |
| `chat_template.jinja` | Conversation formatting template. |
| `added_tokens.json`, `special_tokens_map.json` | Special tokens added during fine-tuning. |
| `.gitattributes` | Marks safetensors shards as git-lfs blobs. Not a model file. |

---

## Server notes (yumingz5-linux-ml)

Account: `tyliu`. No sudo access. Shared server — do not run jobs that monopolize GPU without coordination.

### File permission issue (Docker-created files)

Training runs and LoRA merge scripts run inside Docker as root. Files they create are owned by **root**, so `rm -rf` on those directories fails for `tyliu`.

**Fix — use Docker to delete:**
```bash
docker run --rm \
  -v /home/tyliu/katago_workspace:/workspace \
  alpine rm -rf /workspace/<path-to-delete>
```
`alpine` is a ~5MB image that auto-pulls. If no internet, substitute with an already-present image (e.g. `katago-llm`).

Regular `rm` (single files) works as long as the **parent directory** is `tyliu`-owned — it will prompt for write-protected files but succeed.

### HuggingFace upload gotchas on this server

- **Always add `--repo-type=model`** (or `dataset`) to both `hf upload` and `hf upload-large-folder` — the latter errors without it.
- **`hf upload-large-folder` writes metadata inside the source folder.** If the source folder is root-owned (Docker-created), this fails. Use `hf upload` instead — it does not write into the source directory.
- **Xet cache may be root-owned** (`~/.cache/huggingface/xet/`). If `hf upload` crashes on xet permission errors, redirect:
  ```bash
  mkdir -p ~/xet_cache_new
  export HF_XET_CACHE=~/xet_cache_new
  ```
  To make permanent: `echo 'export HF_XET_CACHE=~/xet_cache_new' >> ~/.bashrc`
- **Repo rename** (no CLI command — use Python):
  ```python
  from huggingface_hub import HfApi
  HfApi().move_repo(from_id='user/old-name', to_id='user/new-name', repo_type='model')
  ```

---

## Local workspace layout (post-recovery, 2026-08-02)

Everything below was verified present on 2026-08-02 after the workspace was
rebuilt. See [`docs/RECOVERY-2026-05-06.md`](./docs/RECOVERY-2026-05-06.md) for
what happened and how each item got here.

| Path | Size | Role |
|------|------|------|
| `data/training_data_v4.jsonl` | 0.8 G | **Active training dataset** — 113,404 records |
| `data/training_data_augmented_shuffled_v3.jsonl` | 1.8 G | v4 dataset source — 270,000 records |
| `data/training_data_augmented_v3.jsonl` | 1.8 G | From HF; the only surviving dataset of the incident |
| `data/training_data_quality_filtered_v3.jsonl` | 0.1 G | 16,875 records, reconstructed from the augmented file |
| `runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-5000` | | v4 LoRA, furthest trained (step 5000, epoch 0.35) |
| `runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-1500` | | v4 LoRA (step 1500, epoch 0.11) |
| `runs/Qwen2.5-7B-GRPO-Go-Pro-v1/checkpoint-800` | | v1 latest (also on HF) |
| `runs/Qwen2.5-7B-GRPO-Go-Pro-v2/checkpoint-1000` | | v2 latest (also on HF) |
| `qwen3-8b-go-v4-Q4_K_M.gguf` | 5.0 G | Deployment GGUF (from HF; built from the lost checkpoint-3000) |
| ~~`merged-qwen3-8b-go-v4/`~~ | — | Full-precision merged v4 — **deleted 2026-08-03 to free disk**. Re-download from HF `brightonliuzZ/qwen3-8b-go-v4` if needed (see `docs/RECOVERY-2026-05-06.md` for the command). |
| `KataGo_engine/katago` + `KataGo18b9x9.gz` | 0.2 G | Engine for data generation. **Needs CUDA 12 + cuDNN 9 — run it inside Docker, not on the bare host.** |
| `KataGo-1.16.4/`, `llama.cpp/`, `llama_env/`, `wandb/`, `squashfs-root/` | | Survived the incident untouched |

**Not present, and not recoverable** (details in the recovery doc):
`data/json_output_with_topk_official.jsonl`, `data/training_ready_data_v3.jsonl`,
`runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-3000`, all
`src/model_training/logs/grpo_training_rollouts_*.jsonl`,
`model_quantization/`, `src/interception/python/`,
`src/data_acquiring/old_scripts/`, `demo_videos/`, the `katallm:v2` Docker image.

**Archived to HF in the 2026-04-06 cleanup** (deliberate, and it is why recovery
was possible at all): `merged-qwen3-8b-go-v4/` and `merged-qwen-7b-go/` →
`qwen3-8b-go-v4` / `qwen2.5-7b-go`; `qwen-7b-go-Q4_K_M.gguf` → `qwen2.5-7b-go`;
`training_data_augmented_v3.jsonl` → `katago-grpo-datasets`. FP16 GGUF
intermediates were deleted as regenerable.

---

## Progress snapshot (as of 2026-08-02)

| Item | Status |
|------|--------|
| v3 data + augmentation | Complete — on HF |
| v4 dataset | Complete — `training_data_v4.jsonl`, 113,404 rows |
| v4 training | Complete — checkpoints 1500 and 5000 local |
| v4 GGUF export | Complete — `qwen3-8b-go-v4-Q4_K_M.gguf` locally + on HF |
| v4 Lizzie integration | Complete — `gtp_proxy_v4.exe` with board tracking, capture logic, reasoning output |
| Base model in Lizzie | Complete — `gtp_proxy_v4_base.exe` with `Qwen3-8B-Q4_K_M.gguf` |
| v1 7B GRPO | Complete — W&B run `hardy-river-3` |
| v1 7B edge deployment | Complete — `qwen-7b-go-Q4_K_M.gguf` on HF |
| Workspace cleanup (2026-04-06) | Complete — ~150G freed, artifacts archived to HF |
| **Workspace recovery (2026-08-02)** | **Complete — see `docs/RECOVERY-2026-05-06.md`** |
| exp01–04 analysis (2026-08-03) | Complete — `notebooks/exp01-04_analysis.ipynb`, W&B curves cached |
| exp05 (v5) preparation (2026-08-03) | Complete — reward+script+tests+split+Dockerfile ready (ADR 0006) |
| exp05 run (2026-08-04→05) | **Stopped at ~1490 steps** — flatline; root cause = ~5× effective-step-size deficit (ADR 0006 Amendment 2); weight sync exonerated by `tmp/discriminate_v5.py` |
| exp05b run (2026-08-05→06) | **Stopped at ~2000 steps** — flat again; trainer learned (`‖lora_B‖`=1.05) but vLLM generated from frozen base: TRL 1.9.2 sleep-mode `reload_weights` bug (trl#5312, ADR 0006 Amendment 3) |
| **exp05c (v5c, 2026-08-06 →)** | **Running (W&B `lc4hyp23`) — first curve that learns: core 0.401→0.492 @1500 steps, eval core 0.4853→0.5059, entropy 0.52 stable. Crashed once at step 1616 (`none_dealloc`, random C-extension bug); resumed. Now launched via `run_with_restart.sh` (ADR 0006 Amendment 4)** |

### Known gaps to close

- `checkpoint-5000` has never been merged, quantized or played against. The
  published GGUF is from the lost `checkpoint-3000`, so the best surviving
  adapter is currently unexercised.
- `checkpoint-1500` and `checkpoint-5000` exist only on this server. Upload them
  to `brightonliuzZ/grpo-go-checkpoints` — that is the exact gap that made
  `checkpoint-3000` unrecoverable.
- v4 training reached epoch 0.35 of a 270,000-step schedule. Resuming is possible
  from `checkpoint-5000` but needs `data/training_data_v4.jsonl`, which is now
  back in place.
