# CLAUDE.md — KataGo LLM workspace

Single source of truth for this repo. Prefer this over stale comments in random files.

---

## What this repo is

- **Goal:** Train LLMs to play **9×9 Go** with **GRPO** (Group Relative Policy Optimization), using **KataGo** as the teacher.
- **Current training stack:** **Qwen/Qwen3-8B** — primary script is **`train_grpo_v4.py`** (dataset `data/training_data_v4.jsonl`).
- **Edge deployment (proven):** v4 checkpoint → GGUF Q4_K_M → **LM Studio** → **C++ GTP proxy** → **Lizzie** (Go GUI). Also supports loading the **un-fine-tuned base model** for comparison.
- **Legacy:** `train_grpo_v3.py` (v3 dataset), `train_grpo.py` (Qwen2.5-7B v1).

---

## Git conventions

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

This builds **`data/training_data_v4.jsonl`**: restores **valid empty coordinates** in each prompt (stripped by augmentation), **downsamples lopsided** positions (root winrate above 0.95 or below 0.05) to **30%**, keeps balanced positions fully, then shuffles (~113k rows as of last run).

---

## Training

**Primary — v4 (recommended):**
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
python test_model.py
python src/model_training/test_base_model.py
python src/model_training/test_reward.py            # no GPU
python src/model_training/test_model_interactive.py  # must use enable_thinking=False + v4 system prompt if testing v4
```

---

## Reward stacks (short)

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
| `train_grpo_v4.py` | `DATASET_PATH`, `OUTPUT_DIR` |
| `train_grpo_v3.py` | `DATASET_PATH`, `OUTPUT_DIR` |
| `train_grpo.py` | `DATASET_PATH`, `OUTPUT_DIR` |
| `filter_9x9.py` | `INPUT_DIR`, `OUTPUT_DIR` |
| `prepare_v4_dataset.py` | `INPUT_FILE`, `OUTPUT_FILE`, lopsided thresholds |
| `export_gguf_model_v4.sh` | `LORA_CKPT_PATH`, `BASE_MODEL` |
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

## Local workspace layout (post-cleanup, 2026-04-06)

| Path | Role | Keep? |
|------|------|-------|
| `data/training_data_v4.jsonl` | Active training dataset | Yes |
| `data/json_output.jsonl` | KataGo-enriched JSONL (v3 pipeline source) | Yes |
| `data/training_ready_data_v3.jsonl` | v3 pipeline intermediate | Yes |
| `data/training_data_quality_filtered_v3.jsonl` | v3 quality-filtered intermediate | Yes |
| `runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-1500` | v4 LoRA checkpoint | Yes |
| `runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-5000` | v4 latest LoRA checkpoint | Yes |
| `runs/Qwen2.5-7B-GRPO-Go-Pro-v1/checkpoint-800` | v1 latest (also on HF) | Yes |
| `runs/Qwen2.5-7B-GRPO-Go-Pro-v2/checkpoint-1000` | v2 latest (also on HF) | Yes |
| `qwen3-8b-go-v4-Q4_K_M.gguf` | Active deployment GGUF (also on HF) | Yes |
| `KataGo_engine/` | KataGo binary + weights for data generation | Yes |
| `llama.cpp/` | Used by export script to quantize | Yes |
| `model_quantization/` | `merge_lora.py` + legacy 7B GGUFs | Partial |

**Deleted / archived to HF:**
- `merged-qwen3-8b-go-v4/` → HF `qwen3-8b-go-v4`
- `merged-qwen-7b-go/` (both copies) → HF `qwen2.5-7b-go`
- `qwen3-8b-go-v4-f16.gguf` — deleted (quantization intermediate)
- `model_quantization/qwen-7b-go-f16.gguf` — deleted (quantization intermediate)
- `model_quantization/qwen-7b-go-Q4_K_M.gguf` → HF `qwen2.5-7b-go`
- `data/training_data_augmented_v3.jsonl` → HF `katago-grpo-datasets`
- `data/training_data_augmented_shuffled_v3.jsonl` — deleted
- v1/v2/v3 non-latest checkpoints — deleted

---

## Progress snapshot (as of 2026-04-06)

| Item | Status |
|------|--------|
| v3 data + augmentation | Complete — archived to HF |
| v4 dataset | Complete — `training_data_v4.jsonl` (~113k rows) |
| v4 training | Complete — checkpoints 1500 and 5000 kept locally |
| v4 GGUF export | Complete — `qwen3-8b-go-v4-Q4_K_M.gguf` locally + on HF |
| v4 Lizzie integration | Complete — `gtp_proxy_v4.exe` with board tracking, capture logic, reasoning output |
| Base model in Lizzie | Complete — `gtp_proxy_v4_base.exe` with `Qwen3-8B-Q4_K_M.gguf` |
| v1 7B GRPO | Complete — W&B run `hardy-river-3` |
| v1 7B edge deployment | Complete — `qwen-7b-go-Q4_K_M.gguf` on HF |
| Workspace cleanup | Complete — ~150G freed, all artifacts archived to HF |
