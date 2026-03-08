# KataGo-LLM-Team

**Teaching a Large Language Model to Play 9×9 Go through Reinforcement Learning from AI Feedback (RLAIF)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)
[![TRL](https://img.shields.io/badge/TRL-0.29-blueviolet.svg)](https://github.com/huggingface/trl)
[![W&B Run](https://img.shields.io/badge/W%26B-training%20run-orange.svg)](https://wandb.ai/brighton_zz-uc-san-diego/huggingface/runs/eb6zhzfo)

---

Large language models are not naturally equipped for spatial board games like Go — they lack built-in 2D coordinate reasoning, board legality awareness, and any understanding of territory or life-and-death. This project bridges that gap by using [KataGo](https://github.com/lightvector/KataGo) as an expert teacher and training [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with **Group Relative Policy Optimization (GRPO)** to produce legal, tactically sound moves on a 9×9 board.

The key insight is an **outcome-regret reward formulation**: instead of rewarding the model for absolute position quality, it penalizes the *gap* between what the model played and what KataGo would have played, scaled by a regret coefficient. This forces the model to learn not just "play somewhere reasonable" but "play the best available move."

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Reward Function Design](#4-reward-function-design)
5. [Model Training](#5-model-training)
6. [Setup & Installation](#6-setup--installation)
7. [Running the Full Pipeline](#7-running-the-full-pipeline)
8. [Testing the Model](#8-testing-the-model)
9. [GTP Proxy — Live Game Capture](#9-gtp-proxy--live-game-capture)
10. [Results](#10-results)
11. [Citations](#11-citations)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────┐
│  Raw SGF Files  (9x9 Go games)      │
└──────────────┬──────────────────────┘
               │  filter + convert
               ▼
┌─────────────────────────────────────┐
│  json_output.jsonl                  │
│  {id, moves, initialStones, rules}  │
└──────────────┬──────────────────────┘
               │  KataGo analysis
               ▼
┌─────────────────────────────────────┐
│  json_output_with_topk.jsonl        │
│  + top-10 moves with winrates       │
└──────────────┬──────────────────────┘
               │  format prompts
               ▼
┌─────────────────────────────────────┐
│  training_ready_data.jsonl          │
│  {user_prompt, katago_all,          │
│   katago_best}                      │
└──────────────┬──────────────────────┘
               │  quality filter → augment → shuffle
               ▼
┌─────────────────────────────────────┐
│  training_data_augmented_           │
│  shuffled.jsonl  (~5 000 examples)  │
└──────────────┬──────────────────────┘
               │  GRPO training (TRL)
               ▼
┌─────────────────────────────────────┐
│  Qwen2.5-7B-Instruct + LoRA        │
│  4-layer reward · outcome-regret   │
└──────────────┬──────────────────────┘
               │  inference
               ▼
     <think> … </think>
     MOVE: C4
     EXPLAIN: Secures the corner.
```

---

## 2. Repository Structure

```
KataGo-LLM-Team/
│
├── Dockerfile              # Production container (RTX Ada, vLLM, Flash Attention 2)
├── Dockerfile.local        # Dev container (lighter, 4-bit quant, no vLLM)
├── test_model.py           # Quick inference test with a loaded LoRA checkpoint
├── grpo_training_rollouts.jsonl  # Sampled training rollouts with reward breakdowns
│
├── data/
│   ├── 20_70_filtered/     # Filtered 9x9 SGF source files (hash-named)
│   ├── json_output.jsonl                        # Stage 1 output: parsed games
│   ├── json_output_with_topk.jsonl              # Stage 2 output: + KataGo analysis
│   ├── training_ready_data.jsonl                # Stage 3 output: formatted prompts
│   ├── training_data_quality_filtered.jsonl     # Stage 4 output: 312 curated examples
│   ├── training_data_augmented.jsonl            # Stage 5 output: 16× augmented
│   └── training_data_augmented_shuffled.jsonl   # Stage 6 output: final training set
│
├── KataGo_engine/
│   ├── katago.exe          # KataGo v1.16.4 Windows binary
│   ├── KataGo18b9x9.gz    # 9x9-specialised neural network weights
│   ├── analysis_example.cfg
│   └── default_gtp.cfg
│
├── src/
│   ├── data_acquiring/
│   │   ├── preprocessing/
│   │   │   ├── filter_9x9.py              # Filter SGFs by board size (SZ=9)
│   │   │   ├── sgf_to_jsonl.py            # Convert SGF → structured JSONL
│   │   │   └── total_move_histogram.py    # EDA: move-count distribution
│   │   ├── get_topk_from_katago/
│   │   │   ├── run_katago_analysis.py     # Query KataGo, extract top-K winrates
│   │   │   └── analysis.cfg              # KataGo analysis config (500 visits)
│   │   └── get_train_ready_data/
│   │       ├── prepare_training_data.py   # Render ASCII board + build prompt
│   │       ├── data_quality_cut.py        # Curate 312 tactical examples
│   │       ├── data_augmentation.py       # D4 symmetry augmentation (16×)
│   │       ├── shuffle_data.py            # Shuffle with seed 42
│   │       └── EDA/
│   │           └── best_winrate_distribution.ipynb
│   │
│   ├── interception/
│   │   ├── my_gtp_proxy.py    # Python GTP proxy (Lizzie ↔ KataGo interceptor)
│   │   ├── gtp_proxy.cpp      # C++ version of same proxy
│   │   ├── gtp_proxy.exe      # Pre-compiled Windows binary
│   │   ├── CMakeLists.txt
│   │   └── GTP_PROXY_GUIDE.md
│   │
│   └── model_training/
│       ├── train_grpo.py          # Main training script (Qwen2.5-7B, RTX Ada)
│       ├── train_grpo_laptop.py   # Memory-optimised variant (Qwen2.5-0.5B, 4-bit)
│       ├── train_grpo_local.py    # Smoke-test variant (7B, 5 steps, small dataset)
│       ├── test_base_model.py     # Baseline: un-fine-tuned model on random boards
│       ├── test_model.py          # Inference with loaded LoRA checkpoint
│       └── test_reward.py         # Unit tests for reward function parsing logic
│
└── MyModel/
    └── Qwen2.5-7B-GRPO-Go-Pro-v1/   # Saved LoRA adapters + training artefacts
```

---

## 3. Data Pipeline

### Board Coordinate System

```
   A B C D E F G H J       ← Columns  (A–J, skipping I to avoid confusion with 1)
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .       'X' = Black stone
 6 . . . . . . . . .       'O' = White stone
 5 . . . . . . . . .       '.' = empty intersection
 4 . . . . . . . . .
 3 . . . . . . . . .       Rows 1–9 (1 = bottom, 9 = top)
 2 . . . . . . . . .
 1 . . . . . . . . .
```

A move is written as `[Column][Row]`, e.g. `C4`, `H9`, `PASS`.

---

### Stage 1 — SGF Filtering & Conversion

| Script | Input | Output |
|--------|-------|--------|
| `filter_9x9.py` | Raw SGF directory | Deduplicated 9×9-only SGFs (SHA-1 named) |
| `sgf_to_jsonl.py` | Filtered SGF files | `json_output.jsonl` |

`json_output.jsonl` record format:
```json
{
  "id": "00049257e1c84e64...",
  "boardXSize": 9,
  "boardYSize": 9,
  "rules": "chinese",
  "komi": 7.5,
  "initialStones": [],
  "moves": [["B","D4"],["W","F6"],...],
  "analyzeTurns": [0, 1, 2, ...]
}
```

---

### Stage 2 — KataGo Analysis

| Script | Input | Output |
|--------|-------|--------|
| `run_katago_analysis.py` | `json_output.jsonl` | `json_output_with_topk.jsonl` |

The script spawns KataGo as a subprocess in analysis mode (500 visits per position, 6 search threads). For each position it receives `moveInfos` and extracts the **top-10 moves** with their winrates and principal variations.

Key config (`analysis.cfg`):
```
maxVisits = 500
numAnalysisThreads = 12
numSearchThreads = 6
reportAnalysisWinratesAs = SIDETOMOVE
```

---

### Stage 3 — Prompt Formatting

| Script | Input | Output |
|--------|-------|--------|
| `prepare_training_data.py` | `json_output_with_topk.jsonl` | `training_ready_data.jsonl` |

Each record becomes:
```json
{
  "original_id": "00049257_pos12",
  "user_prompt": "[Current 9x9 Board State]\n   A B C D E F G H J\n 9 . . . . . . . . .\n...\n[Match History]\n...\n[Instruction]\nPlayer to move: White.",
  "katago_all": {"D6": 0.621, "E5": 0.598, "C5": 0.541, ...},
  "katago_best": 0.621
}
```

---

### Stage 4 — Quality Filtering

| Script | Input | Output |
|--------|-------|--------|
| `data_quality_cut.py` | `training_ready_data_shuffled.jsonl` | `training_data_quality_filtered.jsonl` |

Applies three filters to select **312 tactically rich examples**:

1. **Competitive winrate** — best winrate between 0.20 and 0.80, *or* regret > 0.15 (critical positions)
2. **Move count** — between 8 and 60 (excludes opening theory and terminal filling)
3. **Stratified sampling** — balances across game phase (Early / Mid / Late) × position type (Advantageous / Even / Disadvantageous)

A custom **sharpness score** (`Top1_wr − mean(Top2…5_wr)`) is computed to bias toward positions where the best move genuinely matters.

---

### Stage 5 — D4 Symmetry Augmentation

| Script | Input | Output |
|--------|-------|--------|
| `data_augmentation.py` | `training_data_quality_filtered.jsonl` | `training_data_augmented.jsonl` |

The 9×9 board has an 8-element symmetry group (D4 — four rotations × two reflections). For each of the 312 base positions, 8 transformed boards are generated, then the whole set is **colour-flipped** (Black ↔ White), yielding **16 variants per example → ~5 000 training examples**.

Winrates are preserved as "winrate of player to move" (no inversion needed on colour flip).

---

### Stage 6 — Shuffle

| Script | Input | Output |
|--------|-------|--------|
| `shuffle_data.py` | `training_data_augmented.jsonl` | `training_data_augmented_shuffled.jsonl` |

Fixed seed (`seed=42`) for reproducibility.

---

## 4. Reward Function Design

Training uses four reward functions composed inside `GRPOTrainer`. Each receives the model's generated text alongside the KataGo ground truth for that position.

### Layer 1 — Format & Spatial Legality

```
Score range: [−1.0, +0.2]
```

| Check | Score |
|-------|-------|
| `<think>…</think>` tags present | +0.2 |
| Valid move extracted (e.g. `C4`, `PASS`) | ±0.0 |
| No valid move found | −1.0 |
| Move lands on occupied stone (`X` or `O`) | −0.8 |

The **spatial collision detector** parses the ASCII board in the prompt and checks whether the target coordinate is `.` (empty) before granting legality.

---

### Layer 2 — Thinking Quality & Consistency

```
Score range: [−0.5, +0.5]
```

| Check | Score |
|-------|-------|
| Final move **not mentioned** in `<think>` block | −0.5 |
| Final move **is mentioned** in `<think>` block | +0.2 |
| Thinking length 50–800 chars (sweet spot) | +0.1 |
| Zero-length thinking | −0.5 |
| Thinking > 800 chars (rambling) | −0.1 |
| Move appears in KataGo top-10 candidates | +0.2 |

This layer enforces *chain-of-thought coherence*: a model that thinks about move `D6` but outputs `C4` is penalised for logical inconsistency.

---

### Layer 3 — Outcome Regret (Core Reward)

```
Score range: [−1.0, +1.0]
```

$$\text{reward} = \text{wr}_{\text{chosen}} - 1.5 \times (\text{wr}_{\text{best}} - \text{wr}_{\text{chosen}})$$

Where:
- $\text{wr}_{\text{best}}$ = KataGo's best available winrate for this position
- $\text{wr}_{\text{chosen}}$ = KataGo's winrate for the move the model actually played
- *Regret* $= \text{wr}_{\text{best}} - \text{wr}_{\text{chosen}}$: how much winrate was left on the table

If the model's move is **not in KataGo's top-10 candidates**, a fallback penalty applies:

$$\text{reward} = \max(0,\; \text{wr}_{\text{worst\_candidate}} - 0.1) - 1.5 \times \text{regret} - 0.2$$

The 1.5× regret multiplier ensures that sub-optimal but still-reasonable moves receive lower rewards than optimal ones, even in winning positions. This breaks the degenerate strategy of "play any legal move in a won game."

---

### Layer 4 — Logging (Non-Scoring)

This function never changes the model's gradient. It randomly samples rollouts and writes structured entries to `grpo_training_rollouts.jsonl` for offline analysis, including the board state, full model generation, extracted move, KataGo ground truth, and a partial reward breakdown.

---

## 5. Model Training

### Base Model & Method

| Item | Value |
|------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Training method | GRPO (Group Relative Policy Optimization) |
| Fine-tuning | LoRA (`r=16`, `lora_alpha=32`) |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Precision | `bf16` |

### Key Hyperparameters (`train_grpo.py`)

| Parameter | Value |
|-----------|-------|
| `learning_rate` | `5e-6` |
| `per_device_train_batch_size` | `8` |
| `num_generations` | `8` |
| `gradient_accumulation_steps` | `2` (effective batch = 16) |
| `max_completion_length` | `1024` |
| `warmup_ratio` | `0.1` |
| `lr_scheduler_type` | `cosine` |
| `max_steps` | `1000` |
| `lora_dropout` | `0.05` |

### Training Script Variants

| Script | Model | GPU Requirement | Use Case |
|--------|-------|-----------------|----------|
| `train_grpo.py` | Qwen2.5-**7B** | ≥ 48 GB VRAM (e.g. RTX 6000 Ada) | Full-scale training |
| `train_grpo_laptop.py` | Qwen2.5-**0.5B** | ≤ 12 GB VRAM, 4-bit quant | Laptop / consumer GPU |
| `train_grpo_local.py` | Qwen2.5-**7B** | 4-bit quant, 5 steps only | Local smoke-test |

### System Prompt (enforced at inference and training time)

```
You are an expert 9x9 Go (Weiqi) Player. You must determine the best tactical next move.
...
OUTPUT FORMAT:
<think>
Step 1: Analyze the global board state, evaluating territory, influence, and any urgent tactical situations.
Step 2: Propose 2 to 3 candidate moves and briefly compare their pros and cons.
Step 3: VERIFY that your final intended move is currently an empty '.' on the board.
</think>
MOVE: [Coordinate]
EXPLAIN: [1-2 short sentences explaining the strategy.]
```

---

## 6. Setup & Installation

### Prerequisites

- **NVIDIA GPU**
  - Minimum: 8 GB VRAM (laptop variant only, Qwen2.5-0.5B)
  - Recommended for full 7B training: 48 GB VRAM (RTX 6000 Ada or equivalent)
- **Docker** (recommended) — or Python 3.10+ with conda/venv
- **KataGo engine + 9x9 model weights** (see [Step D](#d-katago-engine-setup) below)
- **Windows or Linux** (KataGo binary in `KataGo_engine/` is a Windows `.exe`; Linux users should download the appropriate KataGo binary from the [KataGo releases page](https://github.com/lightvector/KataGo/releases))

---

### A. Clone the Repository

```bash
git clone https://github.com/BrightonLiu-zZ/KataGo-LLM-Team.git
cd KataGo-LLM-Team
```

---

### B. Option 1: Docker (Recommended)

**Production** (RTX Ada, CUDA 12.1, PyTorch 2.4, vLLM, Flash Attention 2):

```bash
docker build -f Dockerfile -t katago-llm .
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -w /workspace \
  katago-llm bash
```

**Local development** (PyTorch 2.1, 4-bit quantization, no vLLM):

```bash
docker build -f Dockerfile.local -t katago-llm-local .
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -w /workspace \
  katago-llm-local bash
```

---

### C. Option 2: Manual Install (conda / pip)

```bash
conda create -n katago-llm python=3.10 -y
conda activate katago-llm

pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl
pip install bitsandbytes pandas scipy wandb sgfmill
```

> For production training with Flash Attention 2:
> ```bash
> pip install flash-attn --no-build-isolation
> ```

---

### D. KataGo Engine Setup

1. **Windows**: The repository already includes `KataGo_engine/katago.exe` (v1.16.4).  
   **Linux/macOS**: Download the appropriate binary from the [KataGo releases page](https://github.com/lightvector/KataGo/releases) and place it at `KataGo_engine/katago`.

2. Download the **9×9-specialised model weights** (`KataGo18b9x9.gz`) from the [KataGo model page](https://katagotraining.org/networks/) and place it at `KataGo_engine/KataGo18b9x9.gz`.

3. Edit `src/data_acquiring/get_topk_from_katago/analysis.cfg` to confirm the paths:
   ```
   # Verify these three lines match your local setup:
   KATAGO_EXE  =  KataGo_engine/katago.exe
   CONFIG_FILE =  src/data_acquiring/get_topk_from_katago/analysis.cfg
   MODEL_FILE  =  KataGo_engine/KataGo18b9x9.gz
   ```
   Also update the hardcoded paths in `run_katago_analysis.py` if they differ from the defaults.

---

## 7. Running the Full Pipeline

Run the following commands **in order** from the repository root. Each step's output is the next step's input.

```bash
# Step 1 — Filter raw SGF files to 9x9 only
#   Edit INPUT_DIR / OUTPUT_DIR inside the script to point to your SGF collection.
python src/data_acquiring/preprocessing/filter_9x9.py

# Step 2 — Convert SGF files to structured JSONL
python src/data_acquiring/preprocessing/sgf_to_jsonl.py
# Output: data/json_output.jsonl

# Step 3 — Query KataGo for top-10 moves per position
python src/data_acquiring/get_topk_from_katago/run_katago_analysis.py
# Output: data/json_output_with_topk.jsonl
# Note: set MAX_LIMIT in the script (default 10 000) to control how many positions to analyse.

# Step 4 — Format board states into training prompts
python src/data_acquiring/get_train_ready_data/prepare_training_data.py
# Output: data/training_ready_data.jsonl

# Step 5 — Shuffle (pre-augmentation)
python src/data_acquiring/get_train_ready_data/shuffle_data.py
# Output: data/training_ready_data_shuffled.jsonl

# Step 6 — Quality filtering: select 312 tactically rich positions
python src/data_acquiring/get_train_ready_data/data_quality_cut.py
# Output: data/training_data_quality_filtered.jsonl

# Step 7 — D4 symmetry augmentation (16× expansion)
python src/data_acquiring/get_train_ready_data/data_augmentation.py
# Output: data/training_data_augmented.jsonl

# Step 8 — Final shuffle
python src/data_acquiring/get_train_ready_data/shuffle_data.py
# Output: data/training_data_augmented_shuffled.jsonl

# Step 9 — Train
#   High-end GPU (7B, ~1 000 steps):
python src/model_training/train_grpo.py

#   Laptop / consumer GPU (0.5B, 4-bit):
python src/model_training/train_grpo_laptop.py

#   Quick smoke-test (7B, 5 steps, first 50 examples):
python src/model_training/train_grpo_local.py
```

Training checkpoints are saved to `runs/` every 100 steps. Training rollouts are logged to `grpo_training_rollouts.jsonl`.

---

## 8. Testing the Model

### Test the fine-tuned model (LoRA checkpoint)

```bash
python test_model.py
```

This loads `Qwen/Qwen2.5-7B-Instruct` + the LoRA adapters saved in `runs/` and queries it on a sample board position. Expected output:

```
<think>
Step 1: Black has a cutting stone at E5 threatening to split White's groups...
Step 2: Candidate moves — F5 (defend the cut), D6 (expand territory), C4 (corner approach).
Step 3: Verifying F5 is '.': yes, it is empty.
</think>
MOVE: F5
EXPLAIN: Defends the key cutting point and maintains connection.
```

### Test the base model (no fine-tuning)

```bash
python src/model_training/test_base_model.py
```

Loads the raw Qwen2.5-7B with 4-bit quantisation. Use this to compare against the fine-tuned version.

### Validate reward functions

```bash
python src/model_training/test_reward.py
```

Runs unit tests on the coordinate extraction regex and spatial legality checker. No GPU required.

---

## 9. GTP Proxy — Live Game Capture

The `src/interception/` module contains a **GTP proxy** that sits transparently between **Lizzie** (a Go GUI) and **KataGo**, intercepting every move and saving the game alongside KataGo's analysis to `game_data.json` in real time.

```
Lizzie ──► gtp_proxy.exe ──► KataGo
           (saves moves)
               │
               ▼
         game_data.json
```

This is used to generate additional training data from real games played against KataGo via the GUI. See the full setup guide: [src/interception/GTP_PROXY_GUIDE.md](src/interception/GTP_PROXY_GUIDE.md).

---

## 10. Results

Training was tracked with Weights & Biases:  
**[View training run on W&B →](https://wandb.ai/brighton_zz-uc-san-diego/huggingface/runs/eb6zhzfo)**

The final model checkpoint is saved at `MyModel/Qwen2.5-7B-GRPO-Go-Pro-v1/` (trained with TRL 0.29.0, Transformers 4.57.6, PyTorch 2.9.1).

---

## 11. Citations

**GRPO / DeepSeekMath**
```bibtex
@article{shao2024deepseekmath,
  title   = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author  = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and
             Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
  year    = {2024},
  eprint  = {arXiv:2402.03300}
}
```

**TRL**
```bibtex
@software{vonwerra2020trl,
  title   = {TRL: Transformers Reinforcement Learning},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and
             Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and
             Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```

**KataGo**
```bibtex
@article{wu2019accelerating,
  title   = {Accelerating Self-Play Learning in Go},
  author  = {David J Wu},
  year    = {2019},
  eprint  = {arXiv:1902.10565}
}
```

**Qwen2.5**
```bibtex
@article{qwen2025qwen25,
  title   = {Qwen2.5 Technical Report},
  author  = {Qwen Team},
  year    = {2025},
  url     = {https://arxiv.org/abs/2412.15115}
}
```
