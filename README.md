Here’s a clean, copy-paste-ready `README.md` that documents the full pipeline—from data acquiring to preprocessing/quality-filtering to GRPO model training—using the files you listed.

---

# KataGo × LLM Project

**Goal:** teach an instruction-tuned LLM to suggest and *explain* strong 9×9 Go moves, using KataGo to label positions and a Group Relative Policy Optimization (GRPO) trainer to align the model with KataGo-derived rewards.

<p align="center">
  <img alt="pipeline" src="https://dummyimage.com/820x2/eee/eee.png&text=">
</p>

## TL;DR pipeline

**Skeleton of the end-to-end process**

1) **Filter to 9×9 SGFs**  
   - **Script:** `script/data_acquiring/filter_9x9.py`  
   - **Input:** Raw SGF folder (mixed board sizes)  
   - **Output:** 9×9-only SGF folder

2) **Inspect game lengths & pick window**  
   - **Script:** `script/data_acquiring/total_move_histogram.py`  
   - **Input:** 9×9 SGF folder  
   - **Output:** Histogram → choose move-count window (e.g., 20–70)

3) **Subsample by move range (balanced)**  
   - **Script:** `script/data_acquiring/pick_random_sgf_based_on_total_move.py`  
   - **Input:** 9×9 SGF folder, chosen window  
   - **Output:** `sgf_list_20_70.txt` (optionally archived as `data/20_70_filtered.7z`)

4) **Label positions with KataGo (top-k)**  
   - **Script:** `script/data_acquiring/label_and_select.py`  
   - **Uses:** `src/katago.exe`, `script/data_acquiring/analysis.cfg`, `src/KataGo18b9x9.gz`  
   - **Output:** `katago_labeled_raw.jsonl`

5) **Compute quality metrics**  
   - **Script:** `script/data_acquiring/compute_metrics_from_topk.py`  
   - **Input:** `katago_labeled_raw.jsonl`  
   - **Output:** `katago_labeled_with_metrics.jsonl`

6) **Filter & build final training JSONL**  
   - **Script:** `script/data_acquiring/build_filtered_training_zh_final.py`  
   - **Input:** `katago_labeled_with_metrics.jsonl`  
   - **Output:** `data/cleaned_katago_output.jsonl` (keeps diverse/meaningful states, downweights one-sided)

7) **Train the model (GRPO)**  
   - **Script (baseline):** `script/model_training/model_training_first_trial.py`  
   - **Script (current):** `grpo_vlm.py`  
   - **Input:** `data/cleaned_katago_output.jsonl`  
   - **Output:** checkpoints under `runs/...`

---

## Repository layout

```
.
|-- grpo_vlm.py                      # GRPO training script (current)
|-- README.md                        # (this file)
|
|-- data/
|   |-- 20_70_filtered.7z            # 9x9 data package (compressed)
|   `-- cleaned_katago_output.jsonl  # cleaned, high-quality JSONL for training
|
|-- script/
|   |-- data_acquiring/
|   |   |-- analysis.cfg
|   |   |-- build_filtered_training_zh_final.py
|   |   |-- compute_metrics_from_topk.py
|   |   |-- data_mining.py
|   |   |-- filter_9x9.py
|   |   |-- label_and_select.py
|   |   |-- pick_random_sgf_based_on_total_move.py
|   |   `-- total_move_histogram.py
|   |
|   `-- model_training/
|       `-- model_training_first_trial.py  # first GRPO attempt (baseline)
|
`-- src/
    |-- katago.exe                 # KataGo engine (Windows)
    `-- KataGo18b9x9.gz            # KataGo 18B 9x9 weights
```

---

## Environment

We use Python 3.10+.

**Core Python deps (minimum):**

* `pandas`, `numpy`, `tqdm`, `ujson` (or `jsonlines`)
* `matplotlib` (for histograms)
* `transformers`, `accelerate`, `peft`, `datasets`
* `trl` (for GRPO)
* `torch` (CUDA recommended)

Example (conda + pip):

```bash
conda create -n katallm python=3.10 -y
conda activate katallm

pip install -U torch --index-url https://download.pytorch.org/whl/cu121   # choose wheel matching your CUDA
pip install -U transformers accelerate peft datasets trl
pip install -U pandas numpy matplotlib tqdm ujson jsonlines
```

> **Windows note:** `src/katago.exe` is included; no build needed. Make sure your GPU drivers and CUDA runtime match your PyTorch wheel.

---

## 1) Data acquiring & cleaning

### 1.1 Extract 9×9 games only

From a folder of mixed-board SGFs, keep only 9×9.

```bash
python script/data_acquiring/filter_9x9.py \
  --input_sgf_dir "D:\raw_sgfs_mixed" \
  --output_sgf_dir "D:\sgfs_9x9"
```

**What it does:** detects board size from SGF headers and copies 9×9 games to the output folder.

---

### 1.2 Inspect game lengths (choose a move range)

We like mid-game positions (not too short, not endgame only). Plot a histogram:

```bash
python script/data_acquiring/total_move_histogram.py \
  --sgf_dir "D:\sgfs_9x9" \
  --out_png "D:\sgfs_9x9_move_hist.png"
```

Use the figure to choose a reasonable **move count window** (e.g., **20–70** moves) to avoid trivial openings and late garbage time.

---

### 1.3 Subsample SGFs by total moves (optional but recommended)

Keep a balanced set in the chosen window:

```bash
python script/data_acquiring/pick_random_sgf_based_on_total_move.py \
  --sgf_dir "D:\sgfs_9x9" \
  --min_moves 20 --max_moves 70 \
  --per_bucket 300 \
  --seed 42 \
  --out_list "D:\sgf_list_20_70.txt"
```

This creates a list of SGF paths (balanced across length buckets).
We sometimes zip these SGFs as `data/20_70_filtered.7z` for portability.

---

## 2) KataGo labeling (policy + value, top-k)

We query KataGo to obtain **top-k candidate moves** with **winrate/visor score** per position. Settings are in `script/data_acquiring/analysis.cfg`. We use:

* **Engine:** `src/katago.exe`
* **Weights:** `src/KataGo18b9x9.gz`

### 2.1 Run labeling + basic selection

```bash
python script/data_acquiring/label_and_select.py \
  --katago_exe ".\src\katago.exe" \
  --katago_model ".\src\KataGo18b9x9.gz" \
  --analysis_cfg ".\script\data_acquiring\analysis.cfg" \
  --sgf_list "D:\sgf_list_20_70.txt" \
  --top_k 10 \
  --out_jsonl "D:\katago_labeled_raw.jsonl" \
  --max_positions_per_game 40 \
  --min_seconds_per_move 0.5
```

**What it does:**

* Steps through positions in each SGF.
* Asks KataGo for the **top-k** moves and their stats (policy, winrate, score lead).
* Saves position metadata plus the candidates into a JSONL.

---

## 3) Quality metrics & filtering

We want to **downweight or drop “one-sided” positions** (when all reasonable moves are nearly the same, the model might learn “anything is fine”). We compute a few metrics:

* **Policy entropy** over top-k
* **Winrate spread** across top-k
* **Score-lead spread** across top-k

### 3.1 Compute metrics

```bash
python script/data_acquiring/compute_metrics_from_topk.py \
  --in_jsonl "D:\katago_labeled_raw.jsonl" \
  --out_jsonl "D:\katago_labeled_with_metrics.jsonl"
```

### 3.2 Build the final cleaned training set

```bash
python script/data_acquiring/build_filtered_training_zh_final.py \
  --in_jsonl "D:\katago_labeled_with_metrics.jsonl" \
  --out_jsonl ".\data\cleaned_katago_output.jsonl" \
  --entropy_min 0.6 \
  --winrate_spread_min 0.05 \
  --keep_fraction_easy 0.10
```

**Recommended defaults (tune as needed):**

* Keep most positions where **policy entropy** is reasonably high (diverse plausible moves).
* Keep a **small** fraction (~10%) of “easy/one-sided” positions to teach *“convert/maintain lead / stabilize / endgame basics.”*

Result: **`data/cleaned_katago_output.jsonl`** — our final, high-quality training file.
(We also keep a compressed pack `data/20_70_filtered.7z` for archival.)

---

## 4) Data format (JSONL)

We currently support two training formats:

### 4.1 **SFT-style** (single best move + explanation target)

Each line (example):

```json
{
  "board_size": 9,
  "komi": 7.5,
  "moves": "B D4; W E5; B C3; W C5; ...",
  "player_to_move": "B",
  "topk": [
    {"move": "D5", "policy": 0.32, "winrate": 0.72, "scoreLead": 3.1},
    {"move": "E4", "policy": 0.18, "winrate": 0.68, "scoreLead": 2.7}
  ],
  "target": {
    "move": "D5",
    "rationale": "Expands influence on the top; contests E5; keeps moyo compact."
  },
  "lang": "zh"
}
```

### 4.2 **GRPO-style** (multiple candidates + reward)

Each line groups candidates for the *same* prompt:

```json
{
  "prompt": "You are a Go coach. Board=9x9, komi=7.5, moves=B D4; W E5; B C3; W C5; ... Next to play: B. Suggest ONE move and explain briefly.",
  "candidates": [
    {"text": "D5 — clamps the top; aims at E5. ...", "move": "D5", "reward": 0.72},
    {"text": "E4 — pressure on E5, keeps shape. ...", "move": "E4", "reward": 0.68}
  ],
  "meta": {"board_size": 9, "lang": "zh"}
}
```

> The **reward** can be derived from KataGo winrate or score lead (normalized). Our GRPO code reads this directly.

---

## 5) Model training

We include two GRPO trainers:

* `script/model_training/model_training_first_trial.py` — **baseline** GRPO, minimal features.
* `grpo_vlm.py` — **current** GRPO trainer (cleaner arguments, better logging/checkpointing).

We default to **Qwen2.5-7B-Instruct** locally (or another instruct model you have).

### 5.1 First trial (baseline)

```bash
python script/model_training/model_training_first_trial.py \
  --train_jsonl ".\data\cleaned_katago_output.jsonl" \
  --base_model "Qwen2.5-7B-Instruct" \
  --output_dir ".\runs\first_trial" \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 3000 \
  --save_steps 500 \
  --logging_steps 50
```

### 5.2 Improved GRPO trainer

```bash
python grpo_vlm.py \
  --train_jsonl ".\data\cleaned_katago_output.jsonl" \
  --base_model "Qwen2.5-7B-Instruct" \
  --trust_remote_code true \
  --use_lora true \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --max_seq_len 1024 \
  --learning_rate 3e-6 \
  --beta1 0.9 --beta2 0.98 --weight_decay 0.01 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --group_size 4 \
  --reward_key "reward" \
  --num_train_epochs 1 \
  --save_steps 500 \
  --logging_steps 50 \
  --warmup_ratio 0.03 \
  --output_dir ".\runs\grpo_qwen2p5_7b_9x9"
```

**Notes:**

* **Group size** = number of candidate completions per prompt (our JSONL has multiple top-k lines per position).
* If you only have SFT-style data, set `--group_size 1` (GRPO reduces to per-sample RL; consider converting to the GRPO format above).
* Use **LoRA** to fit into single-GPU VRAM (24–48 GB recommended). For multi-GPU, enable `accelerate`.

---

## 6) Evaluation (quick sanity checks)

After training:

* **Move@k accuracy:** what fraction of times the model’s top suggestion matches KataGo top-k.
* **Regret:** difference between KataGo’s best reward and reward of model-suggested move.
* **Language check:** explanation contains appropriate shape/tesuji terms.

A small helper is embedded in `script/data_acquiring/compute_metrics_from_topk.py` (set an `--eval_jsonl` and a `--model` flag if you’ve implemented generation hooks there), otherwise evaluate in a notebook by sampling positions and asking the model to output a single move; match against top-k and compute regret.

---

## 7) Reproduce end-to-end (example)

**Windows PowerShell (pseudo):**

```powershell
# 0) Activate env
conda activate katallm

# 1) Filter to 9x9
python script/data_acquiring/filter_9x9.py --input_sgf_dir "D:\raw_sgfs" --output_sgf_dir "D:\sgfs_9x9"

# 2) Inspect lengths
python script/data_acquiring/total_move_histogram.py --sgf_dir "D:\sgfs_9x9" --out_png "D:\len_hist.png"

# 3) Subsample by 20–70
python script/data_acquiring/pick_random_sgf_based_on_total_move.py --sgf_dir "D:\sgfs_9x9" --min_moves 20 --max_moves 70 --per_bucket 300 --seed 42 --out_list "D:\sgf_list_20_70.txt"

# 4) Label with KataGo
python script/data_acquiring/label_and_select.py --katago_exe ".\src\katago.exe" --katago_model ".\src\KataGo18b9x9.gz" --analysis_cfg ".\script\data_acquiring\analysis.cfg" --sgf_list "D:\sgf_list_20_70.txt" --top_k 10 --out_jsonl "D:\katago_labeled_raw.jsonl"

# 5) Compute metrics
python script/data_acquiring/compute_metrics_from_topk.py --in_jsonl "D:\katago_labeled_raw.jsonl" --out_jsonl "D:\katago_labeled_with_metrics.jsonl"

# 6) Build final JSONL
python script/data_acquiring/build_filtered_training_zh_final.py --in_jsonl "D:\katago_labeled_with_metrics.jsonl" --out_jsonl ".\data\cleaned_katago_output.jsonl" --entropy_min 0.6 --winrate_spread_min 0.05 --keep_fraction_easy 0.10

# 7) Train (GRPO)
python .\grpo_vlm.py --train_jsonl ".\data\cleaned_katago_output.jsonl" --base_model "Qwen2.5-7B-Instruct" --use_lora true --output_dir ".\runs\grpo_qwen2p5_7b_9x9"
```

---

## 8) Design choices

* **Move-range filter (e.g., 20–70)**: avoids trivial openings and ultra-late positions; focuses the model on “real decisions.”
* **Downweight one-sided states**: prevents the model from learning “anything is fine.” We still keep a small slice for *conversion/closing* patterns.
* **GRPO vs SFT**: GRPO uses **relative rewards** among candidates (top-k) to shape the policy; this captures nuances that a single best-move SFT target would hide.

---

## 9) Troubleshooting

* **KataGo fails to start**: verify `analysis.cfg` path, `katago.exe` location, and that `KataGo18b9x9.gz` exists. Try `.\src\katago.exe version`.
* **CUDA OOM during training**: lower `--max_seq_len`, increase `--gradient_accumulation_steps`, enable `--use_lora`, or use 4-bit quantization (QLoRA).
* **Weird JSON errors**: ensure each JSONL line is a single valid JSON object; prefer `ujson`/`jsonlines` for speed/robustness.

---

## 10) Acknowledgments

* **KataGo** for world-class Go analysis.
* **TRL** (Hugging Face) for GRPO implementations.
* **Qwen** for strong instruction base models.

---

## 11) License

This repository is for research/educational use. Please respect third-party licenses (KataGo, base LLM weights) when redistributing data or checkpoints.

---

**Contact:** open an issue or PR if you find a bug or want to add a feature (e.g., 19×19 support, English prompts, richer rationales, better reward shaping).
