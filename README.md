# KataGo-LLM-Team: GRPO-Aligned Go AI — Edge-Deployed at 42.5 tok/sec on 8GB VRAM

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)
[![TRL](https://img.shields.io/badge/TRL-0.29-blueviolet.svg)](https://github.com/huggingface/trl)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Edge%20Inference-green.svg)](https://github.com/ggerganov/llama.cpp)
[![W&B Run](https://img.shields.io/badge/W%26B-training%20run-orange.svg)](https://wandb.ai/brighton_zz-uc-san-diego/huggingface/runs/eb6zhzfo)

---

## 📖 About The Project

<video src="lmstudio_lizzie_demonstration.mp4" controls="controls" style="max-width: 100%; height: auto;"></video>

This project bridges two major challenges in modern AI: **algorithmic alignment for spatial reasoning** and **model compression for edge device deployment**.

Large language models are not naturally equipped for spatial board games like Go — they lack built-in 2D coordinate reasoning, board legality awareness, and any understanding of territory or life-and-death. We bridge that gap by using [KataGo](https://github.com/lightvector/KataGo) as an expert teacher (for training data generation only) and training [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with Group Relative Policy Optimization (GRPO). The key insight is an *outcome-regret reward formulation* that forces the model to learn not just "play somewhere reasonable" but "play the best available move."

**Beyond algorithmic alignment, this project demonstrates a complete full-stack ML pipeline.** The fine-tuned model is compressed into a 4-bit GGUF format and served through a native C++ GTP proxy that relays moves from any standard Go GUI to a locally-hosted LM Studio instance — achieving real-time inference on consumer-grade hardware with zero cloud dependency.

---

## 🌟 Key Features & Highlights

* 🧠 **RLHF/GRPO Alignment for Spatial Reasoning:** Designed a multi-layered reward function using KataGo's top-K winrates to penalise regret and prevent reward hacking. The model is trained on ~5,000 board positions derived from 312 expert-curated examples expanded via D4 symmetry augmentation (8 rotations/reflections × colour-flip = 16× expansion). Successfully teaches a text-based LLM the spatial and tactical rules of 9×9 Go.

* ⚡ **Edge-Device Deployment & Compression:** Automated pipeline merges LoRA weights and compresses the Qwen2.5-7B model via 4-bit `Q4_K_M` GGUF quantization, reducing the 14GB+ model to ~4.5GB. Achieves blazing-fast real-time inference on a consumer edge device (**RTX 4060 Laptop GPU, 8GB VRAM**) with **~42.5 tokens/sec** throughput and **0.52s Time-to-First-Token (TTFT)**.

* ⚙️ **C++ Native REST Microservice:** A lightweight, standalone C++ GTP proxy (`gtp_proxy.exe`) built with `cpp-httplib` intercepts GUI commands and sends stateless REST API requests (`POST /v1/chat/completions`) to a local LM Studio server, eliminating all Python overhead from the inference path.

---

## Table of Contents

1. [Key Features & Highlights](#-key-features--highlights)
2. [Architecture Overview](#-architecture-overview)
3. [Quick Start — End-to-End Edge Deployment](#-quick-start--end-to-end-edge-deployment)
4. [Automated Deployment Pipeline (For Geeks)](#-automated-deployment-pipeline-for-geeks)
5. [Training & Algorithm Background](#-training--algorithm-background)
6. [Results](#-results)
7. [Citations](#-citations)

---

## 🔭 Architecture Overview

```text
┌─────────────────────────────────────────┐
│  Raw SGF Files  (9x9 Go games)          │
└──────────────┬──────────────────────────┘
               │  KataGo analysis (training only)
               ▼
┌─────────────────────────────────────────┐
│  GRPO Training  (Qwen2.5-7B + LoRA)     │
│  Outcome-Regret Reward · D4 Augment     │
└──────────────┬──────────────────────────┘
               │  Merge LoRA + 4-bit Quantization
               ▼
┌─────────────────────────────────────────┐
│  qwen-7b-go-Q4_K_M.gguf  (~4.5 GB)      │
│  Loaded into LM Studio (localhost:1234) │
└──────────────┬──────────────────────────┘
               │  ~42.5 tok/sec · 0.52s TTFT
               ▼
┌───────────────────────────────────────────────────────────────────┐
│  Lizzie (Go GUI)  ──GTP──►  gtp_proxy.exe  ──REST──►  LM Studio  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start — End-to-End Edge Deployment

### Prerequisites

* **OS:** Windows 11
* **GPU:** Consumer-grade NVIDIA GPU with **8GB+ VRAM** (e.g. RTX 4060 Laptop)
* **Software:** [LM Studio](https://lmstudio.ai/) installed

---

### Step 1 — Get the Go GUI (Lizzie)

Download and install the latest **Lizzie** release from the official page:

> **[https://github.com/featurecat/lizzie/releases](https://github.com/featurecat/lizzie/releases)**

---

### Step 2 — Download the Model

Download the quantized model file `qwen-7b-go-Q4_K_M.gguf` from Hugging Face:

> **[https://huggingface.co/brightonliuzZ/qwen2.5-7b-go-GGUF/tree/main](https://huggingface.co/brightonliuzZ/qwen2.5-7b-go-GGUF/tree/main)**

---

### Step 3 — Start the Local LLM Server

1. Open **LM Studio**.
2. Load `qwen-7b-go-Q4_K_M.gguf` into the model slot.
3. Navigate to the **Local Server** tab and click **Start Server**.
4. Confirm the server is listening on `localhost:1234`.

---

### Step 4 — Connect the Engine to Lizzie

1. Download the latest **`gtp_proxy_windows.zip`** from the [GitHub Releases page](https://github.com/BrightonLiu-zZ/KataGo-LLM-Team/releases).
2. Extract `gtp_proxy.exe` to any folder on your machine.
3. In Lizzie, open **Settings → Engine** and set the engine path to the **absolute path** of `gtp_proxy.exe`. For example:
   ```
   C:\tools\gtp_proxy.exe
   ```
4. **Do not add any additional arguments** — no `gtp`, no `-model`, no configuration flags. The engine is fully self-contained and connects to LM Studio automatically.
5. Save settings and restart the engine. Lizzie is now powered by your local GRPO-aligned Go AI.

---

## 🔧 Automated Deployment Pipeline (For Geeks)

> **Optional / Advanced** — Only needed if you want to rebuild the GGUF model from scratch (e.g., after continued training).

The script [`src/deployment/export_gguf_model.sh`](src/deployment/export_gguf_model.sh) automates the entire post-training compression lifecycle in four sequential, idempotent phases:

| Phase | What Happens |
|-------|-------------|
| **1 — CPU-Bound LoRA Merge** | Merges GRPO LoRA adapters into the base `Qwen2.5-7B-Instruct` model using CPU-only execution (`CUDA_VISIBLE_DEVICES=-1`) to avoid OOM on shared GPU servers. Outputs `./merged-qwen-7b-go`. |
| **2 — llama.cpp Build** | Clones the `llama.cpp` repository (if absent) and compiles the `llama-quantize` binary via CMake in Release mode. |
| **3 — FP16 GGUF Conversion** | Creates an isolated Python venv, installs lightweight conversion deps, and converts the merged HuggingFace weights to a full-precision GGUF file (`qwen-7b-go-f16.gguf`). |
| **4 — 4-bit Quantization** | Runs `llama-quantize` to compress the FP16 GGUF into `Q4_K_M` format, producing the final `qwen-7b-go-Q4_K_M.gguf` (~4.5GB, edge-ready). |

```bash
# Run from repository root (Linux/WSL required)
bash src/deployment/export_gguf_model.sh
```

The script includes fail-fast error handling (`set -e`) and skip-logic so completed phases are not re-executed on rerun.

---

## 📚 Training & Algorithm Background

### Problem

Text-based LLMs have no innate spatial reasoning for board games. Given an ASCII 9×9 Go board, a base model cannot reliably identify legal moves, detect capture threats, or evaluate territory. Standard supervised fine-tuning on expert moves is insufficient — the model learns to imitate syntax without understanding tactics.

### Approach: GRPO with Outcome-Regret Reward

We use **Group Relative Policy Optimization (GRPO)** to train `Qwen2.5-7B-Instruct` with a four-layer reward function. KataGo serves as a pure teacher oracle during data generation — it is not used at inference time.

**Reward layers (composed inside `GRPOTrainer`):**

| Layer | Function | Range |
|-------|----------|-------|
| **Format & Legality** | Checks `<|im_start|>…ground` tags; penalises illegal or occupied moves via ASCII board parsing | `[−1.0, +0.2]` |
| **Thinking Coherence** | Penalises chain-of-thought inconsistency (thinking `D6`, playing `C4`); rewards correct length and top-10 move inclusion | `[−0.5, +0.5]` |
| **Outcome Regret** | Core signal: $\text{reward} = \text{wr}_{\text{chosen}} - 1.5 \times (\text{wr}_{\text{best}} - \text{wr}_{\text{chosen}})$ — heavily penalises suboptimal moves even in winning positions | `[−1.0, +1.0]` |
| **Logging** | Non-scoring; samples rollouts to `grpo_training_rollouts.jsonl` for offline analysis | — |

The **1.5× regret multiplier** is critical: it breaks the degenerate strategy of "play any legal move in a won game" by making the regret cost exceed the base winrate reward for poor moves.

### Data Pipeline (Summary)

312 tactically rich positions are selected from filtered 9×9 SGF games, then expanded to **~5,000 training examples** via:
- **D4 symmetry augmentation** — 8 transforms (4 rotations × 2 reflections)
- **Colour flip** — Black ↔ White (total 16× expansion per position)

KataGo analyses each position at 500 visits to supply top-10 candidate moves with winrates, forming the ground truth for the regret reward.

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Training method | GRPO + LoRA (`r=16`, `lora_alpha=32`) |
| Learning rate | `5e-6` (cosine decay, 10% warmup) |
| Batch size | `8` per device, `2` grad. accum. (effective 16) |
| Generations per step | `8` |
| Max completion length | `1024` tokens |
| Total steps | `1000` |
| Precision | `bf16` |

Training was run on a **48GB VRAM RTX 6000 Ada** using `src/model_training/train_grpo.py`. A memory-optimised laptop variant for Qwen2.5-0.5B is available in `src/model_training/train_grpo_laptop.py`.

---

## 📊 Results

Training was tracked with Weights & Biases:
**[View training run on W&B →](https://wandb.ai/brighton_zz-uc-san-diego/huggingface/runs/eb6zhzfo)**

The final model checkpoint (LoRA adapters + training artefacts) is saved at `MyModel/Qwen2.5-7B-GRPO-Go-Pro-v1/` (trained with TRL 0.29.0, Transformers 4.57.6, PyTorch 2.9.1).

**Edge inference performance (RTX 4060 Laptop, 8GB VRAM, 100% GPU offload):**

| Metric | Value |
|--------|-------|
| Throughput | ~42.5 tokens/sec |
| Time-to-First-Token (TTFT) | ~0.52s |
| Model size on disk | ~4.5GB (Q4_K_M GGUF) |
| Network overhead | 0 (fully local) |

---

## 📎 Citations

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
