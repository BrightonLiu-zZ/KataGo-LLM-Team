# Debug Log: Training `train_grpo_v3.py` — Current Status (2026-03-20)

## ✅ Resolved Issues

| Issue | Root Cause | Fix Applied |
|---|---|---|
| `from transformers import Trainer` hung indefinitely | `flash_attn` C extension opens `/dev/nvidia*` directly, blocks when GPU is in Exclusive Process mode | Removed `flash-attn` from `Dockerfile` (vLLM bundles its own flash attention) |
| `KeyError: getpwuid(): uid not found: 0` | `/etc/passwd` is bind-mounted from host, has no entry for uid 0 (root); PyTorch inductor calls `getpass.getuser()` | Add `-e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache` to `docker run` |
| `cudaErrorDevicesUnavailable` | GPU was in `Exclusive Process` compute mode; MPS server socket at `/tmp/nvidia-mps/` was empty (broken state) | Server admin ran `sudo nvidia-smi -c 0` to reset compute mode to Default |

---

## Current State

- **Dockerfile**: `flash-attn` removed. Image rebuilt as `katago-llm`. ✅
- **GPU compute mode**: Reset to Default by admin. ✅
- **Training script**: `src/model_training/train_grpo_v3.py` — ready to run. ✅
- **Blocker**: Docker run command has not been successfully executed yet (copy-paste formatting errors in terminal).

---

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 97887 MiB VRAM |
| Driver | 570.211.01 |
| CUDA | 12.8 |
| Host OS | Linux (shared server, user `tyliu` has no sudo) |
| Container image | `katago-llm` (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel base) |
| Python | 3.11 (conda in `/opt/conda`) |
| Training script | `src/model_training/train_grpo_v3.py` |
| Model | `Qwen/Qwen3-8B` (cached at `/workspace/.cache/huggingface`) |
| Dataset | `data/training_data_augmented_shuffled_v3.jsonl` (270,000 examples) |

---

## The Working Docker Run Command

**Type this manually or paste carefully — do not copy the backslashes with trailing spaces:**

```bash
docker run --gpus all --ipc=host -e HF_HOME=/workspace/.cache/huggingface -e HOME=/workspace -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache -e WANDB_API_KEY=YOUR_KEY_HERE -v /tmp/docker_passwd:/etc/passwd:ro -v $(pwd):/workspace -w /workspace -it katago-llm bash
```

Replace `YOUR_KEY_HERE` with a fresh WandB API key from wandb.ai → Settings → API keys.

> **Note:** The previous two WandB keys were accidentally exposed in terminal output and must be treated as compromised. Always regenerate after exposure.

---

## Steps to Start Training (once inside the container)

**Step 1: Sanity check GPU access**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected output: `True NVIDIA RTX PRO 6000 Blackwell`

**Step 2: Start training**
```bash
python -u src/model_training/train_grpo_v3.py 2>&1 | tee logs/train_v3_run.log
```

**Step 3: Monitor**
```bash
# In another terminal/pane:
tail -f logs/train_v3_run.log
nvidia-smi
```

---

## Training Configuration (in `train_grpo_v3.py`)

- Model: `Qwen/Qwen3-8B` with LoRA (`r=16`, `lora_alpha=32`)
- Dataset: 270,000 examples
- `use_vllm=True`, `vllm_gpu_memory_utilization=0.4`
- `per_device_train_batch_size=8`, `num_generations=8`
- `max_steps=270000` (stop manually with Ctrl+C when ready)
- Checkpoints saved to `runs/Qwen3-8B-GRPO-Go-Pro-v3/` every 500 steps
- W&B logging enabled (`report_to="wandb"`)

---

## If Training Fails

### `cudaErrorDevicesUnavailable` again
The compute mode may have been reset back to Exclusive Process. Ask the admin:
```bash
sudo nvidia-smi -c 0      # reset to Default
nvidia-smi                # verify Compute M. shows "Default"
```

### `CUDA out of memory`
Reduce `vllm_gpu_memory_utilization` from `0.4` to `0.3` in `train_grpo_v3.py`.

### vLLM port conflict
vLLM opens a server on a port. If another process is using it, set:
```bash
-e VLLM_PORT=8001   # or any free port
```

### W&B not logging
Run `wandb login` inside the container before starting training, or set `WANDB_DISABLED=true` to skip W&B entirely.
