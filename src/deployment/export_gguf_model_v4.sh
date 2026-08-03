#!/bin/bash
# Fail-fast on any error
set -e

echo "============================================================"
echo "Starting Edge Deployment Pipeline — Qwen3-8B v4"
echo "============================================================"

# --- Path configuration ---
BASE_MODEL="Qwen/Qwen3-8B"
# NOTE: this used to point at checkpoint-3000, which produced the GGUF currently
# published on HuggingFace (brightonliuzZ/qwen3-8b-go-v4). That adapter no longer
# exists anywhere — it was removed in the 2026-04-06 cleanup and lost for good in
# the 2026-05-06 incident (see docs/RECOVERY-2026-05-06.md). The surviving local
# v4 adapters are checkpoint-1500 and checkpoint-5000; 5000 is the furthest
# trained, so it is the default here. Re-running this script therefore produces a
# DIFFERENT model from the one published on HuggingFace.
LORA_CKPT_PATH="./runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-5000"
MERGED_MODEL_DIR="./merged-qwen3-8b-go-v4"

FP16_GGUF_PATH="./qwen3-8b-go-v4-f16.gguf"
QUANTIZED_GGUF_PATH="./qwen3-8b-go-v4-Q4_K_M.gguf"
QUANT_METHOD="Q4_K_M"

LLAMA_CPP_DIR="./llama.cpp"
VENV_DIR="./llama_env"

# --- Pre-flight: ensure a fresh llama.cpp with Qwen3 support ---
# The old clone may predate Qwen3 architecture support.
# Run this script inside Docker (root) so the rm succeeds.
if [ -d "$LLAMA_CPP_DIR" ]; then
    echo "Removing old llama.cpp to ensure Qwen3 compatibility..."
    rm -rf "$LLAMA_CPP_DIR"
fi

# ==========================================
# Phase 1: Merge LoRA weights (forced CPU to prevent OOM)
# ==========================================
echo -e "\n[Phase 1/4] Merging LoRA weights with Base Model..."
if [ ! -d "$MERGED_MODEL_DIR" ]; then
    cat << 'EOF' > tmp_merge.py
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = sys.argv[1]
lora_path = sys.argv[2]
save_path = sys.argv[3]

print(f"Loading Base Model: {base_path}")
tokenizer = AutoTokenizer.from_pretrained(base_path)
base_model = AutoModelForCausalLM.from_pretrained(base_path, device_map="cpu")

print(f"Applying LoRA: {lora_path}")
model = PeftModel.from_pretrained(base_model, lora_path, device_map="cpu")
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {save_path}")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
EOF
    python3 tmp_merge.py "$BASE_MODEL" "$LORA_CKPT_PATH" "$MERGED_MODEL_DIR"
    rm tmp_merge.py
    echo "Merge completed!"
else
    echo "Merged model directory already exists. Skipping merge."
fi

# ==========================================
# Phase 2: Prepare llama.cpp build environment
# ==========================================
echo -e "\n[Phase 2/4] Preparing llama.cpp and C++ build environment..."
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    echo "Compiling llama.cpp using CMake..."
    mkdir -p "$LLAMA_CPP_DIR/build"
    cd "$LLAMA_CPP_DIR/build"
    cmake ..
    cmake --build . --config Release -j 8
    cd ../..
    echo "CMake build completed!"
else
    echo "llama-quantize executable found. Skipping build."
fi

# ==========================================
# Phase 3: Convert HuggingFace -> GGUF FP16
# ==========================================
echo -e "\n[Phase 3/4] Converting HuggingFace model to GGUF (FP16)..."
if [ ! -f "$FP16_GGUF_PATH" ]; then
    echo "Setting up isolated Python environment for conversion..."
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    pip install -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"

    echo "Running conversion script..."
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL_DIR" --outfile "$FP16_GGUF_PATH"

    deactivate
    echo "FP16 GGUF conversion completed!"
else
    echo "FP16 GGUF file already exists. Skipping conversion."
fi

# ==========================================
# Phase 4: Quantize to 4-bit
# ==========================================
echo -e "\n[Phase 4/4] Quantizing GGUF model to $QUANT_METHOD (Edge Device Ready)..."
if [ ! -f "$QUANTIZED_GGUF_PATH" ]; then
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$FP16_GGUF_PATH" "$QUANTIZED_GGUF_PATH" "$QUANT_METHOD"
    echo "Quantization completed! Model is ready for Edge Deployment."
else
    echo "Quantized model already exists. Skipping."
fi

echo "============================================================"
echo "Pipeline finished successfully!"
echo "Edge-ready model: $QUANTIZED_GGUF_PATH"
echo "============================================================"
