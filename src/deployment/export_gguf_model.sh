#!/bin/bash
# 遇到任何错误立即停止执行 (Fail-fast)，这是工业级脚本的标配
set -e

echo "============================================================"
echo "🚀 Starting Edge Deployment Pipeline for KataGo-LLM"
echo "============================================================"

# --- 环境与路径配置 ---
# 这里统一管理路径，后续修改只需改动这里
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
LORA_CKPT_PATH="./runs/Qwen2.5-7B-GRPO-Go-Pro-v2/checkpoint-1000"
MERGED_MODEL_DIR="./merged-qwen-7b-go"

FP16_GGUF_PATH="./qwen-7b-go-f16.gguf"
QUANTIZED_GGUF_PATH="./qwen-7b-go-Q4_K_M.gguf"
QUANT_METHOD="Q4_K_M"

LLAMA_CPP_DIR="./llama.cpp"
VENV_DIR="./llama_env"

# ==========================================
# 阶段一：合并 LoRA 权重 (强制使用 CPU 防 OOM)
# ==========================================
echo -e "\n[Phase 1/4] Merging LoRA weights with Base Model..."
if [ ! -d "$MERGED_MODEL_DIR" ]; then
    # 使用 Here-Doc 动态生成 Python 脚本，保证这个 Bash 脚本是完全独立可运行的
    cat << 'EOF' > tmp_merge.py
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 强制 CPU 运行

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
    echo "✅ Merge completed!"
else
    echo "⏭️ Merged model directory already exists. Skipping merge."
fi

# ==========================================
# 阶段二：准备 llama.cpp 与 C++ 编译环境
# ==========================================
echo -e "\n[Phase 2/4] Preparing llama.cpp and C++ build environment..."
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

# 使用 CMake 编译 C++ 代码
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    echo "Compiling llama.cpp using CMake..."
    mkdir -p "$LLAMA_CPP_DIR/build"
    cd "$LLAMA_CPP_DIR/build"
    cmake ..
    cmake --build . --config Release -j 8
    cd ../..
    echo "✅ CMake build completed!"
else
    echo "⏭️ llama-quantize executable found. Skipping build."
fi

# ==========================================
# 阶段三：转换格式 (HuggingFace -> GGUF FP16)
# ==========================================
echo -e "\n[Phase 3/4] Converting HuggingFace model to GGUF (FP16)..."
if [ ! -f "$FP16_GGUF_PATH" ]; then
    echo "Setting up isolated Python environment for conversion..."
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    
    # 只安装转换所需的轻量级依赖，绝对不污染宿主机环境
    pip install -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
    
    echo "Running conversion script..."
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL_DIR" --outfile "$FP16_GGUF_PATH"
    
    deactivate
    echo "✅ FP16 GGUF conversion completed!"
else
    echo "⏭️ FP16 GGUF file already exists. Skipping conversion."
fi

# ==========================================
# 阶段四：执行 4-bit 量化
# ==========================================
echo -e "\n[Phase 4/4] Quantizing GGUF model to $QUANT_METHOD (Edge Device Ready)..."
if [ ! -f "$QUANTIZED_GGUF_PATH" ]; then
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$FP16_GGUF_PATH" "$QUANTIZED_GGUF_PATH" "$QUANT_METHOD"
    echo "✅ Quantization completed! Model is ready for Edge Deployment."
else
    echo "⏭️ Quantized model already exists. Skipping."
fi

echo "============================================================"
echo "🎉 Pipeline finished successfully!"
echo "📦 Your Edge-ready model is located at: $QUANTIZED_GGUF_PATH"
echo "============================================================"