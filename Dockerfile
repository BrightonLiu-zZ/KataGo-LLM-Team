# 1. Base image: the CUDA toolkit must be >=12.9 because flashinfer refuses to
#    JIT-compile sm_120 (Blackwell) kernels with an older nvcc, and flashinfer
#    is the only attention backend that works here: vLLM's bundled
#    flash-attention ships PTX-only sm_120 kernels, and the host's 12.8 driver
#    (570.x, no sudo to upgrade) cannot JIT 12.9+ PTX. Toolkit 12.9 in the
#    container is fine on the 12.8 driver (CUDA minor-version compatibility);
#    the conda torch 2.8 in this base is replaced by pip in step 7a.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# 2. 设置工作目录
WORKDIR /workspace

# 3. 设置时区
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

# 4. 安装系统基础工具 (加入 tmux 方便断线重连和共享监控)
RUN apt-get update && \
    apt-get install -y git curl vim htop wget tmux ninja-build && \
    rm -rf /var/lib/apt/lists/*

# 5. 升级 pip
RUN pip install --upgrade pip

# 6. Target GPU architecture: Blackwell workstation (RTX PRO 6000) is sm_120.
ENV TORCH_CUDA_ARCH_LIST="12.0"

# 7a. vLLM + torch, pinned to CUDA 12.9 builds.
#     The host driver is 570.x (CUDA 12.8). PyPI's vllm 0.25.1 pulls
#     torch 2.11+cu130, which needs a >=580 driver and fails to initialize.
#     No cu128 wheel exists in TRL 1.9.2's supported range (0.17.0–0.25.1),
#     but vLLM ships a +cu129 wheel and CUDA 12.x minor-version compatibility
#     lets 12.9-built binaries run on the 12.8 driver. Torch must come from
#     the matching cu129 index.
RUN pip install --no-cache-dir \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129

# 7b. Core training stack.
#     TRL is pinned: exp05 (train_grpo_v5.py) requires >=1.8 for adaptive
#     entropy control. Installed WITHOUT the [vllm] extra — its
#     "vllm<=0.25.1" specifier does not match the local version 0.25.1+cu129
#     and would drag the cu130 build back in; the extra's own deps are listed
#     explicitly instead. If vLLM fails on this machine, set USE_VLLM=False
#     in train_grpo_v5.py — training falls back to in-process generation.
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    "trl==1.9.2" \
    fastapi \
    pydantic \
    "aiohttp>=3.13.3" \
    requests \
    uvicorn \
    pandas \
    wandb \
    scipy

# 8. (flash-attn removed: vLLM bundles its own flash-attention kernels; the standalone
#    package opens /dev/nvidia* at import time, which hangs when the host MPS server is
#    running and the container lacks full device access.)

# 9. 默认启动命令
CMD ["/bin/bash"]
