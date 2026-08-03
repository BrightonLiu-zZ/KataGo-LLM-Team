# 1. Base image: CUDA 12.8 for the RTX PRO 6000 Blackwell (sm_120).
#    The old 2.5.1-cuda12.4 image has no sm_120 kernels and cannot run on the
#    current GPU.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

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

# 7. Core training stack.
#    TRL is pinned: exp05 (train_grpo_v5.py) requires >=1.8 for adaptive
#    entropy control, and the trl[vllm] extra installs the vLLM version that
#    this TRL release was tested against (colocate mode + LoRA). If vLLM
#    fails on this machine, set USE_VLLM=False in train_grpo_v5.py — training
#    falls back to in-process generation, no other change needed.
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    "trl[vllm]==1.9.2" \
    pandas \
    wandb \
    scipy

# 8. (flash-attn removed: vLLM bundles its own flash-attention kernels; the standalone
#    package opens /dev/nvidia* at import time, which hangs when the host MPS server is
#    running and the container lacks full device access.)

# 9. 默认启动命令
CMD ["/bin/bash"]
