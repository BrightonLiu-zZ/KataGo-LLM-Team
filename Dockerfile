# 1. 升级基础镜像：适配最新 vLLM 需求的 PyTorch 2.4.0
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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

# 6. 设置目标显卡架构 (针对 RTX 6000 Ada 优化编译过程)
ENV TORCH_CUDA_ARCH_LIST="8.9"

# 7. 先安装基础核心 AI 依赖
# Pin trl>=0.15 and vllm>=0.8 together — earlier TRL versions expect the old
# vllm.worker module layout which was removed in vLLM 0.8.
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    "trl>=0.15.0" \
    pandas \
    wandb \
    scipy 

# 7.5 vLLM disabled: use_vllm=False in train_grpo_v3.py avoids the TRL/vLLM
# version-compatibility treadmill. The RTX PRO 6000 Blackwell has 97 GB VRAM
# so in-process generation is fast enough. Re-enable if a stable TRL+vLLM
# pair that supports SM 10.0 (Blackwell) becomes available.
# RUN pip install --no-cache-dir "vllm==..."

# 8. (flash-attn removed: vLLM bundles its own flash-attention kernels; the standalone
#    package opens /dev/nvidia* at import time, which hangs when the host MPS server is
#    running and the container lacks full device access.)

# 9. 默认启动命令
CMD ["/bin/bash"]