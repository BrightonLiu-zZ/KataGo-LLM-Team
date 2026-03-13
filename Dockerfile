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
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    pandas \
    wandb \
    scipy 

# 7.5 单独安装极易报错的 vLLM
RUN pip install --no-cache-dir vllm

# 8. 安装 Flash Attention 2
RUN MAX_JOBS=2 pip install flash-attn --no-build-isolation

# 9. 默认启动命令
CMD ["/bin/bash"]