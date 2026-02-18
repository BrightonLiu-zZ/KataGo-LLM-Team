# 1. 基础镜像：PyTorch 2.1.2 + CUDA 12.1 + cuDNN 8
# 这个镜像完美适配你的 RTX 6000 Ada
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# 2. 设置工作目录
WORKDIR /workspace

# 3. 设置时区（避免某些安装过程卡在交互界面）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

# 4. 安装系统基础工具
RUN apt-get update && \
    apt-get install -y git curl vim htop wget && \
    rm -rf /var/lib/apt/lists/*

# 5. 升级 pip
RUN pip install --upgrade pip

# 6. 安装核心 AI 依赖
# vllm: 服务器端做大模型推理/训练加速的核心库
# flash-attn: 显存优化神器 (编译可能需要几分钟，请耐心等待)
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    pandas \
    tensorboard \
    scipy \
    vllm 

# 7. (可选) 安装 Flash Attention 2
# RTX 6000 Ada 强烈建议安装这个，训练速度能提升 2-3 倍
# 如果构建报错，可以先把这行注释掉
RUN pip install flash-attn --no-build-isolation

# 8. 默认启动命令
CMD ["/bin/bash"]