# Ramen QC Vision — 推理以 PyTorch/Ultralytics 为主；可选 ONNX（onnxruntime）
# 构建: docker build -t ramen-qc-vision .
# 运行: 见 docker-compose.yml（需挂载 data、models、weights）
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAMEN_BIND_HOST=0.0.0.0 \
    RAMEN_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU 版 PyTorch（镜像内无 CUDA）；GPU 宿主请改用 NVIDIA 官方 PyTorch 镜像或自行 pip 安装 cu 版 torch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "start_web.py"]
