# V100 (sm_70) kernel build + benchmark image. CUDA 12.6 / cu126 torch still ship sm_70.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -U pip "setuptools<81" wheel \
    && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1 \
    && /opt/venv/bin/pip install --no-cache-dir numpy pytest ninja
ENV PATH=/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=7.0 PYTHONUNBUFFERED=1
