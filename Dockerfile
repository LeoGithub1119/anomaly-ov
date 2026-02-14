FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates \
    build-essential ninja-build cmake pkg-config \
    python3 python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip

COPY . /app

RUN pip install -e ".[train]" \
 && pip install accelerate==0.29.3 pynvml
CMD ["/bin/bash"]
