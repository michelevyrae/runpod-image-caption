FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf \
    TRANSFORMERS_CACHE=/app/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    (conda clean -afy || true) && pip cache purge && \
    rm -rf /root/.cache /opt/conda/pkgs

# Cache directory con spazio
RUN mkdir -p /app/hf

COPY handler.py /app/handler.py

CMD ["python", "-c", "import handler, runpod; runpod.serverless.start({'handler': handler.handler})"]
