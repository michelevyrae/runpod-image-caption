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
    pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Pre-scarica i pesi per ridurre il cold start
RUN python - << 'PY'
from transformers import AutoProcessor, AutoModelForCausalLM
m = "Qwen/Qwen2-VL-2B-Instruct"
AutoProcessor.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m, torch_dtype="auto")
print("Qwen2-VL-2B cached.")
PY

# Avvio worker serverless
CMD ["python", "-c", "import handler, runpod; runpod.serverless.start({'handler': handler.handler})"]
