# Usa l'immagine base di RunPod con CUDA aggiornata
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file necessari
COPY requirements.txt .
COPY handler.py .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Pre-scarica il modello per velocizzare il primo avvio (opzionale ma consigliato)
RUN python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; \
    AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf'); \
    print('Processor downloaded')"

# Comando per avviare l'handler
CMD ["python", "-u", "handler.py"]
