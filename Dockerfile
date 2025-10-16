# Usa l'immagine base di RunPod con CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file necessari
COPY requirements.txt .
COPY handler.py .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Comando per avviare l'handler
CMD ["python", "-u", "handler.py"]
