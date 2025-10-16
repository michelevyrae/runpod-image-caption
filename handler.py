import runpod
import torch
from PIL import Image
import io
import base64
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Variabile globale per il modello
model = None
processor = None
device = None

def load_model():
    """Carica il modello all'avvio del container"""
    global model, processor, device
    
    print("Caricamento modello in corso...")
    
    # Rileva se c'è GPU disponibile
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")
    
    # Usa BLIP-2 che è non censurato e performante
    model_name = "Salesforce/blip2-opt-2.7b"
    
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    print("Modello caricato con successo!")

def decode_image(image_data):
    """Decodifica l'immagine da base64 o URL"""
    try:
        # Se è base64
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Errore nella decodifica dell'immagine: {str(e)}")

def generate_caption(image, prompt=None):
    """Genera la descrizione dell'immagine"""
    global model, processor, device
    
    # Prepara l'input
    if prompt is None:
        prompt = "Describe this image in detail:"
    
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Genera la caption
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=200,
            num_beams=5,
            temperature=0.7
        )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()

def handler(event):
    """Handler principale chiamato da RunPod"""
    try:
        # Estrai l'input
        input_data = event['input']
        
        # Ottieni l'immagine (base64)
        if 'image' not in input_data:
            return {"error": "Nessuna immagine fornita. Usa il campo 'image' con dati base64"}
        
        image_data = input_data['image']
        prompt = input_data.get('prompt', None)
        
        # Decodifica e processa l'immagine
        image = decode_image(image_data)
        
        # Genera la caption
        caption = generate_caption(image, prompt)
        
        return {
            "caption": caption,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    # Carica il modello all'avvio
    load_model()
    
    # Avvia il serverless worker
    print("Avvio RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
