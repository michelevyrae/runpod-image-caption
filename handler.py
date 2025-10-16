import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Carica modello e processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()

def generate_caption(image, prompt="Descrivi dettagliatamente l'immagine in italiano.", **gen_kwargs):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    gen_kwargs.setdefault("max_new_tokens", 200)
    gen_kwargs.setdefault("temperature", 0.7)
    gen_kwargs.setdefault("top_p", 0.9)

    output = model.generate(**inputs, **gen_kwargs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption.strip()

def handler(job):
    inp = job.get("input", {})
    image_url = inp.get("image_url")
    if not image_url:
        return {"error": "Missing image_url"}

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    caption = generate_caption(
        image,
        prompt=inp.get("prompt", "Descrivi dettagliatamente l'immagine in italiano."),
        max_new_tokens=int(inp.get("max_new_tokens", 200)),
        temperature=float(inp.get("temperature", 0.7)),
        top_p=float(inp.get("top_p", 0.9)),
        repetition_penalty=float(inp.get("repetition_penalty", 1.1))
    )
    return {"caption": caption}

