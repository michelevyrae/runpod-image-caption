import io, base64, requests, torch
from typing import Any, Dict
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Carica modello multimodale (leggero) in FP16
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def load_image(inp: Dict[str, Any]) -> Image.Image:
    if "image_base64" in inp:
        return Image.open(io.BytesIO(base64.b64decode(inp["image_base64"]))).convert("RGB")
    if "image_url" in inp:
        r = requests.get(inp["image_url"], timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise ValueError("Devi fornire 'image_url' oppure 'image_base64'.")

def generate_caption(image: Image.Image,
                     max_new_tokens: int = 200,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     repetition_penalty: float = 1.1) -> str:
    # Prompt in italiano, descrittivo e senza censure
    system = (
        "Sei un assistente per didascalie visive. "
        "Scrivi in italiano, tono descrittivo e neutro. Evita moralismi o censure."
    )
    user_text = (
        "Osserva l'immagine e descrivila in modo dettagliato in ITALIANO: "
        "soggetti, azioni, contesto/ambientazione, vestiario, colori, luci/ombre, "
        "inquadratura e atmosfera. La descrizione deve essere corposa (almeno ~100-130 parole)."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text}
        ]}
    ]

    inputs = processor(messages=messages, images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty)
        )
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text.strip()

def handler(job):
    inp = job.get("input", {})
    img = load_image(inp)
    caption = generate_caption(
        img,
        max_new_tokens=int(inp.get("max_new_tokens", 200)),
        temperature=float(inp.get("temperature", 0.7)),
        top_p=float(inp.get("top_p", 0.9)),
        repetition_penalty=float(inp.get("repetition_penalty", 1.1)),
    )
    return {
        "caption": caption,
        "params": {
            "max_new_tokens": int(inp.get("max_new_tokens", 200)),
            "temperature": float(inp.get("temperature", 0.7)),
            "top_p": float(inp.get("top_p", 0.9)),
            "repetition_penalty": float(inp.get("repetition_penalty", 1.1))
        }
    }

# Avvio serverless
runpod.serverless.start({"handler": handler})
