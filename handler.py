import os, shutil, torch, requests
from io import BytesIO
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import runpod

# --- Cache Hugging Face su /app/hf (più spazio) e pulizia vecchie cache ---
os.environ["HF_HOME"] = "/app/hf"
os.environ["TRANSFORMERS_CACHE"] = "/app/hf"
for p in ["/root/.cache/huggingface", "/root/.cache/torch", "/opt/conda/pkgs"]:
    shutil.rmtree(p, ignore_errors=True)
os.makedirs("/app/hf", exist_ok=True)

MODEL_ID = "Qwen/Qwen2-VL-1.5B-Instruct"  # se ancora spazio insufficiente, useremo 1.5B

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

def _load_image(inp):
    if "image_base64" in inp:
        import base64
        return Image.open(BytesIO(base64.b64decode(inp["image_base64"]))).convert("RGB")
    if "image_url" in inp:
        r = requests.get(inp["image_url"], timeout=60); r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    raise ValueError("Fornisci 'image_url' oppure 'image_base64'.")

def _caption(img, prompt="Descrivi dettagliatamente l'immagine in italiano.", **gen):
    gen.setdefault("max_new_tokens", 200)
    gen.setdefault("temperature", 0.7)
    gen.setdefault("top_p", 0.9)
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

def handler(job):
    i = job.get("input", {})
    img = _load_image(i)
    cap = _caption(
        img,
        prompt=i.get("prompt", "Descrivi dettagliatamente l'immagine in italiano."),
        max_new_tokens=int(i.get("max_new_tokens", 200)),
        temperature=float(i.get("temperature", 0.7)),
        top_p=float(i.get("top_p", 0.9)),
        repetition_penalty=float(i.get("repetition_penalty", 1.1))
    )
    return {"caption": cap}

runpod.serverless.start({"handler": handler})
