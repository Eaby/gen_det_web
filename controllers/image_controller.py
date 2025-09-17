from __future__ import annotations
from PIL import Image
from models.registry import get_model
from models.image_model import infer_image

def run_single_image(image: Image.Image, ckpt: str, threshold: float, device: str):
    model = get_model("image", ckpt, device)
    label, pnat = infer_image(model, image, threshold)
    caption = f"{label} (pNatural={pnat:.2f})"
    return [(image, caption)], f"<h4>Image Inference</h4><p>Prediction: <b>{label}</b> (pNatural={pnat:.2f}).</p>"
