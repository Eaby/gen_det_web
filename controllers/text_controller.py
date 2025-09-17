from __future__ import annotations
from models.registry import get_model
from models.text_model import infer_text

def run_text_detection(text: str, ckpt: str, threshold: float, device: str):
    model = get_model("text", ckpt, device)
    label, pnat = infer_text(model, text, threshold)
    return f"<h4>Text Inference</h4><p>Prediction: <b>{label}</b> (pNatural={pnat:.2f}).</p>"
