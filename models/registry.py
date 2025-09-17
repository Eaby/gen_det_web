from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel
from .image_model import load_image_model
from .mixed_model import load_mixed_model
from .text_model import load_text_model

class ModelKey(BaseModel):
    kind: str  # "image" | "mixed" | "text"
    ckpt: str
    config: Optional[str] = None

_CACHE: Dict[str, object] = {}

def _key(k: ModelKey) -> str:
    return f"{k.kind}::{k.ckpt}::{k.config or ''}"

def get_model(kind: str, ckpt: str, device: str = "auto", config: Optional[str] = None):
    mk = ModelKey(kind=kind, ckpt=ckpt, config=config)
    key = _key(mk)
    if key in _CACHE:
        return _CACHE[key]
    if kind == "image":
        model = load_image_model(ckpt_path=ckpt, device=device)
    elif kind == "mixed":
        if not config:
            raise FileNotFoundError("Mixed model requires a YAML config.")
        model = load_mixed_model(ckpt_path=ckpt, config_path=config, device=device)
    elif kind == "text":
        model = load_text_model(ckpt_path=ckpt, device=device)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    _CACHE[key] = model
    return model
