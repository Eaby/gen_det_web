from pathlib import Path
from typing import Dict, Optional, Tuple
import importlib
import importlib.util
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

_MODEL_CACHE: Dict[str, nn.Module] = {}

# ---------- helpers ----------
def _to_device(device: str) -> torch.device:
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def _import_custom_model():
    """
    Try to import your Lightning module class CustomResNet50 from common locations.
    Falls back to loading by file path if needed.
    """
    candidates = [
        ("resnet_train", "CustomResNet50"),
        ("models.resnet_train", "CustomResNet50"),
        ("generative_detector_webui.models.resnet_train", "CustomResNet50"),
    ]
    # normal imports
    for mod_name, cls_name in candidates:
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, cls_name):
                return getattr(m, cls_name)
        except Exception:
            pass
    # by file path
    search_paths = [
        Path.cwd() / "resnet_train.py",
        Path(__file__).resolve().parent / "resnet_train.py",
        Path(__file__).resolve().parent.parent / "resnet_train.py",
        Path(__file__).resolve().parent / "models" / "resnet_train.py",
        Path(__file__).resolve().parent.parent / "models" / "resnet_train.py",
    ]
    for p in search_paths:
        if p.exists():
            spec = importlib.util.spec_from_file_location("resnet_train_dyn", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "CustomResNet50"):
                return getattr(mod, "CustomResNet50")
    return None

def _preprocess(pil: Image.Image) -> torch.Tensor:
    tform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tform(pil.convert("RGB")).unsqueeze(0)

def _prob_natural_from_logits(logits: torch.Tensor) -> float:
    # Support 1-logit or 2-logit heads
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.ndim == 2 and logits.size(1) == 1:
        # single logit (your CustomResNet50)
        return torch.sigmoid(logits[0, 0]).item()
    if logits.ndim == 1:
        return torch.sigmoid(logits[0]).item()
    if logits.ndim == 2 and logits.size(1) == 2:
        # assume index 1 = Natural
        return torch.softmax(logits, dim=1)[0, 1].item()
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

# ---------- public API ----------
def load_image_model(ckpt_path: str, device: str = "auto") -> nn.Module:
    """
    Preferred path: use CustomResNet50.load_from_checkpoint for your .ckpt.
    If Lightning version mismatch blocks that, falls back to building the class
    and loading state_dict with strict=False.
    """
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if ckpt_path in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_path]

    ModelClass = _import_custom_model()
    if ModelClass is None:
        raise ImportError(
            "Could not import CustomResNet50. Ensure resnet_train.py is next to app.py "
            "or inside generative_detector_webui/models/ with __init__.py files."
        )

    dev = _to_device(device)
    # First try: Lightning-native load (best)
    try:
        model = ModelClass.load_from_checkpoint(checkpoint_path=str(ckpt), map_location=dev)
        model = model.to(dev).eval()
        _MODEL_CACHE[ckpt_path] = model
        return model
    except Exception as e:
        print(f"[image_model] load_from_checkpoint failed ({e}). Trying state_dict fallback...")

    # Fallback: construct and load 'state_dict' flexibly
    state = torch.load(str(ckpt), map_location=dev)
    state_dict = None
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state_dict = state["model"]
        else:
            state_dict = state
    else:
        state_dict = state

    model = ModelClass()  # assumes your __init__ has no required args
    # Strip common prefixes (Lightning/DDP) and load non-strict
    cleaned = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[image_model] Fallback state_dict load | missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(dev).eval()
    _MODEL_CACHE[ckpt_path] = model
    return model

def infer_image(model: nn.Module, pil_image: Image.Image, threshold: float = 0.5) -> Tuple[str, float]:
    """
    Controller-expected signature:
        label, p_natural = infer_image(model, pil_image, threshold)
    """
    if pil_image is None:
        return "Unknown", 0.0
    # choose the model's device
    try:
        dev = next(model.parameters()).device  # type: ignore[attr-defined]
    except Exception:
        dev = _to_device("auto")

    x = _preprocess(pil_image).to(dev)
    with torch.no_grad():
        logits = model(x)
        p_nat = _prob_natural_from_logits(logits)
    label = "Natural" if p_nat >= float(threshold) else "AI"
    return label, float(p_nat)

def infer_image_from_ckpt(pil_image: Image.Image, ckpt_path: str, threshold: float = 0.5, device: str = "auto") -> Tuple[str, float]:
    """
    Optional convenience if you ever want to pass a path instead of a model.
    Returns the same (label, p_natural).
    """
    model = load_image_model(ckpt_path, device=device)
    return infer_image(model, pil_image, threshold)
