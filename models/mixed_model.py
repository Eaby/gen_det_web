from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import yaml, torch, numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Import your ensemble (adjust path/module as needed)
try:
    from mixed_model_loader import UniTraxEnsemble
    MIXED_MODEL_AVAILABLE = True
except Exception as e:
    UniTraxEnsemble = None
    MIXED_MODEL_AVAILABLE = False

def load_mixed_model(ckpt_path: str, config_path: str, device: str = "auto") -> nn.Module:
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not MIXED_MODEL_AVAILABLE:
        raise ImportError("UniTraxEnsemble not available. Place mixed_model_loader.py next to app or adjust import.")

    dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']; dataset_cfg = config['dataset']
    img_cfg = {"img_size": dataset_cfg["img_size"], "patch_size": dataset_cfg["patch_size"], "backbone_name": model_cfg["backbone_name"]}
    vid_cfg = {**img_cfg, "temporal_type": model_cfg["temporal_type"], "temporal_depth": model_cfg["temporal_depth"], "temporal_heads": model_cfg["temporal_heads"]}

    model = UniTraxEnsemble(img_cfg, vid_cfg, num_classes=2)
    checkpoint = torch.load(ckpt_path, map_location=dev)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state, strict=True)
    return model.to(dev).eval()

# Clips of frames -> video prediction
def infer_mixed_video(model: nn.Module, frame_paths: list[str], threshold: float, max_frames: int = 32):
    if not frame_paths:
        raise FileNotFoundError("No frames provided for mixed model")
    idx = np.linspace(0, len(frame_paths)-1, num=min(len(frame_paths), max_frames), dtype=int)
    selected = [frame_paths[i] for i in idx]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = [transform(Image.open(p).convert("RGB")) for p in selected]
    x = torch.stack(frames).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(x, cues=None, T_list=[x.shape[1]], is_video=True)
    logits_clip = output['logits_clip']
    prob_natural = torch.softmax(logits_clip, dim=1)[0, 1].item()
    label = "Natural" if prob_natural >= threshold else "AI"
    return label, prob_natural, selected
