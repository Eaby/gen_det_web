from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import torch
import torch.nn as nn
from PIL import Image

import timm
from timm.data import resolve_model_data_config, create_transform

# ──────────────────────────────────────────────────────────────────────────────
# Robust loader for mixed .pt checkpoints (state_dict) + EVA/ViT backbones.
# Uses timm's data_config for correct normalization and materializes fc_s/fc_a.
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE: Dict[str, nn.Module] = {}
_TFORM_CACHE: Dict[str, torch.nn.Module] = {}

# A shortlist of likely ViT backbones to try (tweak if you know the exact one)
_BACKBONE_CANDIDATES = [
    "eva02_base_patch14_224",
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "vit_large_patch16_224",
    "vit_small_patch16_224",
    "eva02_small_patch14_224",
]

def _to_device(device: str) -> torch.device:
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

class UniTraxLiteCompat(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        temporal_type: str = "mean",
        temporal_depth: int = 2,
        temporal_heads: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        # IMPORTANT: pretrained=False to avoid internet downloads
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        # infer embed dim
        if hasattr(self.backbone, "num_features"):
            self.embed_dim = int(self.backbone.num_features)
        elif hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(self.backbone.embed_dim)
        else:
            self.embed_dim = 768

        self.backbone_head = None

        temporal_type = (temporal_type or "mean").lower()
        self.temporal_type = temporal_type
        self.temporal_depth = int(temporal_depth)
        self.temporal_heads = int(temporal_heads)

        if temporal_type == "transformer":
            enc = nn.TransformerEncoderLayer(
                d_model=self.embed_dim, nhead=self.temporal_heads, batch_first=True, norm_first=True
            )
            self.temporal = nn.TransformerEncoder(enc, num_layers=self.temporal_depth)
        else:
            self.temporal = None

        self.fc_frame = nn.Linear(self.embed_dim, num_classes)
        self.fc_clip  = nn.Linear(self.embed_dim, num_classes)

        # Will be created if checkpoint contains them (we set to None initially)
        self.fc_s: Optional[nn.Linear] = None
        self.fc_a: Optional[nn.Linear] = None

        self.scale_bias = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.add_bias   = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # cache a transform aligned to the backbone's default_cfg
        cfg = resolve_model_data_config(self.backbone)
        # is_training=False -> eval-time resize/crop + mean/std
        self.eval_transform = create_transform(**cfg, is_training=False)

    def _ensure_proj_for_ckpt(self, sd: Dict[str, torch.Tensor]):
        """If checkpoint contains fc_s / fc_a create matching Linear layers before load."""
        for name in ("fc_s", "fc_a"):
            w_key, b_key = f"{name}.weight", f"{name}.bias"
            if w_key in sd and b_key in sd:
                in_features = sd[w_key].shape[1]
                setattr(self, name, nn.Linear(in_features, self.embed_dim))

    def forward(self, x: torch.Tensor, cues=None, T_list=None, is_video: bool = True):
        # Expect [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        D = feats.size(-1)
        feats = feats.view(B, T, D)

        # FiLM-ish scaling/bias
        scale = None
        add = None

        if cues is not None and isinstance(cues, dict):
            s, a = cues.get("s"), cues.get("a")
        else:
            s = a = None

        # If checkpoint had fc_s/fc_a but no cues provided, feed zeros to use learned biases
        if s is None and self.fc_s is not None:
            s = torch.zeros(B * T, self.fc_s.in_features, device=feats.device, dtype=feats.dtype)
        if a is None and self.fc_a is not None:
            a = torch.zeros(B * T, self.fc_a.in_features, device=feats.device, dtype=feats.dtype)

        if s is not None:
            if s.dim() == 1: s = s.unsqueeze(0)
            if s.dim() == 2 and s.size(0) != B*T: s = s.expand(B*T, -1)
            s_raw = self.fc_s(s).view(B, T, D) if self.fc_s is not None else None
            scale = torch.sigmoid(s_raw) if s_raw is not None else None

        if a is not None:
            if a.dim() == 1: a = a.unsqueeze(0)
            if a.dim() == 2 and a.size(0) != B*T: a = a.expand(B*T, -1)
            a_raw = self.fc_a(a).view(B, T, D) if self.fc_a is not None else None
            add = a_raw if a_raw is not None else None

        if scale is None: scale = torch.sigmoid(self.scale_bias.expand(B, T, D))
        if add   is None: add   = self.add_bias.expand(B, T, D)

        feats = feats * scale + add

        if self.temporal is None:
            clip_feat = feats.mean(dim=1)
        else:
            clip_feat = self.temporal(feats)[:, -1, :]

        logits_frame = self.fc_frame(feats)
        logits_clip  = self.fc_clip(clip_feat)
        return {"logits_frame": logits_frame, "logits_clip": logits_clip}

# ──────────────────────────────────────────────────────────────────────────────
# State-dict driven architecture guessing
# ──────────────────────────────────────────────────────────────────────────────
def _guess_temporal(sd: Dict[str, torch.Tensor]) -> Tuple[str, int]:
    depth = -1
    for k in sd.keys():
        if k.startswith("temporal.layers."):
            m = re.match(r"temporal\.layers\.(\d+)\.", k)
            if m:
                depth = max(depth, int(m.group(1)))
    if depth >= 0:
        return "transformer", depth + 1
    return "mean", 0

def _count_shape_matches(model: nn.Module, sd: Dict[str, torch.Tensor]) -> int:
    msd = model.state_dict()
    cnt = 0
    for k, v in sd.items():
        if k in msd and tuple(msd[k].shape) == tuple(v.shape):
            cnt += 1
    return cnt

def _clean_state_dict(raw) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        sd = raw["model"]
    else:
        sd = raw if isinstance(raw, dict) else {}
    cleaned = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith("module."): k = k[len("module."):]
        if k.startswith("model."):  k = k[len("model."):]
        cleaned[k] = v
    return cleaned

def _build_model_for_state_dict(sd: Dict[str, torch.Tensor]) -> UniTraxLiteCompat:
    temporal_type, temporal_depth = _guess_temporal(sd)

    best = None
    best_score = -1
    best_name = None
    for name in _BACKBONE_CANDIDATES:
        try:
            m = UniTraxLiteCompat(
                backbone_name=name,
                temporal_type=temporal_type,
                temporal_depth=max(1, temporal_depth) if temporal_type == "transformer" else 0,
                temporal_heads=4,
                num_classes=2,
            )
            # materialize fc_s / fc_a before scoring/load if present
            m._ensure_proj_for_ckpt(sd)
            score = _count_shape_matches(m, sd)
            if score > best_score:
                best, best_score, best_name = m, score, name
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Could not construct a compatible backbone for this checkpoint.")
    print(f"[video_model] auto-selected backbone='{best_name}', temporal='{temporal_type}', depth={max(1, temporal_depth) if temporal_type=='transformer' else 0} (matches={best_score})")
    return best

def _make_transform_for(model: UniTraxLiteCompat) -> torch.nn.Module:
    # Use cached tfm per backbone to avoid re-building
    key = model.backbone.__class__.__name__
    if key in _TFORM_CACHE:
        return _TFORM_CACHE[key]
    cfg = resolve_model_data_config(model.backbone)
    tfm = create_transform(**cfg, is_training=False)
    _TFORM_CACHE[key] = tfm
    return tfm

# ──────────────────────────────────────────────────────────────────────────────
# Public loader used by the Video tab
# ──────────────────────────────────────────────────────────────────────────────
def load_mixed_pt_model(ckpt_path: str, device: str = "auto") -> UniTraxLiteCompat:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Mixed model .pt not found: {ckpt_path}")
    key = str(ckpt.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    dev = _to_device(device)

    # 1) Try TorchScript first (will fail for checkpoints — that's OK)
    try:
        m = torch.jit.load(str(ckpt), map_location=dev)
        m = m.to(dev).eval()
        _MODEL_CACHE[key] = m
        print("[video_model] Loaded TorchScript mixed model.")
        return m  # in this path, we'll fall back to generic transforms below
    except Exception as e:
        print(f"[video_model] TorchScript load failed ({e}); trying checkpoint dict...")

    # 2) Load checkpoint dict and rebuild model
    obj = torch.load(str(ckpt), map_location=dev)
    sd = _clean_state_dict(obj)
    if not sd:
        raise RuntimeError("Loaded object is not a TorchScript module nor a checkpoint dict with 'model' weights.")

    model = _build_model_for_state_dict(sd).to(dev)
    try:
        model.load_state_dict(sd, strict=True)
        print("[video_model] Loaded weights with strict=True.")
    except Exception as e:
        print(f"[video_model] strict load failed ({e}); retrying with strict=False...")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[video_model] Fallback load: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    _MODEL_CACHE[key] = model
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Inference on a clip (fixed-frame sample)
# ──────────────────────────────────────────────────────────────────────────────
def _prob_natural_from_logits(out) -> float:
    if isinstance(out, dict):
        out = out.get("logits_clip", None)
        if out is None:
            raise ValueError("Output dict missing 'logits_clip'.")
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not torch.is_tensor(out):
        raise ValueError(f"Unexpected output type: {type(out)}")
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.size(-1) == 1:
        return torch.sigmoid(out)[0, 0].item()
    if out.size(-1) == 2:
        return torch.softmax(out, dim=-1)[0, 1].item()
    raise ValueError(f"Unexpected logits shape: {tuple(out.shape)}")

def run_mixed_pt_model_on_clip(frame_paths: List[Path], ckpt_path: str,
                               device: str, threshold: float,
                               max_frames: int = 32) -> Tuple[Dict[str, int], float]:
    if not frame_paths:
        return {"AI": 0, "Natural": 0}, 0.0

    model = load_mixed_pt_model(ckpt_path, device=device)
    # Evenly sample up to max_frames
    import numpy as np
    idx = np.linspace(0, len(frame_paths) - 1, num=min(len(frame_paths), max_frames), dtype=int)
    sel = [frame_paths[i] for i in idx]

    # Use backbone-aligned transform if available
    try:
        tfm = _make_transform_for(model)
    except Exception:
        # fallback to vanilla imagenet transform
        from torchvision import transforms as _T
        tfm = _T.Compose([
            _T.Resize((224, 224)),
            _T.ToTensor(),
            _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    imgs = []
    for p in sel:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(tfm(img))
        except Exception:
            continue
    if not imgs:
        return {"AI": 0, "Natural": 0}, 0.0

    x = torch.stack(imgs, dim=0)  # [T,C,H,W]
    dev = next(model.parameters()).device if hasattr(model, "parameters") else _to_device(device)
    x = x.to(dev)

    with torch.no_grad():
        # prefer [1,T,C,H,W]
        try:
            out = model(x.unsqueeze(0))
        except Exception:
            out = model(x)
        p_nat = _prob_natural_from_logits(out)

    counts = {"AI": int(p_nat < threshold), "Natural": int(p_nat >= threshold)}
    return counts, float(p_nat)
