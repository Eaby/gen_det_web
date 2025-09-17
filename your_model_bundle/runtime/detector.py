# your_model_bundle/runtime/detector.py
from __future__ import annotations
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import json
import math
import yaml
import torch
import torch.nn as nn
import numpy as np
import timm
import cv2

# Local runtime helpers
from .video_io import load_image, load_video_uniform, IMAGE_EXTS, VIDEO_EXTS

# ------------------------------
# A tiny, local UniTraxLite
# (no training repo dependency)
# ------------------------------
class _UniTraxLite(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,        # kept for compatibility
        backbone_name: str = "vit_base_patch16_224",
        temporal_type: str = "mean", # "mean" or "transformer" (we keep "mean" for runtime stability)
        temporal_depth: int = 2,     # kept for compatibility
        temporal_heads: int = 4,     # kept for compatibility
        num_classes: int = 2,
    ):
        super().__init__()
        self.img_size = int(img_size)
        self.temporal_type = str(temporal_type).lower()

        # NOTE: pretrained=False (we load your checkpoint, not the HF weights)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,   # pooled feature head
        )

        if hasattr(self.backbone, "num_features"):
            self.embed_dim = int(self.backbone.num_features)
        elif hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(self.backbone.embed_dim)
        else:
            self.embed_dim = 768

        # No temporal transformer at inference: mean-aggregate frames for a clip
        self.temporal = None

        self.fc_frame = nn.Linear(self.embed_dim, num_classes)
        self.fc_clip  = nn.Linear(self.embed_dim, num_classes)

        # Simple learned biases if cues are absent (we don’t use cues in packaged runtime)
        self.scale_bias = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.add_bias   = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, C, H, W] in ImageNet-normalized scale.
        No external cues; we use learned bias only (as in training fallback).
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)             # [B*T, C, H, W]

        feats = self.backbone(x)              # [B*T, D]
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        D = feats.size(-1)

        feats = feats.view(B, T, D)           # [B, T, D]
        scale = torch.sigmoid(self.scale_bias.expand(B, T, D))
        add   = self.add_bias.expand(B, T, D)
        feats = feats * scale + add

        if self.temporal is None:
            clip_feat = feats.mean(dim=1)     # [B, D]
        else:
            clip_feat = self.temporal(feats)[:, -1, :]

        logits_frame = self.fc_frame(feats)   # [B, T, 2]
        logits_clip  = self.fc_clip(clip_feat)# [B, 2]
        return {"logits_frame": logits_frame, "logits_clip": logits_clip}


# ------------------------------
# Utilities
# ------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _imread_rgb(path: str, target_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img

def _to_tensor_bchw(imgs: List[np.ndarray]) -> torch.Tensor:
    # imgs: list of HxWx3 RGB uint8
    arr = np.stack(imgs, axis=0).astype(np.float32) / 255.0  # [T, H, W, 3]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(0, 3, 1, 2)  # [T, 3, H, W]
    t = torch.from_numpy(arr)        # [T, 3, H, W]
    return t

def _device_select(device: str) -> torch.device:
    device = (device or "auto").strip().lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda" or device == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _Assets:
    ckpt_path: Path
    cfg_path: Optional[Path]
    threshold: Optional[float]
    temperature: Optional[float]


def _load_assets_from_package() -> _Assets:
    """Discover embedded package files under your_model_bundle.assets"""
    pkg = files("your_model_bundle.assets")

    # Required
    ckpt = pkg.joinpath("best.pt")
    if not ckpt.is_file():
        raise FileNotFoundError("your_model_bundle.assets/best.pt not found in the package.")

    # Optional
    cfg  = pkg.joinpath("config.yaml")
    thrj = pkg.joinpath("threshold.json")
    temp = pkg.joinpath("temperature.json")

    threshold = None
    if thrj.is_file():
        try:
            threshold = float(json.loads(thrj.read_text()).get("threshold", None))
        except Exception:
            threshold = None

    temperature = None
    if temp.is_file():
        try:
            temperature = float(json.loads(temp.read_text()).get("temperature", None))
        except Exception:
            temperature = None

    return _Assets(
        ckpt_path=Path(str(ckpt)),
        cfg_path=Path(str(cfg)) if cfg.is_file() else None,
        threshold=threshold,
        temperature=temperature,
    )


def _build_model_from_cfg(cfg_path: Optional[Path]) -> Tuple[_UniTraxLite, int, str]:
    """Read backbone & size from config.yaml if present; otherwise sensible defaults."""
    img_size = 224
    patch_size = 16
    backbone = "vit_base_patch16_224"
    temporal_type = "mean"
    temporal_depth = 2
    temporal_heads = 4

    if cfg_path and cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            ds = cfg.get("dataset", {})
            md = cfg.get("model", {})
            img_size = int(ds.get("img_size", img_size))
            patch_size = int(ds.get("patch_size", patch_size))
            backbone = str(md.get("backbone_name", backbone))
            temporal_type = str(md.get("temporal_type", temporal_type))
            temporal_depth = int(md.get("temporal_depth", temporal_depth))
            temporal_heads = int(md.get("temporal_heads", temporal_heads))
        except Exception:
            pass

    model = _UniTraxLite(
        img_size=img_size,
        patch_size=patch_size,
        backbone_name=backbone,
        temporal_type=temporal_type,
        temporal_depth=temporal_depth,
        temporal_heads=temporal_heads,
        num_classes=2,
    )
    return model, img_size, backbone


def _apply_temperature(logits: torch.Tensor, T: Optional[float]) -> torch.Tensor:
    if T is None or T <= 0:
        return logits
    return logits / float(T)


# ------------------------------
# Public API
# ------------------------------
class MixedModelDetector:
    """
    Fully self-contained mixed detector that only reads assets from the package.
    Use:
        det = MixedModelDetector.from_package(device="auto", threshold=None)
        res = det.predict_frames([...])
    """

    def __init__(
        self,
        model: _UniTraxLite,
        device: torch.device,
        img_size: int,
        default_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.img_size = int(img_size)
        self.default_threshold = default_threshold if default_threshold is None else float(default_threshold)
        self.temperature = temperature if temperature is None else float(temperature)

        # AMP dtype: prefer bfloat16 if available
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # ---- constructors ----
    @classmethod
    def from_package(cls, device: str = "auto", threshold: Optional[float] = None) -> "MixedModelDetector":
        dev = _device_select(device)
        assets = _load_assets_from_package()
        model, img_size, _ = _build_model_from_cfg(assets.cfg_path)

        # Load checkpoint
        ckpt = torch.load(str(assets.ckpt_path), map_location="cpu")
        # Prefer EMA weights if present inside the state dict
        state = ckpt.get("ema", None)
        if state and isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            # fall back to raw model weights
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)

        # threshold priority: user > assets.threshold > None
        thr = float(threshold) if threshold is not None else (assets.threshold if assets.threshold is not None else None)

        return cls(
            model=model,
            device=dev,
            img_size=img_size,
            default_threshold=thr,
            temperature=assets.temperature,
        )

    # ---- inference helpers ----
    @torch.no_grad()
    def _predict_tensor(self, x_btchw: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        x_btchw: [B,T,C,H,W] tensor (ImageNet-normalized)
        returns: (probs[:,1], logits[:,1]) as numpy
        """
        x_btchw = x_btchw.to(self.device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=(self.device.type == "cuda")):
            out = self.model(x_btchw)
            logits = out["logits_clip"]  # [B,2]
            logits = _apply_temperature(logits, self.temperature)
            probs = torch.softmax(logits, dim=1)     # [B,2]
            p_ai = probs[:, 1].float().cpu().numpy()
            z_ai = logits[:, 1].float().cpu().numpy()
        return p_ai, z_ai

    # ---- public methods ----
    @torch.no_grad()
    def predict_frames(
        self,
        frame_paths: List[str],
        threshold: Optional[float] = None,
        out_dir: Optional[str] = None,
        save_visuals: bool = False,
    ) -> Dict[str, Any]:
        """
        Treats provided frames as a single clip.
        """
        if not frame_paths:
            return {"prob_ai": 0.0, "threshold": (threshold or self.default_threshold or 0.5), "n_frames": 0}

        imgs = [_imread_rgb(p, self.img_size) for p in frame_paths]
        t = _to_tensor_bchw(imgs)[None, ...]  # [1,T,3,H,W]
        p_ai, _ = self._predict_tensor(t)
        thr = float(threshold if threshold is not None else (self.default_threshold if self.default_threshold is not None else 0.5))
        label = "AI" if float(p_ai[0]) >= thr else "Natural"

        if save_visuals and out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            txt = Path(out_dir) / "mixed_verdict.txt"
            txt.write_text(f"p(AI)={float(p_ai[0]):.4f}, threshold={thr:.2f}, label={label}\n")

        return {
            "prob_ai": float(p_ai[0]),
            "threshold": thr,
            "label": label,
            "n_frames": len(frame_paths),
        }

    @torch.no_grad()
    def predict_video(
        self,
        video_path: str,
        sample_n: int = 32,
        interval: int = 1,
        threshold: Optional[float] = None,
        out_dir: Optional[str] = None,
        save_visuals: bool = False,
    ) -> Dict[str, Any]:
        """
        Uniformly samples frames from a video as a single clip.
        """
        frames = load_video_uniform(video_path, sample_n=max(1, int(sample_n)), every_n=max(1, int(interval)))
        if not frames:
            return {"prob_ai": 0.0, "threshold": (threshold or self.default_threshold or 0.5), "n_frames": 0}
        imgs = [_imread_rgb(p, self.img_size) for p in frames]
        t = _to_tensor_bchw(imgs)[None, ...]  # [1,T,3,H,W]
        p_ai, _ = self._predict_tensor(t)
        thr = float(threshold if threshold is not None else (self.default_threshold if self.default_threshold is not None else 0.5))
        label = "AI" if float(p_ai[0]) >= thr else "Natural"

        if save_visuals and out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            txt = Path(out_dir) / "mixed_verdict.txt"
            txt.write_text(f"p(AI)={float(p_ai[0]):.4f}, threshold={thr:.2f}, label={label}\n")

        return {
            "prob_ai": float(p_ai[0]),
            "threshold": thr,
            "label": label,
            "n_frames": len(frames),
        }
