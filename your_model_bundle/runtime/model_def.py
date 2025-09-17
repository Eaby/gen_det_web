from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import timm

class UniTraxLite(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: Optional[int] = None,
        backbone_name: str = "eva02_base_patch14_224",
        temporal_type: str = "mean",
        temporal_depth: int = 2,
        temporal_heads: int = 4,
        num_classes: int = 2,
        pretrained_backbone: bool = False,   # IMPORTANT: offline default
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,  # keep False for offline!
            num_classes=0,                   # pooled feature head
        )
        if hasattr(self.backbone, "num_features"):
            self.embed_dim = int(self.backbone.num_features)
        elif hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(self.backbone.embed_dim)
        else:
            self.embed_dim = embed_dim if embed_dim is not None else 768

        if embed_dim is not None and embed_dim != self.embed_dim:
            self.backbone_head = nn.Linear(self.embed_dim, embed_dim)
            self.embed_dim = embed_dim
        else:
            self.backbone_head = None

        # try to enable grad checkpointing (no harm at inference)
        if hasattr(self.backbone, "set_grad_checkpointing"):
            try: self.backbone.set_grad_checkpointing(True)
            except Exception: pass

        self.temporal_type = temporal_type.lower()
        if self.temporal_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim, nhead=temporal_heads,
                batch_first=True, norm_first=True
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_depth)
        else:
            self.temporal = None

        self.fc_frame = nn.Linear(self.embed_dim, num_classes)
        self.fc_clip  = nn.Linear(self.embed_dim, num_classes)

        self.fc_s: Optional[nn.Linear] = None
        self.fc_a: Optional[nn.Linear] = None
        self.scale_bias = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.add_bias   = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def _ensure_proj(self, x: torch.Tensor, attr: str) -> nn.Module:
        in_dim = int(x.shape[-1])
        m = getattr(self, attr)
        if (m is None) or (getattr(m, "in_features", None) != in_dim) or (m.out_features != self.embed_dim):
            proj = nn.Linear(in_dim, self.embed_dim)
            setattr(self, attr, proj.to(x.device))
            return proj
        return m

    def _project_and_reshape(self, x_in: Optional[torch.Tensor], attr: str, B: int, T: int, D: int):
        if x_in is None: return None
        if x_in.dim() == 3: x2 = x_in.contiguous().view(B*T, x_in.size(-1))
        elif x_in.dim() == 2: x2 = x_in
        elif x_in.dim() == 1: x2 = x_in.unsqueeze(0)
        else: raise ValueError(f"Unexpected cue shape: {tuple(x_in.shape)}")
        proj = self._ensure_proj(x2, attr); x_proj = proj(x2)
        N = x_proj.size(0)
        if N == B*T:  out = x_proj.view(B, T, D)
        elif N == T:  out = x_proj.view(1, T, D).expand(B, T, D)
        elif N == B:  out = x_proj.view(B, 1, D).expand(B, T, D)
        elif N == 1:  out = x_proj.view(1, 1, D).expand(B, T, D)
        else: raise ValueError(f"Cannot reshape cue N={N} to [B={B},T={T},D={D}]")
        return out

    def forward(self, x: torch.Tensor, cues: Optional[Dict[str, Any]], T_list):
        # x: [B,T,C,H,W] in *ImageNet-normalized* space upstream
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B*T, C, H, W))
        if isinstance(feats, (tuple, list)): feats = feats[-1]
        if self.backbone_head is not None: feats = self.backbone_head(feats)
        D = feats.size(-1)
        feats = feats.view(B, T, D)

        scale = add = None
        if cues is not None:
            s = cues.get("s", None); a = cues.get("a", None)
            s_raw = self._project_and_reshape(s, "fc_s", B, T, D) if s is not None else None
            a_raw = self._project_and_reshape(a, "fc_a", B, T, D) if a is not None else None
            if s_raw is not None: scale = torch.sigmoid(s_raw)
            if a_raw is not None: add = a_raw

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
