from __future__ import annotations
import torch
import torch.nn.functional as F

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    # x: [B,T,C,H,W], values in [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,1,3,1,1)
    return (x - mean) / std

def resize_short_side(frames: torch.Tensor, size: int) -> torch.Tensor:
    # frames [B,T,C,H,W], bilinear to make min(H,W) == size, then center-crop to size
    B, T, C, H, W = frames.shape
    short, long = (H, W) if H < W else (W, H)
    scale = size / float(short)
    new_h, new_w = int(round(H*scale)), int(round(W*scale))
    x = frames.view(B*T, C, H, W)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # center crop to (size, size)
    top  = max(0, (new_h - size) // 2)
    left = max(0, (new_w - size) // 2)
    x = x[:, :, top:top+size, left:left+size]
    return x.view(B, T, C, size, size)
