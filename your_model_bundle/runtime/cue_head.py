import torch
import torch.nn as nn

class CueHead(nn.Module):
    """Light, fully-offline cue head used at inference (no downloads)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor, meta=None):
        # frames: [B,T,C,H,W], assumed in [0,1] range
        B, T, C, H, W = frames.shape
        x = frames.view(B*T, C, H, W)
        hf = self.conv(x).mean(dim=[2,3]).view(B, T)   # high-frequency proxy
        var = frames.var(dim=[2,3,4])                  # variance as "entropy-like"
        w = hf
        return {"s": var, "a": hf, "w": w}
