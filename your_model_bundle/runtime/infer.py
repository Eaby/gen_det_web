from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
import yaml
import numpy as np

from .model_def import UniTraxLite
from .cue_head import CueHead
from .transforms import normalize_imagenet, resize_short_side
from .video_io import load_image, load_video_uniform, IMAGE_EXTS, VIDEO_EXTS

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def predict_path(bundle_dir: str, input_path: str):
    bundle = Path(bundle_dir)
    cfg = load_cfg(str(bundle/"config.yaml"))
    device = torch.device(cfg.get("runtime",{}).get("device","cuda") if torch.cuda.is_available() else "cpu")

    # model
    mcfg = cfg["model"]; dscfg = cfg["dataset"]
    model = UniTraxLite(
        img_size=dscfg["img_size"],
        patch_size=dscfg["patch_size"],
        backbone_name=mcfg["backbone_name"],
        temporal_type=mcfg.get("temporal_type","mean"),
        temporal_depth=int(mcfg.get("temporal_depth",2)),
        temporal_heads=int(mcfg.get("temporal_heads",4)),
        num_classes=2,
        pretrained_backbone=False,  # offline!
    ).to(device).eval()
    cue = CueHead().to(device).eval()

    ckpt = torch.load(str(bundle/"model.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if "cue" in ckpt: cue.load_state_dict(ckpt["cue"], strict=False)

    # load threshold / temperature if present
    threshold = 0.5
    tpath = bundle/"threshold.json"
    if tpath.exists():
        try:
            threshold = float(json.loads(tpath.read_text())["threshold"])
        except Exception:
            pass

    # input
    p = Path(input_path)
    size = int(dscfg["img_size"])

    if p.suffix.lower() in IMAGE_EXTS:
        x = load_image(str(p), size)            # [1,1,3,H,W]
    elif p.suffix.lower() in VIDEO_EXTS:
        x = load_video_uniform(str(p), size, T=16)  # [1,T,3,H,W]
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    # (Optional) resize short side if not square
    # x = resize_short_side(x, size)

    # normalize
    x = x.to(device)
    x = normalize_imagenet(x)

    # cues & forward
    use_amp = bool(cfg.get("runtime",{}).get("use_amp", True))
    with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type=="cuda")):
        cues = cue(x, None)
        out = model(x, cues, T_list=[x.size(1)])
        logits = out["logits_clip"]              # [1,2]
        prob_ai = torch.softmax(logits, dim=1)[0,1].item()
        pred = int(prob_ai >= threshold)

    return {
        "path": str(p),
        "prob_ai": float(prob_ai),
        "pred": int(pred),        # 1=AI, 0=Real
        "threshold": float(threshold),
        "label_name": "AI" if pred==1 else "Real",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to your_model_bundle directory")
    ap.add_argument("--input", required=True, help="Path to image or video")
    args = ap.parse_args()
    out = predict_path(args.bundle, args.input)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
