from __future__ import annotations
import cv2
import torch
import numpy as np

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".webm",".m4v"}

def load_image(path: str, size: int) -> torch.Tensor:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    im = torch.from_numpy(im).float().div(255.0).permute(2,0,1)  # [C,H,W]
    return im.unsqueeze(0).unsqueeze(0)  # [B=1,T=1,C,H,W]

def load_video_uniform(path: str, size: int, T: int = 16) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise FileNotFoundError(path)
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(length-1, 0), T).astype(int) if length>0 else np.arange(T)
    i = 0; ret=True
    while ret and i <= idxs[-1]:
        ret, frame = cap.read()
        if not ret: break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(frame).float().div(255.0).permute(2,0,1))
        i += 1
    cap.release()
    if not frames:
        # fallback single black frame
        frames = [torch.zeros(3, size, size)]
    x = torch.stack(frames, dim=0)      # [T,C,H,W]
    return x.unsqueeze(0)                # [B=1,T,C,H,W]
