from __future__ import annotations
import cv2, subprocess, numpy as np, glob
from pathlib import Path
from typing import List, Dict, Tuple
from utils.io import ensure_dir, is_image
from utils.logging import nice

# --- I-frames via ffmpeg ---

def extract_i_frames(video_path: str, out_dir: Path, log: List[str]):
    out = ensure_dir(out_dir / "i_frames")
    cmd = [
        "ffmpeg", "-y", "-skip_frame", "nokey", "-i", video_path,
        "-vf", "scale=in_range=limited:out_range=full,format=rgb24",
        "-vsync", "0", "-frame_pts", "true", str(out / "iframe_%03d.png")
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        log.append(nice("WARN: ffmpeg I-frame extraction may have failed."))

# --- Fixed interval frames ---

def extract_fixed_interval_frames(video_path: str, out_dir: Path, interval: int, log: List[str]):
    out = ensure_dir(out_dir / "fixed_frames")
    cap = cv2.VideoCapture(video_path)
    idx = saved = 0
    if not cap.isOpened():
        log.append(nice("ERROR: Unable to open video for fixed-interval extraction.")); return
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % interval == 0:
            cv2.imwrite(str(out / f"fixed_{saved:04d}.png"), frame)
            saved += 1
        idx += 1
    cap.release(); log.append(nice(f"Saved {saved} fixed-interval frames (every {interval})."))

# --- High motion frames ---

def extract_high_motion_frames(video_path: str, out_dir: Path, sensitivity: float, log: List[str]):
    out = ensure_dir(out_dir / "motion_frames"); cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        log.append(nice("ERROR: Unable to read the first frame.")); cap.release(); return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY); saved = 0; window: List[float] = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        m = float(np.mean(mag)); window.append(m)
        if len(window) > 30: window.pop(0)
        if m > np.mean(window) * sensitivity:
            cv2.imwrite(str(out / f"motion_{saved:04d}.png"), frame); saved += 1
        prev_gray = gray
    cap.release(); log.append(nice(f"Saved {saved} high-motion frames (sensitivity {sensitivity}×)."))

# --- Landmark jitter frames (optional) ---
try:
    import mediapipe as mp  # type: ignore
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

def extract_landmark_jitter_frames(video_path: str, out_dir: Path, jitter_threshold: float, log: List[str]):
    out = ensure_dir(out_dir / "landmark_jitter_frames")
    if not MP_AVAILABLE:
        log.append(nice("WARN: MediaPipe not installed; skipping.")); return
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    cap = cv2.VideoCapture(video_path); prev_lm = None; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = np.array([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark])
            if prev_lm is not None:
                if float(np.mean(np.linalg.norm(lm - prev_lm, axis=1))) > jitter_threshold:
                    cv2.imwrite(str(out / f"jitter_{saved:04d}.png"), frame); saved += 1
            prev_lm = lm
    cap.release(); log.append(nice(f"Saved {saved} landmark-jitter frames (threshold {jitter_threshold})."))

# --- Entropy frames ---

def extract_entropy_frames(base_dir: Path, out_dir: Path, top_n: int, log: List[str]):
    src, out = base_dir / "fixed_frames", ensure_dir(base_dir / "entropy_frames")
    if not src.exists(): log.append(nice("WARN: No fixed-interval frames found.")); return
    scores, paths = [], []
    for f in sorted(src.iterdir()):
        if not f.is_file(): continue
        img = cv2.imread(str(f));
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist /= (hist.sum() + 1e-9)
        scores.append(float(-np.sum(hist * np.log2(hist + 1e-12))))
        paths.append(f)
    if not scores: log.append(nice("WARN: No frames to score for entropy.")); return
    idxs = np.argsort(scores)[-top_n:]
    for i, idx in enumerate(idxs):
        img = cv2.imread(str(paths[idx]))
        if img is not None:
            cv2.imwrite(str(out / f"entropy_{i:04d}.png"), img)
    log.append(nice(f"Saved {len(idxs)} high-entropy frames (top {top_n})."))

# --- Scene change (ffmpeg select) ---

def extract_scene_change_frames(video_path: str, out_dir: Path, threshold: float, log: List[str]):
    out = ensure_dir(out_dir / "scene_frames")
    select = f"select='gt(scene\,{threshold})',showinfo"
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vf",
        f"{select},scale=in_range=limited:out_range=full,format=rgb24",
        "-vsync", "vfr", str(out / "scene_%03d.png")
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0: log.append(nice("WARN: ffmpeg scene-change extraction may have failed."))

# --- Frequency analysis (entropy+FFT hybrid) ---

def run_frequency_analysis(video_path: str, out_dir: Path, sample_every_n: int, z_thresh_sigma: float, log: List[str]):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.append(nice("ERROR: Unable to open video.")); return [], [], [], {}, 0.0
    ent_list, freq_list, frame_list = [], [], []
    while True:
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES)); ret, frame = cap.read()
        if not ret: break
        if current % sample_every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]); hist /= (hist.sum() + 1e-9)
            ent_list.append(float(-np.sum(hist * np.log2(hist + 1e-12))))
            f = np.fft.fft2(gray); fshift = np.fft.fftshift(f)
            freq_list.append(float(np.mean(20 * np.log(np.abs(fshift) + 1))))
            frame_list.append(current)
    cap.release()
    if not frame_list:
        log.append(nice("WARN: No frames sampled.")); return [], [], [], {}, 0.0
    e_arr, f_arr = np.array(ent_list), np.array(freq_list)
    ez, fz = (e_arr - e_arr.mean()) / (e_arr.std() + 1e-9), (f_arr - f_arr.mean()) / (f_arr.std() + 1e-9)
    hybrid = ez + fz; thresh = float(hybrid.mean() + z_thresh_sigma * hybrid.std())
    freq_out = ensure_dir(out_dir / "frequency_frames"); cap2 = cv2.VideoCapture(video_path); frame_to_path = {}
    for fn in frame_list:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fn); r, fr2 = cap2.read()
        if r:
            fp = str(freq_out / f"freq_{fn:05d}.png"); cv2.imwrite(fp, fr2); frame_to_path[fn] = fp
    cap2.release(); log.append(nice(f"Frequency analysis: N={len(frame_list)}, threshold={thresh:.2f}."))
    return frame_list, hybrid.tolist(), [fn for fn, h in zip(frame_list, hybrid) if h > thresh], frame_to_path, thresh

# Utility

def gather_images(folder: Path) -> list[str]:
    if not folder.exists(): return []
    return [str(p) for p in sorted(folder.iterdir()) if p.is_file() and is_image(p)]
