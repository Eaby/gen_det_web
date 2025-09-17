# controllers/video_controller.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
import shutil
import subprocess
import time
import sys, os
import math
import importlib.util  # <-- ADDED IMPORT

import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- USER CONFIGURATION: Set this to your model bundle directory name ---
# This is the name of the directory containing the 'runtime' module.
# For example, if your model is in 'my_cool_model/runtime/...',
# then set this value to "my_cool_model".
MODEL_BUNDLE_DIR_NAME = "your_model_bundle"  # <-- !!! CORRECTED VALUE !!!
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Keep this import for the "Video Model" placeholder branch (unchanged)
from models.video_model import run_mixed_pt_model_on_clip  # noqa: F401

# ------------------ MAKE BUNDLE IMPORTABLE RELIABLY ----------------------
def _ensure_bundle_on_path():
    """
    Ensure the directory that CONTAINS the model bundle is on sys.path.
    Priority:
      1) YMB_PATH env var (points to the parent dir of the bundle)
      2) Project root inferred from this file (../)
      3) Current working directory
    """
    env = os.environ.get("YMB_PATH", "").strip()
    if env:
        p = Path(env).resolve()
        if (p / MODEL_BUNDLE_DIR_NAME).exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return

    here = Path(__file__).resolve()
    proj = here.parents[1]  # controllers/.. -> project root
    if (proj / MODEL_BUNDLE_DIR_NAME).exists() and str(proj) not in sys.path:
        sys.path.insert(0, str(proj))
        return

    cwd = Path.cwd()
    if (cwd / MODEL_BUNDLE_DIR_NAME).exists() and str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
        return

_ensure_bundle_on_path()

# REPLACE WITH THIS BLOCK
try:
    # Use importlib to dynamically load the module based on the configured directory name
    module_name = f"{MODEL_BUNDLE_DIR_NAME}.runtime.detector"
    spec = importlib.util.find_spec(module_name)
    if spec and spec.loader:
        detector_module = importlib.util.module_from_spec(spec)
        # Add the module to sys.modules BEFORE executing it
        sys.modules[module_name] = detector_module  # <-- THIS IS THE FIX
        spec.loader.exec_module(detector_module)
        MixedModelDetector = detector_module.MixedModelDetector
        _BUNDLE_OK = True
    else:
        raise ImportError(f"Could not find the module: {module_name}")
except Exception as e:
    # This will print the true, underlying error to your console
    print("--- CAUGHT THE REAL EXCEPTION ---")
    import traceback
    traceback.print_exc()
    print("---------------------------------")
    MixedModelDetector = None  # type: ignore
    _BUNDLE_OK = False


# ---- cache the packaged detector so we don’t reload every click
_MIXED_DET = None
def _get_mixed_detector(device: str = "auto"):
    """Create/cached MixedModelDetector from the model bundle."""
    global _MIXED_DET
    if not _BUNDLE_OK:
        # Improved error message for better debugging
        raise RuntimeError(
            f"'{MODEL_BUNDLE_DIR_NAME}' MixedModelDetector not importable. "
            f"Please check the following:\n"
            f"1) Is MODEL_BUNDLE_DIR_NAME set correctly at the top of video_controller.py?\n"
            f"2) Is the '{MODEL_BUNDLE_DIR_NAME}' directory located in your project root?\n"
            f"3) If it's located elsewhere, is its parent directory in your PYTHONPATH or YMB_PATH?"
        )
    if _MIXED_DET is None:
        _MIXED_DET = MixedModelDetector.from_package(device=device)
    return _MIXED_DET

# ------------------ OPTIONAL MEDIAPIPE (landmark jitter) -----------------
try:
    import mediapipe as mp  # type: ignore
    MP_OK = True
except Exception:
    MP_OK = False

# ------------------ SAFETY LIMITS / BUDGETS ------------------------------
FFMPEG_TIMEOUT_S    = 45     # max seconds per ffmpeg call
TOTAL_TIME_BUDGET_S = 120    # soft cap for the whole pipeline
FREQ_MAX_SAMPLES    = 800    # max points for frequency analysis
FIXED_MAX_SAVE      = 1200   # cap saved fixed frames
MOTION_STRIDE       = 4      # evaluate every Nth frame for motion
MOTION_MAX_EVAL     = 3000   # cap number of motion evaluations
JITTER_STRIDE       = 6      # evaluate every Nth frame for jitter
JITTER_MAX_EVAL     = 1500   # cap number of jitter evaluations

# ------------------ PATHS / META ----------------------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _copy_upload_to_session(video_file: Any, base_root: Path) -> Tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = _ensure_dir(base_root / f"session_{ts}")
    src = Path(video_file.name)
    dst = session / src.name
    shutil.copy(src, dst)
    return session, dst

def _video_meta(path: Path) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return frames, fps

# ------------------ EXTRACTORS (bounded) --------------------------------
def _extract_fixed_interval_frames(video_path: Path, out_dir: Path, every_n: int) -> List[Path]:
    out = _ensure_dir(out_dir / "fixed_frames")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []
    idx, saved = 0, 0
    paths: List[Path] = []
    every_n = max(1, int(every_n))
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % every_n == 0:
            p = out / f"fixed_{saved:05d}.png"
            cv2.imwrite(str(p), frame); paths.append(p); saved += 1
            if saved >= FIXED_MAX_SAVE:
                break
        idx += 1
    cap.release()
    return paths

def _run_ffmpeg_with_timeout(cmd: List[str], timeout_s: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as e:
        # attach a helpful message and bubble up
        e.stdout = e.stdout or ""
        e.stderr = (e.stderr or "") + "\n[timeout] ffmpeg killed due to timeout."
        raise

def _extract_i_frames(video_path: Path, out_dir: Path, start_ts: float, logs: List[str]) -> List[Path]:
    out = _ensure_dir(out_dir / "i_frames")
    cmd = [
        "ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y",
        "-skip_frame","nokey","-i",str(video_path),
        "-vf","scale=in_range=limited:out_range=full,format=rgb24",
        "-vsync","0","-frame_pts","true",str(out / "iframe_%03d.png")
    ]
    try:
        _run_ffmpeg_with_timeout(cmd, FFMPEG_TIMEOUT_S)
    except Exception as e:
        logs.append(f"[warn] I-frame extraction skipped ({type(e).__name__})")
    return sorted(out.glob("iframe_*.png"))

def _extract_scene_change_frames(video_path: Path, out_dir: Path, threshold: float, start_ts: float, logs: List[str]) -> List[Path]:
    out = _ensure_dir(out_dir / "scene_frames")
    select = f"select=gt(scene\\,{threshold})"
    cmd = [
        "ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y",
        "-i",str(video_path),
        "-vf",f"{select},scale=in_range=limited:out_range=full,format=rgb24",
        "-vsync","vfr",str(out / "scene_%03d.png")
    ]
    try:
        _run_ffmpeg_with_timeout(cmd, FFMPEG_TIMEOUT_S)
    except Exception as e:
        logs.append(f"[warn] Scene-change extraction skipped ({type(e).__name__})")
    return sorted(out.glob("scene_*.png"))

def _extract_high_motion_frames(video_path: Path, out_dir: Path, sensitivity: float, start_ts: float) -> List[Path]:
    out = _ensure_dir(out_dir / "motion_frames")
    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    if not ret: cap.release(); return []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    window: List[float] = []
    saved = 0
    evals = 0
    idx = 0
    while True:
        ok = cap.grab()  # fast skip
        if not ok: break
        if idx % MOTION_STRIDE == 0:
            ret, frame = cap.retrieve()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            m = float(np.mean(mag))
            window.append(m)
            if len(window) > 30: window.pop(0)
            denom = np.mean(window) if window else 1.0
            denom = denom if not math.isclose(denom, 0.0) else 1.0
            if m > denom * float(sensitivity):
                cv2.imwrite(str(out / f"motion_{saved:04d}.png"), frame); saved += 1
            prev_gray = gray
            evals += 1
            if evals >= MOTION_MAX_EVAL: break
        idx += 1
        if time.time() - start_ts > TOTAL_TIME_BUDGET_S: break
    cap.release()
    return sorted(out.glob("motion_*.png"))

def _extract_landmark_jitter_frames(video_path: Path, out_dir: Path, jitter_thresh: float, start_ts: float) -> List[Path]:
    out = _ensure_dir(out_dir / "landmark_jitter_frames")
    if not MP_OK: return []
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    cap = cv2.VideoCapture(str(video_path))
    prev_lm = None; saved = 0; evals = 0; idx = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % JITTER_STRIDE == 0:
            ret, frame = cap.retrieve()
            if not ret: break
            res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = np.array([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark])
                if prev_lm is not None:
                    if float(np.mean(np.linalg.norm(lm - prev_lm, axis=1))) > float(jitter_thresh):
                        cv2.imwrite(str(out / f"jitter_{saved:04d}.png"), frame); saved += 1
                prev_lm = lm
            evals += 1
            if evals >= JITTER_MAX_EVAL: break
        idx += 1
        if time.time() - start_ts > TOTAL_TIME_BUDGET_S: break
    cap.release()
    return sorted(out.glob("jitter_*.png"))

def _extract_entropy_frames(base_dir: Path, top_n: int) -> List[Path]:
    src = base_dir / "fixed_frames"; out = _ensure_dir(base_dir / "entropy_frames")
    if not src.exists(): return []
    scores: List[float] = []; paths: List[Path] = []
    for f in sorted(src.iterdir()):
        if f.suffix.lower() not in {".png",".jpg",".jpeg"}: continue
        img = cv2.imread(str(f)); 
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256]); hist /= (hist.sum()+1e-9)
        ent = float(-(hist * np.log2(hist+1e-12)).sum())
        scores.append(ent); paths.append(f)
    if not scores: return []
    idxs = np.argsort(scores)[-int(top_n):]
    out_paths = []
    for i, idx in enumerate(idxs):
        img = cv2.imread(str(paths[idx]))
        if img is not None:
            p = out / f"entropy_{i:04d}.png"
            cv2.imwrite(str(p), img)
            out_paths.append(p)
    return sorted(out_paths)

def _frequency_analysis(video_path: Path, sample_every_n: int, z_thresh_sigma: float) -> Tuple[List[int], List[float], float, List[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], 0.0, []
    ent_list, frq_list, fr_ids = [], [], []
    step = max(1, int(sample_every_n))
    emitted = 0
    while True:
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = cap.read()
        if not ok: break
        if current % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray],[0],None,[256],[0,256]); hist /= (hist.sum() + 1e-9)
            ent = float(-(hist * np.log2(hist + 1e-12)).sum())
            ent_list.append(ent)
            f = np.fft.fftshift(np.fft.fft2(gray))
            frq_list.append(float(np.mean(20 * np.log(np.abs(f) + 1))))
            fr_ids.append(current)
            emitted += 1
            if emitted >= FREQ_MAX_SAMPLES:
                break
    cap.release()
    if not fr_ids:
        return [], [], 0.0, []
    ent = np.array(ent_list); frq = np.array(frq_list)
    ez = (ent - ent.mean()) / (ent.std() + 1e-9)
    fz = (frq - frq.mean()) / (frq.std() + 1e-9)
    hybrid = ez + fz
    thresh = float(hybrid.mean() + z_thresh_sigma * hybrid.std())
    anomalies = [int(f) for f, h in zip(fr_ids, hybrid) if h > thresh]
    return fr_ids, hybrid.tolist(), thresh, anomalies

def _make_plot(frames: List[int], hybrid: List[float], threshold: float):
    fig, ax = plt.subplots(figsize=(8, 3))
    if frames and hybrid:
        ax.plot(frames, hybrid, marker="o", label="Hybrid")
        ax.axhline(threshold, ls="--", label=f"Thresh={threshold:.2f}")
        above = [(x, y) for x, y in zip(frames, hybrid) if y > threshold]
        if above:
            xs, ys = zip(*above); ax.scatter(xs, ys, zorder=5, label="Anomaly")
        ax.set_xlabel("Frame #"); ax.set_ylabel("Score"); ax.legend(loc="best")
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout(); return fig

# ------------------ MAIN ENTRY (Gradio) ----------------------------------
def run_video_pipeline(
    video_file,               # gr.File
    model_kind: str,          # "mixed" | "video"
    ckpt_path: str,           # .pt for mixed (ignored if using packaged bundle)
    config_file,              # unused for packaged bundle; kept for signature
    device: str,
    threshold: float,
    interval: int,
    motion_sens: float,
    lm_jitter: float,
    entropy_top: int,
    scene_thr: float,
    sample_n: int,
    z_sigma: float,
    max_frames: int = 2000,
):
    logs: List[str] = []
    if video_file is None:
        return ("Upload a video first.", [], [], [], [], [], [],
                None, [], None, "<p>No video.</p>", "No logs.", "<p></p>")

    t0 = time.time()
    try:
        session_dir, input_video = _copy_upload_to_session(video_file, Path("processed_frames"))
        logs.append(f"[ok] Session: {session_dir.name}")

        total_frames, fps = _video_meta(input_video)
        if total_frames and fps:
            dur_s = total_frames / max(fps, 1.0)
            logs.append(f"[meta] frames={total_frames}, fps={fps:.2f}, duration≈{dur_s:.1f}s")

        # 1) all (bounded) extractors
        fx = _extract_fixed_interval_frames(input_video, session_dir, every_n=int(interval))
        logs.append(f"[ok] Fixed frames: {len(fx)}")

        i_paths  = _extract_i_frames(input_video, session_dir, t0, logs);                 logs.append(f"[ok] I-frames: {len(i_paths)}")
        if time.time() - t0 > TOTAL_TIME_BUDGET_S: logs.append("[cap] time budget hit after I-frames")

        m_paths  = _extract_high_motion_frames(input_video, session_dir, float(motion_sens), t0); logs.append(f"[ok] Motion frames: {len(m_paths)}")
        if time.time() - t0 > TOTAL_TIME_BUDGET_S: logs.append("[cap] time budget hit after Motion")

        lm_paths = _extract_landmark_jitter_frames(input_video, session_dir, float(lm_jitter), t0); logs.append(f"[ok] Jitter frames: {len(lm_paths)}" + ("" if MP_OK else " (mediapipe not installed)"))
        if time.time() - t0 > TOTAL_TIME_BUDGET_S: logs.append("[cap] time budget hit after Jitter")

        en_paths = _extract_entropy_frames(session_dir, int(entropy_top));                 logs.append(f"[ok] Entropy frames: {len(en_paths)}")

        sc_paths = _extract_scene_change_frames(input_video, session_dir, float(scene_thr), t0, logs); logs.append(f"[ok] Scene-change frames: {len(sc_paths)}")

        # 2) frequency analysis (bounded)
        fr_ids, hybrid, thresh, anomalies = _frequency_analysis(input_video, sample_every_n=int(sample_n), z_thresh_sigma=float(z_sigma))
        logs.append(f"[ok] Frequency analysis: N={len(fr_ids)} samples, anomalies={len(anomalies)}, thresh={thresh:.2f}")

        # 3) Inference
        mk = (model_kind or "").strip().lower()
        if "mixed" in mk:
            # Use the packaged your_model_bundle
            det = _get_mixed_detector(device=device)

            # Score up to 32 frames for speed + stability
            sel = fx[:32] if fx else []
            if not sel:
                return (f"Saved to: {session_dir}", [str(p) for p in i_paths], [str(p) for p in m_paths],
                        [str(p) for p in lm_paths], [str(p) for p in fx], [str(p) for p in en_paths], [str(p) for p in sc_paths],
                        _make_plot(fr_ids, hybrid, thresh), [str(a) for a in anomalies],
                        str(fx[0]) if fx else None,
                        "<p>No frames to score.</p>", "\n".join(logs), "<p></p>")

            # Prefer clip-level method if available; otherwise average per-frame
            try:
                p_nat = float(det.predict_clip(sel))
                try:
                    frame_scores = det.predict_paths(sel)
                except Exception:
                    frame_scores = [p_nat] * len(sel)
            except AttributeError:
                frame_scores = det.predict_paths(sel)  # list/ndarray of p(Nat)
                p_nat = float(np.mean(frame_scores))

            thr = float(threshold)
            nat_count = int(sum(1 for s in frame_scores if s >= thr))
            ai_count  = len(frame_scores) - nat_count
            counts = {"Natural": nat_count, "AI": ai_count}

            logs.append(f"[ok] Mixed model clip verdict p(Natural)={p_nat:.3f} on {len(sel)} frames")

        elif "video" in mk:
            # Your existing placeholder (or call your video-only model here)
            counts, p_nat = {"AI": 0, "Natural": 0}, 0.0
            logs.append("[info] Video-only model placeholder — no inference run.")
        else:
            return ("Select 'mixed' or 'video' for Video Detection.", [], [], [], [], [], [],
                    None, [], None, "<p>Unsupported model type for Video tab.</p>", "\n".join(logs), "<p></p>")

        # Verdict
        verdict = "Likely Natural" if p_nat >= 0.60 else ("Mixed" if 0.40 < p_nat < 0.60 else "Likely AI")
        verdict_html = f"<h3>Overall Verdict</h3><p><b>{verdict}</b> — p(Natural) ≈ <b>{p_nat*100:.1f}%</b></p>"

        # 4) plot + anomalies
        plot_fig = _make_plot(fr_ids, hybrid, thresh)
        anomaly_choices = [str(a) for a in anomalies]
        show_img = str(fx[0]) if (fx and anomalies) else (str(fx[0]) if fx else None)

        # 5) galleries (populate all)
        i_g  = [str(p) for p in i_paths]
        m_g  = [str(p) for p in m_paths]
        lm_g = [str(p) for p in lm_paths]
        fx_g = [str(p) for p in fx]
        en_g = [str(p) for p in en_paths]
        sc_g = [str(p) for p in sc_paths]

        # 6) report
        if "mixed" in mk:
            lab = "Natural" if p_nat >= float(threshold) else "AI"
            report = f"<h4>Clip Summary (Mixed)</h4><p>Verdict: <b>{lab}</b> (p(Natural)={p_nat:.2f}) using {min(len(fx), 32)} frames.</p>"
        else:
            report = "<h4>Video-only Model</h4><p>Placeholder — model not yet integrated.</p>"

        out_dir_text = f"Saved to: {session_dir}"
        return (
            out_dir_text, i_g, m_g, lm_g, fx_g, en_g, sc_g,
            plot_fig, anomaly_choices, show_img, report, "\n".join(logs), verdict_html
        )

    except Exception as e:
        err = f"[error] {type(e).__name__}: {e}"
        # Also print traceback for detailed debugging in the console
        import traceback
        traceback.print_exc()
        return ("Error.", [], [], [], [], [], [],
                None, [], None, f"<pre>{err}</pre>", err, "<p></p>")