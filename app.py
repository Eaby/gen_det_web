#!/usr/bin/env python3
from __future__ import annotations
import os, yaml
import gradio as gr

# Theme + layout
from ui.theme import HUB_CSS
from ui.layout import build_hub

# Controllers (unchanged)
from controllers.image_controller import run_single_image
from controllers.video_controller import run_video_pipeline
from controllers.text_controller import run_text_detection

# ------------------------- Load config -------------------------
CFG_PATH = os.environ.get("GDW_CONFIG", "configs/default.yaml")
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

ui_cfg = CFG.get("ui", {})
title_from_cfg = ui_cfg.get("title", "Synthetic Data Detection Hub")
dark_default = ui_cfg.get("dark_mode_default", True)

# ------------------------- Helpers -----------------------------
def _prefill_mixed_ckpt_path():
    # Default Mixed model .pt path (spaces are OK)
    return "/media/eaby/Trained Files from New model/eva224_tt_balanced/best.pt"

def _map_model_kind(kind: str) -> str:
    """Map UI dropdown labels to internal tokens expected by the video pipeline."""
    if not kind:
        return ""
    k = kind.strip().lower()
    if "mixed" in k: return "mixed"
    if "video" in k: return "video"
    return k

def _run_video_wrapper(
    video_file, model_kind, ckpt_path, config_file, device, threshold,
    interval, motion_sens, lm_jitter, entropy_top, scene_thr, sample_n, z_sigma
):
    return run_video_pipeline(
        video_file=video_file,
        model_kind=_map_model_kind(model_kind),   # "mixed" | "video"
        ckpt_path=ckpt_path,                      # .pt for mixed
        config_file=config_file,                  # ignored for mixed .pt (kept for compatibility)
        device=device,
        threshold=threshold,
        interval=interval,
        motion_sens=motion_sens,
        lm_jitter=lm_jitter,
        entropy_top=entropy_top,
        scene_thr=scene_thr,
        sample_n=sample_n,
        z_sigma=z_sigma,
    )

# ------------------------- Build UI ----------------------------
C = build_hub(app_title=title_from_cfg, dark_default=dark_default, css_logo=True)
demo = C["demo"]

with demo:
    # Prefill Mixed model path on load (video page)
    if C.get("v_ckpt_path"):
        demo.load(fn=_prefill_mixed_ckpt_path, inputs=None, outputs=[C["v_ckpt_path"]])

    # ---------- Landing → Page navigation ----------
    def _nav(to_page: str):
        # Hide all, show only the requested page (and landing off)
        show_landing = to_page == "landing"
        show_img = to_page == "image"
        show_vid = to_page == "video"
        show_txt = to_page == "text"
        return (
            gr.update(visible=show_landing),
            gr.update(visible=show_img),
            gr.update(visible=show_vid),
            gr.update(visible=show_txt),
        )

    C["btn_go_image"].click(lambda: _nav("image"), outputs=[C["landing"], C["page_image"], C["page_video"], C["page_text"]], queue=False)
    C["btn_go_video"].click(lambda: _nav("video"), outputs=[C["landing"], C["page_image"], C["page_video"], C["page_text"]], queue=False)
    C["btn_go_text"].click(lambda: _nav("text"),  outputs=[C["landing"], C["page_image"], C["page_video"], C["page_text"]], queue=False)

    # Back buttons
    C["btn_back_from_img"].click(lambda: _nav("landing"), outputs=[C["landing"], C["page_image"], C["page_video"], C["page_text"]], queue=False)
    C["btn_back_from_vid"].click(lambda: _nav("landing"), outputs=[C["landing"], C["page_image"], C["page_video"], C["page_text"]], queue=False)
    C["btn_back_from_txt"].click(lambda: _nav("landing"), outputs=[C["landing"], C["page_image"], C["page_video"], C["page_text"]], queue=False)

    # ---------- IMAGE page wiring (unchanged logic) ----------
    C["img_btn"].click(
        fn=run_single_image,
        inputs=[C["img_input"], C["i_ckpt_path"], C["i_threshold"], C["i_device"]],
        outputs=[C["img_gallery"], C["img_html"]],
        queue=True,
        concurrency_limit=2,  # allow 2 image jobs at once
    )

    # ---------- VIDEO page wiring (mixed/video) ----------
    video_inputs = [
        C["v_video_file"],     # gr.File
        C["v_model_kind"],     # "Mixed Model" | "Video Model"
        C["v_ckpt_path"],      # str (.pt for mixed)
        C["v_config_file"],    # ignored here
        C["v_device"],         # "auto" | "cpu" | "cuda"
        C["v_threshold"],      # float
        C["v_interval"],       # int
        C["v_motion_sens"],    # float
        C["v_lm_jitter"],      # float
        C["v_entropy_top"],    # int
        C["v_scene_thr"],      # float
        C["v_sample_n"],       # int
        C["v_z_sigma"],        # float
    ]

    video_outputs = [
        C["v_out_dir"], C["v_i_g"], C["v_m_g"], C["v_lm_g"], C["v_fx_g"], C["v_en_g"], C["v_sc_g"],
        C["v_plot"], C["v_anomaly_dd"], C["v_show_fr"],
        C["v_report_html"], C["v_logs"], C["v_verdict_html"],
    ]

    C["v_process_btn"].click(
        fn=_run_video_wrapper,
        inputs=video_inputs,
        outputs=video_outputs,
        queue=True,
        concurrency_limit=1,   # process one video at a time (keeps UI snappy)
    )

    # ---------- TEXT page wiring (unchanged logic / placeholder ok) ----------
    C["t_btn"].click(
        fn=run_text_detection,
        inputs=[C["t_input"], C["t_ckpt_path"], C["t_threshold"], C["t_device"]],
        outputs=[C["t_html"]],
        queue=True,
        concurrency_limit=2,
    )

# -------------------------- Launch -----------------------------
if __name__ == "__main__":
    demo.launch(
        server_name=ui_cfg.get("server_name", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", ui_cfg.get("server_port", 7860))),
        show_error=True,
        max_threads=8,  # total worker threads for background jobs
        inbrowser=False,
        debug=False,
    )
