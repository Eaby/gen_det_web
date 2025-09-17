from __future__ import annotations
import gradio as gr

from ui.theme import HUB_CSS, LOGO_HTML

# Tiny JS to toggle dark mode and swap the icon
THEME_TOGGLE_JS = r"""
() => {
  const b = document.querySelector('#theme-toggle button, #theme-toggle');
  const body = document.body;
  const dark = body.classList.toggle('dark');
  if (b) b.innerText = dark ? '☀️' : '🌙';
}
"""

def build_hub(app_title: str = "Synthetic Data Detection Hub", dark_default: bool = True, css_logo: bool = True):
    """
    Returns a dict of components and the outer demo Blocks.
    Pages:
      - landing
      - page_image
      - page_video
      - page_text
    """
    with gr.Blocks(css=HUB_CSS, title=app_title, theme=gr.themes.Soft(primary_hue="blue")) as demo:
        # (Optional) default to dark on first paint, very lightweight
        if dark_default:
            gr.HTML(
                "<script>document.addEventListener('DOMContentLoaded',()=>document.body.classList.add('dark'));</script>",
                visible=False
            )

        # ---------------- Top bar ----------------
        with gr.Row(elem_id="hub-topbar"):
            # Left spacer (for perfect centering)
            left_spacer = gr.HTML("&nbsp;", elem_id="hub-left")
            # Centered title
            title_md = gr.Markdown(f"# {app_title}", elem_id="hub-title")
            # Right controls: theme toggle + logo
            with gr.Row(elem_id="hub-right"):
                btn_theme = gr.Button("☀️" if dark_default else "🌙", elem_id="theme-toggle", variant="secondary")
                gr.HTML('<img src="file=ui/assets/sddh_logo.svg" alt="Hub Logo" style="height:36px;"/>', elem_id="hub-logo")

        # Bind theme toggle JS here (keeps app.py unchanged)
        btn_theme.click(None, js=THEME_TOGGLE_JS)

        # ---------------- Landing ----------------
        with gr.Column(visible=True, elem_id="landing") as landing:
            gr.Markdown(
                "Choose what you want to check for **synthetic (AI-generated)** content.",
                elem_id="hub-subtitle",
            )
            with gr.Row(elem_id="hub-cards"):
                with gr.Column(elem_id="card"):
                    gr.Markdown("### Synthetic Image Detection")
                    btn_go_image = gr.Button("Start Image Check", variant="primary", elem_id="cta")
                with gr.Column(elem_id="card"):
                    gr.Markdown("### Synthetic Video Detection")
                    btn_go_video = gr.Button("Start Video Check", variant="primary", elem_id="cta")
                with gr.Column(elem_id="card"):
                    gr.Markdown("### Synthetic Text Detection")
                    btn_go_text = gr.Button("Start Text Check", variant="primary", elem_id="cta")

        # ---------------- Image Page ----------------
        with gr.Column(visible=False, elem_id="page-image") as page_image:
            with gr.Row():
                gr.Markdown("## Image Check")
                btn_back_from_img = gr.Button("← Back", elem_id="back-btn")
            with gr.Row():
                img_input = gr.Image(label="Upload an image", type="pil", elem_id="tight")
            with gr.Row():
                i_ckpt_path = gr.Textbox(label="Model checkpoint (.ckpt / .pt)", placeholder="Path to image model", lines=1)
            with gr.Row():
                i_device = gr.Dropdown(["auto","cpu","cuda"], value="auto", label="Device")
                i_threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Decision threshold")
                img_btn = gr.Button("Analyze Image", variant="primary")
            img_gallery = gr.Gallery(label="Result", columns=1, height=280)
            img_html = gr.HTML(label="Summary")

        # ---------------- Video Page ----------------
        with gr.Column(visible=False, elem_id="page-video") as page_video:
            with gr.Row():
                gr.Markdown("## Video Check")
                btn_back_from_vid = gr.Button("← Back", elem_id="back-btn")
            with gr.Group():
                v_video_file = gr.File(label="Upload a video")
                with gr.Row():
                    v_model_kind = gr.Dropdown(
                        ["Mixed Model","Video Model"],
                        value="Mixed Model",
                        label="Choose model",
                    )
                    v_ckpt_path = gr.Textbox(label="Mixed model (.pt)", placeholder="/path/to/best.pt")
                    v_device = gr.Dropdown(["auto","cpu","cuda"], value="auto", label="Device")
                    v_threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Decision threshold")

                with gr.Accordion("Advanced extraction settings (optional)", open=False):
                    with gr.Row():
                        v_interval = gr.Slider(5, 120, value=30, step=1, label="Fixed frame stride")
                        v_motion_sens = gr.Slider(1.1, 3.0, value=1.5, step=0.05, label="Motion sensitivity")
                        v_lm_jitter = gr.Slider(0.001, 0.02, value=0.005, step=0.001, label="Face jitter threshold")
                    with gr.Row():
                        v_entropy_top = gr.Slider(5, 50, value=10, step=1, label="Top entropy frames")
                        v_scene_thr = gr.Slider(0.1, 1.0, value=0.35, step=0.05, label="Scene-change threshold")
                        v_sample_n = gr.Slider(1, 60, value=10, step=1, label="Freq. sample stride")
                        v_z_sigma = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="Hybrid Z-threshold")

                v_config_file = gr.File(label="(Optional) Config file", visible=False)
                v_process_btn = gr.Button("Analyze Video", variant="primary")

            with gr.Tabs(elem_id="tabs"):
                with gr.TabItem("Frames"):
                    with gr.Row():
                        v_i_g  = gr.Gallery(label="I-Frames", columns=8, height=160)
                    with gr.Row():
                        v_m_g  = gr.Gallery(label="Motion", columns=8, height=160)
                    with gr.Row():
                        v_lm_g = gr.Gallery(label="Face Jitter", columns=8, height=160)
                    with gr.Row():
                        v_fx_g = gr.Gallery(label="Fixed", columns=8, height=160)
                    with gr.Row():
                        v_en_g = gr.Gallery(label="Entropy", columns=8, height=160)
                    with gr.Row():
                        v_sc_g = gr.Gallery(label="Scene Changes", columns=8, height=160)
                with gr.TabItem("Frequency Analysis"):
                    v_plot = gr.Plot(label="Hybrid Score")
                    v_anomaly_dd = gr.Dropdown(label="Jump to anomaly", choices=[])
                    v_show_fr = gr.Image(label="Selected frame")
                with gr.TabItem("Report"):
                    v_report_html = gr.HTML()
                with gr.TabItem("Logs"):
                    v_logs = gr.Textbox(lines=10)

            v_out_dir = gr.Textbox(label="Output folder")
            v_verdict_html = gr.HTML(label="Overall Verdict")

        # ---------------- Text Page ----------------
        with gr.Column(visible=False, elem_id="page-text") as page_text:
            with gr.Row():
                gr.Markdown("## Text Check")
                btn_back_from_txt = gr.Button("← Back", elem_id="back-btn")
            t_input = gr.Textbox(label="Paste text here", lines=6, placeholder="Paste the text you want to check...")
            with gr.Row():
                t_ckpt_path = gr.Textbox(label="(Optional) Text model path", placeholder="/path/to/text-model.pt")
                t_device = gr.Dropdown(["auto","cpu","cuda"], value="auto", label="Device")
                t_threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Decision threshold")
            t_btn = gr.Button("Analyze Text", variant="primary")
            t_html = gr.HTML()

    # Return all handles in a dict
    return {
        # root
        "demo": demo,
        # containers
        "landing": landing,
        "page_image": page_image,
        "page_video": page_video,
        "page_text": page_text,
        # theme toggle
        "btn_theme": btn_theme,
        # landing buttons
        "btn_go_image": btn_go_image,
        "btn_go_video": btn_go_video,
        "btn_go_text": btn_go_text,
        # back buttons
        "btn_back_from_img": btn_back_from_img,
        "btn_back_from_vid": btn_back_from_vid,
        "btn_back_from_txt": btn_back_from_txt,
        # image page
        "img_input": img_input, "i_ckpt_path": i_ckpt_path, "i_device": i_device, "i_threshold": i_threshold,
        "img_btn": img_btn, "img_gallery": img_gallery, "img_html": img_html,
        # video page
        "v_video_file": v_video_file, "v_model_kind": v_model_kind, "v_ckpt_path": v_ckpt_path,
        "v_config_file": v_config_file, "v_device": v_device, "v_threshold": v_threshold,
        "v_interval": v_interval, "v_motion_sens": v_motion_sens, "v_lm_jitter": v_lm_jitter,
        "v_entropy_top": v_entropy_top, "v_scene_thr": v_scene_thr, "v_sample_n": v_sample_n, "v_z_sigma": v_z_sigma,
        "v_process_btn": v_process_btn,
        "v_i_g": v_i_g, "v_m_g": v_m_g, "v_lm_g": v_lm_g, "v_fx_g": v_fx_g, "v_en_g": v_en_g, "v_sc_g": v_sc_g,
        "v_plot": v_plot, "v_anomaly_dd": v_anomaly_dd, "v_show_fr": v_show_fr,
        "v_report_html": v_report_html, "v_logs": v_logs, "v_out_dir": v_out_dir, "v_verdict_html": v_verdict_html,
        # text page
        "t_input": t_input, "t_ckpt_path": t_ckpt_path, "t_device": t_device, "t_threshold": t_threshold,
        "t_btn": t_btn, "t_html": t_html,
    }
