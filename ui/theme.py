# Pastel Light + Neon Dark. Centered title, compact theme toggle, subtle animated grid & dots.
HUB_CSS = r"""
:root{
  --bg-light:#f4f9ff;
  --panel-light:#ffffff;
  --text-light:#0f172a;
  --muted-light:#475569;
  --accent-light:#2563eb;          /* blue-600 */
  --accent-light-2:#38bdf8;        /* sky-400 */

  --bg-dark:#070b11;
  --panel-dark:#0f1720;
  --text-dark:#e5f0ff;
  --muted-dark:#8aa4bf;
  --accent-dark:#22d3ee;           /* cyan-400 (neon) */
  --accent-dark-2:#60a5fa;         /* blue-400 */

  --radius:12px;
  --shadow-sm:0 4px 14px rgba(2,6,23,0.08);
  --shadow-dk:0 10px 30px rgba(0,0,0,0.35);
}

/* base backgrounds */
body { background: var(--bg-light); color: var(--text-light); }
.dark body { 
  background:
    radial-gradient(1200px 1200px at 100% -100%, rgba(34,211,238,0.08), transparent 60%),
    radial-gradient(900px 700px at 0% -100%, rgba(96,165,250,0.07), transparent 60%),
    var(--bg-dark);
  color: var(--text-dark);
}

/* animated subtle grid + micro-dots on the main container */
.gradio-container{
  background-image:
    radial-gradient(rgba(0,0,0,0.05) 1px, rgba(0,0,0,0) 1px),
    linear-gradient(rgba(0,0,0,0.08) 0.5px, transparent 0.5px);
  background-size: 16px 16px, 32px 32px;
  background-position: 0 0, 0 0;
  animation: grid-move 30s linear infinite;
}
.dark .gradio-container{
  background-image:
    radial-gradient(rgba(34,211,238,0.16) 1px, rgba(0,0,0,0) 1px),
    linear-gradient(rgba(96,165,250,0.16) 0.5px, transparent 0.5px);
}
@keyframes grid-move {
  0%   { background-position: 0 0, 0 0; }
  100% { background-position: 64px 64px, 32px 32px; }
}

/* Top bar: 3 columns -> left spacer | centered title | right controls */
#hub-topbar{
  display:grid; grid-template-columns: 1fr auto 1fr; align-items:center;
  margin: 6px 4px 8px 4px;
}
#hub-left{ opacity:0; } /* invisible spacer */

#hub-title{
  text-align:center !important; margin: 0 !important; padding: 8px 0;
  text-shadow: 0 1px 0 rgba(255,255,255,0.5);
  position: relative;
}
.dark #hub-title{
  text-shadow: 0 0 24px rgba(34,211,238,0.18);
}
#hub-title:after{
  content:""; display:block; width:200px; height:2px; margin:8px auto 0 auto;
  background: linear-gradient(90deg, transparent, var(--accent-light), transparent);
  border-radius:2px;
  opacity:0.55;
}
.dark #hub-title:after{
  background: linear-gradient(90deg, transparent, var(--accent-dark), transparent);
  box-shadow: 0 0 18px rgba(34,211,238,0.35);
}

#hub-right{ justify-content:flex-end; gap:10px; }

/* Tiny theme toggle (top-right) */
#theme-toggle{
  position: fixed; top: 10px; right: 12px; z-index: 1000;
  min-width: 28px !important; width: 28px; height: 28px; padding: 0 !important;
  border-radius: 999px !important; line-height: 28px; font-size: 14px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.dark #theme-toggle{ box-shadow: 0 0 16px rgba(34,211,238,0.28); }
#theme-toggle:hover{ filter:brightness(1.05); }

/* Top-right logo */
#hub-logo{ display:flex; align-items:center; }
#hub-logo svg{ height: 36px; filter: drop-shadow(0 1px 8px rgba(34,211,238,0.35)); }
.dark #hub-logo svg { filter: drop-shadow(0 1px 12px rgba(34,211,238,0.45)); }

/* cards on landing */
#hub-cards{ gap: 16px; }
#card{
  background: var(--panel-light); border-radius: var(--radius); padding:16px; box-shadow: var(--shadow-sm);
  border: 1px solid rgba(2,6,23,0.04);
  transition: transform .15s ease, box-shadow .2s ease, border-color .2s ease;
}
#card:hover{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(2,6,23,0.12); }
.dark #card{
  background: rgba(16,23,32,0.82); border:1px solid rgba(34,211,238,0.16); box-shadow: var(--shadow-dk);
}
.dark #card:hover{
  border-color: rgba(34,211,238,0.35);
  box-shadow: 0 12px 36px rgba(0,0,0,0.55), 0 0 18px rgba(34,211,238,0.18) inset;
}

#cta{ width:100%; font-weight:700; }
#back-btn{ margin-left:auto; }

/* Panels */
#page-image, #page-video, #page-text {
  background: var(--panel-light); border-radius: var(--radius); padding:16px; box-shadow: var(--shadow-sm); margin-top:8px;
  border: 1px solid rgba(2,6,23,0.06);
}
.dark #page-image, .dark #page-video, .dark #page-text{
  background: rgba(16,23,32,0.78); border:1px solid rgba(34,211,238,0.16); box-shadow: var(--shadow-dk);
}

/* Inputs & components */
.gradio-dropdown, .gradio-textbox, .gradio-file, .gradio-slider, .gradio-plot, .gradio-gallery, .gradio-image, .gradio-html{
  border-radius: var(--radius) !important;
  overflow:hidden;
}
button{ border-radius: var(--radius) !important; }
button.primary{ background: var(--accent-light) !important; color:white !important; }
button.primary:hover{ filter: brightness(1.05); }
.dark button.primary{ background: linear-gradient(90deg, var(--accent-dark), var(--accent-dark-2)) !important; }

/* Tabs underline glow */
#tabs .tab-nav button[aria-selected="true"]{
  box-shadow: inset 0 -2px 0 0 var(--accent-light);
}
.dark #tabs .tab-nav button[aria-selected="true"]{
  box-shadow: inset 0 -2px 0 0 var(--accent-dark), 0 0 18px rgba(34,211,238,0.18);
}

/* Tight image input */
#tight canvas, #tight img{ max-height: 360px !important; }

/* Force panel color in dark mode on all three pages */
.dark #page-image,
.dark #page-video,
.dark #page-text{
  background: rgba(16,23,32,0.82) !important;
  border: 1px solid rgba(34,211,238,0.16) !important;
}

/* Neutralize default white wrappers inside pages when dark */
.dark #page-image .gradio-row,
.dark #page-image .gradio-group,
.dark #page-image .gradio-accordion,
.dark #page-image .gradio-column,
.dark #page-video .gradio-row,
.dark #page-video .gradio-group,
.dark #page-video .gradio-accordion,
.dark #page-video .gradio-column,
.dark #page-text .gradio-row,
.dark #page-text .gradio-group,
.dark #page-text .gradio-accordion,
.dark #page-text .gradio-column{
  background: transparent !important;
  box-shadow: none !important;
}

/* Title underline pulse */
@keyframes neon-pulse { 0%{opacity:.4} 50%{opacity:.85} 100%{opacity:.4} }
.dark #hub-title:after{ animation: neon-pulse 3s ease-in-out infinite; }

/* Active tab glow */
.dark #tabs .tab-nav button[aria-selected="true"]{
  box-shadow: inset 0 -2px 0 0 var(--accent-dark), 0 0 18px rgba(34,211,238,0.20);
}
"""

# Simple inline futuristic logo (SVG)
LOGO_HTML = """
<svg viewBox="0 0 256 256" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Hub Logo">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="#22D3EE"/>
      <stop offset="1" stop-color="#60A5FA"/>
    </linearGradient>
  </defs>
  <circle cx="128" cy="128" r="118" stroke="url(#g)" stroke-width="6" opacity="0.9"/>
  <path d="M64 136c32-64 96-64 128 0" stroke="url(#g)" stroke-width="10" stroke-linecap="round"/>
  <circle cx="96" cy="128" r="12" fill="#22D3EE"/>
  <circle cx="160" cy="128" r="12" fill="#60A5FA"/>
  <path d="M80 176h96" stroke="url(#g)" stroke-width="8" stroke-linecap="round" opacity="0.9"/>
</svg>
"""
