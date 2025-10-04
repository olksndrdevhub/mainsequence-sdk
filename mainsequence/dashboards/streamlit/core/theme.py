from __future__ import annotations
import streamlit as st
from pathlib import Path


def inject_css_for_dark_accents():
    st.markdown(
        """
        <style>
        /* Subtle tweaks; Streamlit theme itself comes from .streamlit/config.toml */
        .stMetric > div { background: rgba(255,255,255,0.04); border-radius: 6px; padding: .5rem .75rem; }
        div[data-testid="stMetricDelta"] svg { display: none; } /* clean deltas, hide the arrow */
        </style>
        """,
        unsafe_allow_html=True
    )

def explain_theming():
    st.info(
        "Theme colors come from `.streamlit/config.toml`. "
        "You can’t switch Streamlit’s theme at runtime, but you can tune Plotly’s colors and inject light CSS."
    )




# --- Load spinner frames ONCE from two levels above, files: image_1_base64.txt ... image_5_base64.txt ---
def _load_spinner_frames_for_this_template() -> list[str]:
    base_dir = Path(__file__).resolve().parent.parent
    frames: list[str] = []
    for i in range(1, 6):
        p = base_dir / f"assets/image_{i}_base64.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing spinner frame file: {p}")
        frames.append(p.read_text(encoding="utf-8").strip())
    return frames


_SPINNER_FRAMES_RAW = _load_spinner_frames_for_this_template()


# Expose constants for the function (keeps the code below simple)
IMAGE_1_B64, IMAGE_2_B64, IMAGE_3_B64, IMAGE_4_B64, IMAGE_5_B64 = _SPINNER_FRAMES_RAW

def override_spinners(
    hide_deploy_button: bool = False,
    *,
    # Sizes
    top_px: int = 35,          # top-right toolbar & st.status icon base size
    inline_px: int = 288,       # animation size when centered
    # Timing
    duration_ms: int = 900,
    # Toolbar nudges / spacing
    toolbar_nudge_px: int = -3,
    toolbar_gap_left_px: int = 2,
    toolbar_left_offset_px: int = 0,
    # Centered overlay styling
    center_non_toolbar: bool = True,        # << keep True to center inline + status
    dim_backdrop: bool = True,              # << set False to hide the dark veil
    overlay_blur_px: float = 1.5,
    overlay_opacity: float = 0.35,
    overlay_z_index: int = 9990,            # keep below toolbar; we also lift toolbar above
) -> None:
    """Override Streamlit spinners with a 4-frame animation.
    - Toolbar spinner stays in the toolbar (top-right).
    - All other spinners (inline + st.status icon) are centered on screen.
    """

    def as_data_uri(s: str, mime="image/png") -> str:
        s = s.strip()
        return s if s.startswith("data:") else f"data:{mime};base64,{s}"

    i1 = as_data_uri(IMAGE_1_B64)
    i2 = as_data_uri(IMAGE_2_B64)
    i3 = as_data_uri(IMAGE_3_B64)
    i4 = as_data_uri(IMAGE_4_B64)
    i5= as_data_uri(IMAGE_5_B64)

    veil_bg = f"rgba(0,0,0,{overlay_opacity})"

    st.markdown(f"""
<style>
/* ---- 4-frame animation ---- */
@keyframes st-fourframe {{
  0%% {{ background-image:url("{i1}"); }}
  20%      {{ background-image:url("{i2}"); }}
  40%      {{ background-image:url("{i3}"); }}
  60%      {{ background-image:url("{i4}"); }}
   80% {{ background-image:url("{i5}"); }}
      100% {{ background-image:url("{i5}"); }}
}}

/* ---- CSS variables ---- */
:root {{
  --st-spin-top:{top_px}px;                 /* toolbar/status base size */
  --st-spin-inline:{inline_px}px;           /* centered spinner size */
  --st-spin-dur:{duration_ms}ms;

  --st-spin-toolbar-nudge:{toolbar_nudge_px}px;
  --st-spin-toolbar-gap:{toolbar_gap_left_px}px;
  --st-spin-toolbar-left:{toolbar_left_offset_px}px;

  --st-overlay-z:{overlay_z_index};
  --st-overlay-bg:{veil_bg};
  --st-overlay-blur:{overlay_blur_px}px;
}}

/* Lift toolbar above any overlay so Stop/Deploy remain clickable */
div[data-testid="stToolbar"],
[data-testid="stStatusWidget"] {{
  position: relative;
  z-index: calc(var(--st-overlay-z) + 5);
}}

/* =======================================================================
   1) Top-right toolbar widget  (kept in place, not centered)
   ======================================================================= */
[data-testid="stStatusWidget"] {{
  position:relative;
  padding-left: calc(var(--st-spin-top) + var(--st-spin-toolbar-gap));
}}
[data-testid="stStatusWidget"] svg,
[data-testid="stStatusWidget"] img {{ display:none !important; }}
[data-testid="stStatusWidget"]::before {{
  content:"";
  position:absolute;
  left: var(--st-spin-toolbar-left);
  top:50%;
  transform:translateY(-50%) translateY(var(--st-spin-toolbar-nudge));
  width:var(--st-spin-top);
  height:var(--st-spin-top);
  background:no-repeat center/contain;
  animation:st-fourframe var(--st-spin-dur) linear infinite;
}}

/* Hide the entire toolbar if requested */
{"div[data-testid='stToolbar']{display:none !important;}" if hide_deploy_button else ""}

/* =======================================================================
   2) Inline spinner (st.spinner) — centered overlay
   ======================================================================= */
[data-testid="stSpinner"] svg {{ display:none !important; }}
[data-testid="stSpinner"] {{
  min-height: 0 !important;  /* avoid layout jump, since we center globally */
}}
{ "[data-testid='stSpinner']::after { content:''; position:fixed; inset:0; background:var(--st-overlay-bg); backdrop-filter: blur(var(--st-overlay-blur)); z-index: var(--st-overlay-z); pointer-events: none; }" if dim_backdrop else "" }
[data-testid="stSpinner"]::before {{
  content:"";
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%,-50%);
  width: var(--st-spin-inline);
  height: var(--st-spin-inline);
  background:no-repeat center/contain;
  animation:st-fourframe var(--st-spin-dur) linear infinite;
  z-index: calc(var(--st-overlay-z) + 1);
}}

/* Center the spinner message below the animation (works in sidebar or main) */
[data-testid="stSpinner"] [data-testid="stSpinnerMessage"],
[data-testid="stSpinner"] > div > div:last-child,
[data-testid="stSpinner"] > div > div:only-child {{
  position: fixed !important;
  left: 50% !important;
  top: calc(50% + var(--st-spin-inline) / 2 + 12px) !important;
  transform: translateX(-50%) !important;
  z-index: calc(var(--st-overlay-z) + 2) !important;
  text-align: center !important;
  margin: 0 !important;
  padding: .25rem .75rem !important;
  max-width: min(80vw, 900px) !important;  /* keeps long text from stretching off-screen */
  white-space: normal !important;          /* use `nowrap` if you prefer single-line */
  font-weight: 500 !important;
}}

/* Kill the tiny default glyph wrapper so you don't get a stray dot in the sidebar */
[data-testid="stSpinner"] > div > div:first-child {{
  display: none !important; 
}}

/* We still hide the default SVG everywhere */
[data-testid="stSpinner"] svg {{
  display: none !important; 
}}

/* =======================================================================
   3) st.status(...) icon — centered overlay
   ======================================================================= */
[data-testid="stStatus"] [data-testid="stStatusIcon"] svg,
[data-testid="stStatus"] [data-testid="stStatusIcon"] img {{ display:none !important; }}
{"[data-testid='stStatus']::after { content:''; position:fixed; inset:0; background:var(--st-overlay-bg); backdrop-filter: blur(var(--st-overlay-blur)); z-index: var(--st-overlay-z); pointer-events: none; }" if dim_backdrop else ""}
[data-testid="stStatus"] [data-testid="stStatusIcon"]::before {{
  content:"";
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%,-50%);
  width: var(--st-spin-inline);             /* use same size as inline */
  height: var(--st-spin-inline);
  background:no-repeat center/contain;
  animation:st-fourframe var(--st-spin-dur) linear infinite;
  z-index: calc(var(--st-overlay-z) + 1);
}}

/* Optional: allow 'esc' feel without blocking clicks — achieved via pointer-events:none above. */
</style>
    """, unsafe_allow_html=True)




