from __future__ import annotations
import streamlit as st

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


from .base64_imgs import *


# In streamlit/core/theme.py


import streamlit as st


def override_spinners(
    hide_deploy_button: bool = False,
    *,
    # Sizes
    top_px: int = 35,          # top-right toolbar & st.status icon size
    inline_px: int = 96,       # in-page st.spinner size
    # Timing
    duration_ms: int = 900,
    # Vertical nudges / margins
    toolbar_nudge_px: int = -3,     # push the top-right icon down (+ is lower)
    status_icon_nudge_px: int = 0, # nudge the st.status icon down
    inline_top_margin_px: int = 0, # extra top margin for the inline spinner
toolbar_gap_left_px: int = 2,   # space between spinner and "Stop"
    toolbar_left_offset_px: int = 0 # move the spinner itself left/right

) -> None:
    def as_data_uri(s: str, mime="image/png") -> str:
        s = s.strip()
        return s if s.startswith("data:") else f"data:{mime};base64,{s}"

    # your 4 PNGs as base64 or data: URIs
    i1 = as_data_uri(IMAGE_1_B64)
    i2 = as_data_uri(IMAGE_2_B64)
    i3 = as_data_uri(IMAGE_3_B64)
    i4 = as_data_uri(IMAGE_4_B64)

    st.markdown(f"""
<style>
@keyframes st-fourframe {{
  0%,100% {{ background-image:url("{i1}"); }}
  25%      {{ background-image:url("{i2}"); }}
  50%      {{ background-image:url("{i3}"); }}
  75%      {{ background-image:url("{i4}"); }}
}}

:root {{
  --st-spin-top:{top_px}px;
  --st-spin-inline:{inline_px}px;
  --st-spin-dur:{duration_ms}ms;
  --st-spin-toolbar-nudge:{toolbar_nudge_px}px;
  --st-spin-status-nudge:{status_icon_nudge_px}px;
  --st-spin-inline-mt:{inline_top_margin_px}px;
 --st-spin-toolbar-gap:{toolbar_gap_left_px}px;
  --st-spin-toolbar-left:{toolbar_left_offset_px}px;
}}

/* ---------- st.spinner (inline) ---------- */
[data-testid="stSpinner"] svg,
[data-testid="stSpinner"] > div {{ display:none !important; }}
[data-testid="stSpinner"] {{
  position:relative;
  min-height:calc(var(--st-spin-inline) + var(--st-spin-inline-mt) + 24px);
}}
[data-testid="stSpinner"]::before {{
  content:"";
  display:block;
  width:var(--st-spin-inline);
  height:var(--st-spin-inline);
  margin:var(--st-spin-inline-mt) auto 20px auto; /* real top margin */
  background:no-repeat center/contain;
  animation:st-fourframe var(--st-spin-dur) linear infinite;
}}

/* ---------- st.status(...) icon ---------- */
[data-testid="stStatus"] [data-testid="stStatusIcon"] svg,
[data-testid="stStatus"] [data-testid="stStatusIcon"] img {{ display:none !important; }}
[data-testid="stStatus"] [data-testid="stStatusIcon"]::before {{
  content:"";
  position:relative;
  display:inline-block;
  width:var(--st-spin-top);
  height:var(--st-spin-top);
  margin-right:.25rem;
  transform:translateY(var(--st-spin-status-nudge));  /* nudge down */
  background:no-repeat center/contain;
  animation:st-fourframe var(--st-spin-dur) linear infinite;
}}

/* ---------- Top-right toolbar widget ---------- */
/* Reserve space for our absolutely-positioned icon so text doesn't overlap */
[data-testid="stStatusWidget"] {{
  position:relative;
  padding-left: calc(var(--st-spin-top) + var(--st-spin-toolbar-gap));
}}
/* Hide any built-in icons but keep labels ("Stop", "Deploy") visible */
[data-testid="stStatusWidget"] svg,
[data-testid="stStatusWidget"] img {{ display:none !important; }}

/* Place our icon absolutely and center it vertically; then nudge down */
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

{"div[data-testid='stToolbar']{display:none !important;}" if hide_deploy_button else ""}
</style>
    """, unsafe_allow_html=True)





