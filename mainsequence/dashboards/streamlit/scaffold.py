from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import streamlit as st

from mainsequence.dashboards.streamlit.core.registry import get_page, list_pages
from mainsequence.dashboards.streamlit.core.theme import inject_css_for_dark_accents
from importlib.resources import files as _pkg_files
def _bootstrap_theme_from_package(
    package: str = "mainsequence.dashboards.streamlit",
    resource: str = "assets/config.toml",
) -> None:
    """If $CWD/.streamlit/config.toml is missing, copy it from the package once."""
    try:
        src = _pkg_files(package) / resource
        default_toml = src.read_text(encoding="utf-8")
    except Exception:
        return  # no packaged theme; nothing to do

    cfg_dir = Path.cwd() / ".streamlit"
    cfg_file = cfg_dir / "config.toml"
    if not cfg_file.exists():
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(default_toml, encoding="utf-8")
        # Avoid infinite loop: only rerun once
        if not st.session_state.get("_ms_theme_bootstrapped"):
            st.session_state["_ms_theme_bootstrapped"] = True
            st.rerun()

# --- App configuration contract (provided by the example app) -----------------

HeaderFn      = Callable[[Any], None]
RouteFn       = Callable[[Mapping[str, Any]], str]
ContextFn     = Callable[[MutableMapping[str, Any]], Any]
InitSessionFn = Callable[[MutableMapping[str, Any]], None]
NotFoundFn    = Callable[[], None]

@dataclass
class AppConfig:
    title: str
    build_context: ContextFn                              # required
    route_selector: Optional[RouteFn] = None              # default: first visible page
    render_header: Optional[HeaderFn] = None              # if None, minimal header
    init_session: Optional[InitSessionFn] = None          # set defaults in session_state
    on_not_found: Optional[NotFoundFn] = None             # optional 404 renderer

    # Optional overrides; if None, scaffold uses its bundled defaults.
    logo_path: Optional[Union[str, Path]] = None
    page_icon_path: Optional[Union[str, Path]] = None

    use_wide_layout: bool = True
    hide_streamlit_multipage_nav: bool = True
    inject_theme_css: bool = True
    default_page: Optional[str] = None                    # fallback slug

# --- Internal helpers ---------------------------------------------------------

_HIDE_NATIVE_NAV = """
<style>[data-testid='stSidebarNav']{display:none!important}</style>
"""

def _hide_sidebar() -> None:
    st.markdown("""
        <style>
          [data-testid="stSidebar"]{display:none!important;}
          [data-testid="stSidebarCollapseControl"]{display:none!important;}
        </style>
    """, unsafe_allow_html=True)

def _minimal_header(title: str) -> None:
    st.title(title)

def _resolve_assets(explicit_logo: Optional[Union[str, Path]],
                    explicit_icon: Optional[Union[str, Path]]) -> Tuple[Optional[str], Union[str, None], Optional[str]]:
    """
    Returns a tuple:
      (logo_path_for_st_logo, page_icon_for_set_page_config, icon_path_for_st_logo_param)

    - If no overrides are provided, uses scaffold defaults:
         mainsequence.dashboards.streamlit/assets/logo.png
        mainsequence.dashboards.streamlit/assets/favicon.png
    - If favicon file is missing, falls back to emoji "📊" for set_page_config.
    - st.logo() will only receive icon_image if a real file exists.
    """
    base_assets = Path(__file__).resolve().parent / "assets"
    default_logo = base_assets / "logo.png"
    default_favicon = base_assets / "favicon.png"

    # Pick explicit override or default paths
    logo_path = Path(explicit_logo) if explicit_logo else default_logo
    icon_path = Path(explicit_icon) if explicit_icon else default_favicon

    # Effective values
    logo_for_logo_api: Optional[str] = str(logo_path) if logo_path.exists() else None
    icon_for_page_config: Union[str, None]
    icon_for_logo_param: Optional[str]

    if icon_path.exists():
        icon_for_page_config = str(icon_path)
        icon_for_logo_param = str(icon_path)
    else:
        # Streamlit allows emoji for set_page_config, but st.logo needs a file path.
        icon_for_page_config = "📊"
        icon_for_logo_param = None

    return logo_for_logo_api, icon_for_page_config, icon_for_logo_param

# --- Public entrypoint --------------------------------------------------------

def run_app(cfg: AppConfig) -> None:
    """Run a Streamlit app using the base scaffold."""
    _bootstrap_theme_from_package
    # Resolve assets (defaults shipped with the scaffold)
    _logo, _page_icon, _icon_for_logo = _resolve_assets(cfg.logo_path, cfg.page_icon_path)

    st.set_page_config(
        page_title=cfg.title,
        page_icon=_page_icon,  # can be a path or an emoji fallback
        layout="wide" if cfg.use_wide_layout else "centered"
    )

    if _logo:
        # Only pass icon_image if we have a real file for it
        st.logo(_logo, icon_image=_icon_for_logo)

    if cfg.inject_theme_css:
        inject_css_for_dark_accents()

    if cfg.hide_streamlit_multipage_nav:
        st.markdown(_HIDE_NATIVE_NAV, unsafe_allow_html=True)

    # Allow example app to seed session defaults (e.g., cfg_path)
    if cfg.init_session:
        cfg.init_session(st.session_state)

    # Decide the target route
    qp = st.query_params
    visible_pages = list_pages(visible_only=True)
    default_slug = cfg.default_page or (visible_pages[0].slug if visible_pages else None)
    target_slug = cfg.route_selector(qp) if cfg.route_selector else default_slug

    # Fallback if route not found or missing
    page = get_page(target_slug) if target_slug else None
    if page is None:
        if cfg.on_not_found:
            cfg.on_not_found()
        else:
            st.error("Page not found.")
        return

    # If this page doesn't own a sidebar, hide it (container + burger)
    if not page.has_sidebar:
        _hide_sidebar()

    # Build a UI-free context (example app decides what "context" means)
    ctx = cfg.build_context(st.session_state)

    # Header (example can render a rich caption with context)
    if cfg.render_header:
        cfg.render_header(ctx)
    else:
        _minimal_header(cfg.title)

    # Route to the page
    page.render(ctx)
