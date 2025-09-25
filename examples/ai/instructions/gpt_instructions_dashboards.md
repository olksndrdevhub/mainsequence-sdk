# MainSequence Streamlit Dashboards — Authoring Guide (Scaffold + Examples)

> **GOLDEN RULES (must‑follow)**
>
> 1) **Always integrate with the MainSequence platform**. Import the client as:
>    ```python
>    import mainsequence.client as msc
>    ```
> 2) **Always register required `data_nodes`** once per session (idempotent) **before** any portfolio/asset/trade operations.
> 3) **Always use the provided Streamlit scaffold** (`run_page(PageConfig(...))`) at the top of *every* page.

This guide shows how to create new dashboards that stay consistent with `mainsequence.dashboards` scaffolding and remain fully integrated with the MainSequence platform (assets, portfolios, accounts, trades, etc.).

---

## 0) What you get from the scaffold

- `mainsequence.dashboards.streamlit.scaffold`:
  - `PageConfig` — declarative page config.
  - `run_page(cfg)` — sets page config, injects theme/CSS, initializes session, builds context, and renders header.
  - Automatic first‑run bootstrap of `.streamlit/config.toml` (theme) next to your app.
  - Helper CSS (`inject_css_for_dark_accents`, `override_spinners`) to keep a unified look & feel.

- Theme defaults (dark) shipped at `mainsequence.dashboards.streamlit/assets/config.toml`.

- Opinionated UX helpers in the example app:
  - **Engine HUD** (`dashboards/components/engine_status.py`)
  - **Portfolio/Asset selectors** (`dashboards/components/portfolio_select.py`, `dashboards/components/asset_select.py`)
  - **Curve bump controls** (`dashboards/components/curve_bump.py`)
  - **Date selector** (`dashboards/components/date_selector.py`)
  - **Position yield overlay** (`dashboards/components/position_yield_overlay.py`)
  - **Position JSON I/O** (`dashboards/components/positions_io.py`)
  - **Mermaid graph renderer** (see *Data Nodes — Dependencies* page)

---

## 1) Project layout (recommended)

Use Streamlit’s single‑app + multipage structure:

```shell
your_dashboard/
app.py
pages/
01_<feature_a>.py
02_<feature_b>.py
99_<utility_pages>.py
```


You **do not** need to copy the scaffold files; import them from the package:
```python
from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
```

## 2) Bootstrapping a new app

Create app.py with the scaffold and mandatory MainSequence integration:

```python
# app.py
from __future__ import annotations
import streamlit as st
import mainsequence.client as msc  # ← ALWAYS import the MS client as `msc`

from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
from dashboards.core.data_nodes import get_app_data_nodes  # your app helper (provide one if missing)

def _ensure_data_nodes_once() -> None:
    """Register all external dependency names this app expects (idempotent)."""
    if st.session_state.get("_deps_bootstrapped_root"):
        return
    deps = get_app_data_nodes()
    # Example registrations — adjust to your app needs
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        deps.register(instrument_pricing_table_id="vector_de_precios_valmer")
    # Add more app-level nodes here (examples):
    # deps.register(positions_table_id="positions_current")
    # deps.register(trades_table_id="trades_live")
    st.session_state["_deps_bootstrapped_root"] = True

cfg = PageConfig(
    title="My MS Dashboard",
    hide_streamlit_multipage_nav=False,
    use_wide_layout=True,
    inject_theme_css=True,
)

# Initialize page and theme; returns a context dict if you supply build_context
ctx = run_page(cfg)

# Ensure data_nodes are registered for the whole app
_ensure_data_nodes_once()

st.markdown("> Welcome! Use the left sidebar to navigate pages.")



```
Why this matters

msc import ensures you can call platform models (e.g., msc.Asset, msc.PortfolioIndexAsset, msc.Trade, msc.Account, etc.).

_ensure_data_nodes_once() ensures downstream components (pricing/curves/positions) have the identifiers they need.

## 3) Making a new page (template)

Every page must start with run_page(PageConfig(...)) and must ensure data nodes are registered before querying the platform.

```python
# pages/01_Portfolio_Overview.py
from __future__ import annotations
import streamlit as st
import mainsequence.client as msc  # ← REQUIRED
from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
from dashboards.core.data_nodes import get_app_data_nodes

# 1) Standard page boot
ctx = run_page(PageConfig(
    title="Portfolio Overview",
    use_wide_layout=True,
    inject_theme_css=True,
))

# 2) Ensure data_nodes (safe if repeated)
def _ensure_data_nodes() -> None:
    if st.session_state.get("_deps_bootstrapped_01"):
        return
    deps = get_app_data_nodes()
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        deps.register(instrument_pricing_table_id="vector_de_precios_valmer")
    st.session_state["_deps_bootstrapped_01"] = True

_ensure_data_nodes()

# 3) Use the MS client (msc) to query platform objects
st.sidebar.text_input("Search portfolios", key="q_port")
q = (st.session_state.get("q_port") or "").strip()
if len(q) >= 3:
    with st.spinner("Searching portfolios..."):
        results = msc.PortfolioIndexAsset.filter(current_snapshot__name__contains=q)
else:
    results = []

st.title("Portfolio Overview")
if not results:
    st.info("Type at least 3 characters to search portfolios.")
else:
    # Render basic list
    for p in results:
        name = getattr(getattr(p, "current_snapshot", None), "name", "—")
        st.write(f"- **{name}** (id={getattr(p,'id','?')})")

```

Checklist for every new page

 * import mainsequence.client as msc

 * Call run_page(PageConfig(...)) first

 * Ensure data_nodes registered (idempotent)

 * Use cached queries (@st.cache_data) for expensive platform calls when appropriate

 * Handle errors gracefully (.get_or_none, try/except, user messages)

## 4) PageConfig contract (what you can customize)

* title: str — page window title and default header.

* build_context: Optional[Callable[[MutableMapping], Any]] — build a context object from st.session_state (optional).

* render_header: Optional[Callable[[Any], None]] — custom header renderer if you want more than a title.

* init_session: Optional[Callable[[MutableMapping], None]] — set default values in session_state.

* logo_path, page_icon_path — override default brand assets.

* use_wide_layout: bool — default True.

* hide_streamlit_multipage_nav: bool — hide the native sidebar nav if you render your own.

* inject_theme_css: bool — whether to apply small accent tweaks (theme itself comes from .streamlit/config.toml).

The scaffold auto‑creates .streamlit/config.toml on first run if missing and triggers one safe rerun to apply the theme.

## 5) Reusing the example app patterns
### A) Data Nodes — Dependencies (Mermaid)

Use a data → Mermaid text → HTML/JS iframe pattern:

* Build Mermaid text with sanitized node IDs and styles.

* Render via components.html(...) importing Mermaid ESM:

  * securityLevel: 'loose' to enable click callbacks.

* Provide a click handler to show node details in a modal.

Use case: show lineage across local nodes and remote API nodes (e.g., API_26, API_41).

### B) Curve, Stats & Positions

* Sidebar:

Valuation date (date_selector)

Portfolio search/select (sidebar_portfolio_multi_select)

Curve bumps per family (curve_bump_controls_ex)

* Core flow:

Ensure data nodes (e.g., instrument_pricing_table_id) exist.

Build curves (base & bumped), compute NPVs and carry.

Overlay position YTMs onto par yield curves with Plotly.

Paginated positions table via PortfoliosOperations.

## C) Asset Detail

Accept query params: id or unique_identifier (exclusively one).

Fetch with msc.Asset.get_or_none(...).

Rebuild instrument (mainsequence.instruments) from pricing_detail.instrument_dump to compute cashflows.

Offer CSV download.

## 6) Platform access with msc (cookbook)
# Find assets by name / UID
assets_by_name = msc.Asset.filter(current_snapshot__name__contains="BONOS")
assets_by_uid  = msc.Asset.filter(unique_identifier__contains="MXN:")

# Load a single asset or portfolio safely
asset = msc.Asset.get_or_none(id=1234)
port  = msc.PortfolioIndexAsset.get_or_none(id=5678)

# Trades & accounts (examples; adapt filters to your schema)
recent_trades = msc.Trade.filter(executed_at__gte="2025-01-01")
accounts      = msc.Account.filter(name__contains="ALM")

# Defensive accessors for optional nested fields
uid = getattr(asset, "unique_identifier", None)
snap_name = getattr(getattr(asset, "current_snapshot", None), "name", None)

Caching tip

```python
import streamlit as st

@st.cache_data(show_spinner=False)
def search_portfolios(q: str):
    return msc.PortfolioIndexAsset.filter(current_snapshot__name__contains=q) if len(q)>=3 else []
```

## 7) Registering data_nodes (always do this)

Use your get_app_data_nodes() helper (or provide a similar registry) and register the identifiers your app expects.

```python
def ensure_nodes(flag_key: str = "_deps_bootstrapped_any") -> None:
    import streamlit as st
    from dashboards.core.data_nodes import get_app_data_nodes
    if st.session_state.get(flag_key): return
    deps = get_app_data_nodes()

    # Example registrations (add your own):
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        deps.register(instrument_pricing_table_id="vector_de_precios_valmer")

    # If your pages use trades, accounts, or positions tables, register those too:
    # deps.register(trades_table_id="trades_live")
    # deps.register(positions_table_id="positions_current")
    # deps.register(accounts_table_id="accounts_master")

    st.session_state[flag_key] = True

```

## 8) Ready‑made components you can drop in

Portfolio picker (sidebar):
```
from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
selected_instances = sidebar_portfolio_multi_select(min_chars=3, key_prefix="my_port")
```

Asset picker (sidebar):
```python


from dashboards.components.asset_select import sidebar_asset_single_select
asset = sidebar_asset_single_select(min_chars=3, key_prefix="my_asset")
```

Date selector (sticky valuation date in session):
```python


from dashboards.components.date_selector import date_selector
val_date = date_selector(label="Valuation date", session_cfg_key="cfg", cfg_field="valuation_date")
```

Curve bumps (family‑aware):
```python

from dashboards.components.curve_bump import curve_bump_controls_ex
spec, changed = curve_bump_controls_ex(available_tenors=["1Y","2Y","5Y"], key="kr")

```

Pricing Engine HUD:

```python


from dashboards.components.engine_status import render_engine_status, publish_engine_meta_summary
publish_engine_meta_summary(reference_date=val_date, currency="MXN")
render_engine_status(meta={}, title="Pricing engine", mode="sticky_bar", open=False)
```

Position overlay (Plotly traces only):

```python

from dashboards.components.position_yield_overlay import st_position_yield_overlay
traces = st_position_yield_overlay(position=position, valuation_date=val_date)
# add to your Plotly figure: for tr in traces: fig.add_trace(tr)

```

9) Minimal “Position from Portfolio” page (end‑to‑end)
```python

# pages/02_Position_From_Portfolio.py
from __future__ import annotations
import datetime as dt
import streamlit as st
import QuantLib as ql
import mainsequence.client as msc

from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
from dashboards.components.date_selector import date_selector
from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from dashboards.services.portfolios import PortfoliosOperations
from dashboards.services.positions import PositionOperations
from dashboards.services.curves import build_curves_for_ui, keyrate_grid_for_index, curve_family_key, BumpSpec
from dashboards.plots.curves import plot_par_yield_curves_by_family
from dashboards.core.ql import qld
from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.components.curve_bump import curve_bump_controls_ex
from dashboards.core.formatters import fmt_ccy

# 1) Boot page
ctx = run_page(PageConfig(title="Position from Portfolio"))

# 2) Ensure data_nodes
def _ensure_nodes():
    if st.session_state.get("_deps_bootstrapped_p2"): return
    deps = get_app_data_nodes()
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        deps.register(instrument_pricing_table_id="vector_de_precios_valmer")
    st.session_state["_deps_bootstrapped_p2"] = True
_ensure_nodes()

# 3) Sidebar: valuation date + portfolio search + curve bumps
val_date = date_selector(label="Valuation date", session_cfg_key="pos_cfg", cfg_field="valuation_date")
selected = sidebar_portfolio_multi_select(min_chars=3, key_prefix="pos_from_port")
st.sidebar.markdown("---")
fam_spec, _ = curve_bump_controls_ex(available_tenors=["1Y","2Y","3Y","5Y","7Y","10Y"], key="fam_bumps")

if not selected:
    st.info("Pick a portfolio in the sidebar.")
    st.stop()

active_port = selected[0].reference_portfolio
po = PortfoliosOperations(portfolio_list=[active_port])
notional = float(st.session_state.get("portfolio_notional", 1_000_000.0))

with st.spinner("Building position…"):
    _, pos_map = po.get_all_portfolios_as_positions(portfolio_notional=notional)
position_template = list(pos_map.values())[0]

# 4) Curves (toy build via available indices in position)
def _indices_from_position(p):
    return sorted({getattr(ln.instrument, "floating_rate_index_name", None)
                   for ln in (p.lines or []) if getattr(ln.instrument, "floating_rate_index_name", None)})

indices = _indices_from_position(position_template)
base_curves = {}
bump_curves = {}
for idx in indices:
    fam = curve_family_key(idx)
    kr = fam_spec.get(fam, {}).get("keyrate_bp", {})
    par = float(fam_spec.get(fam, {}).get("parallel_bp", 0.0))
    spec = BumpSpec(keyrate_bp=kr, parallel_bp=par, key_rate_grid={idx: tuple(keyrate_grid_for_index(idx))})
    ts_base, ts_bump, _, _ = build_curves_for_ui(qld(val_date), spec, index_identifier=idx)
    base_curves[idx] = ts_base; bump_curves[idx] = ts_bump

# 5) Instantiate positions and compute stats
ql.Settings.instance().evaluationDate = qld(val_date)
ops = PositionOperations.from_position(position_template, base_curves_by_index=base_curves, valuation_date=val_date)
base_pos = ops.instantiate_base()
ops.set_curves(bumped_curves_by_index=bump_curves)
bumped_pos = ops.instantiate_bumped()
ops.compute_and_apply_z_spreads_from_dirty_price(base_position=base_pos, bumped_position=bumped_pos)

# 6) Plot + overlay
fig = plot_par_yield_curves_by_family(base_curves, bump_curves, max_years=30, step_months=3,
                                      title="Par yield curves — Base vs Bumped")
for tr in st_position_yield_overlay(position=base_pos, valuation_date=val_date): fig.add_trace(tr)
st.plotly_chart(fig, width="stretch")

# 7) Quick stats
stats = PositionOperations.portfolio_style_stats(base_position=base_pos, bumped_position=bumped_pos,
                                                 valuation_date=val_date, cutoff=val_date + dt.timedelta(days=365))
c1, c2, c3 = st.columns(3)
c1.metric("NPV (base)", fmt_ccy(stats["npv_base"]), delta=fmt_ccy(stats["npv_delta"]))
c2.metric("Carry (base)", fmt_ccy(stats["carry_base"]), delta=fmt_ccy(stats["carry_delta"]))
c3.metric("NPV (bumped)", fmt_ccy(stats["npv_bumped"]))

```

## 10) Data Nodes — Dependencies page (copy‑ready)
Use this when you want an interactive lineage diagram with click‑to‑details. Adapt the stub graph to your real nodes/edges payload from MS.

See example: pages/02_data_nodes_dependencies.py

Key ideas you can reuse:

Sanitize node IDs for Mermaid.

Determine label and color per node; compute legible text color.

For clicks, expose window._mmdNodeClick(id) and open a modal.

## 11) Useful UX touches from the scaffold

Accents CSS: inject_css_for_dark_accents() gives subtle visual polish for metrics.

Spinner override: override_spinners() replaces default spinners with a clean inline animation, dims backdrop during long tasks.

Hide native nav: Set hide_streamlit_multipage_nav=True if you render your own navigation.

12) Prompt recipe so an AI helper can generate new pages correctly

Copy this into your AI assistant as instructions it must always follow when generating code:

Always start each Streamlit file with:

import mainsequence.client as msc
from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
ctx = run_page(PageConfig(title="<PAGE TITLE>", use_wide_layout=True, inject_theme_css=True))


Always ensure and idempotently register app data_nodes before querying the platform:
```python


from dashboards.core.data_nodes import get_app_data_nodes
import streamlit as st
if not st.session_state.get("_deps_bootstrapped"):
    deps = get_app_data_nodes()
    try: deps.get("instrument_pricing_table_id")
    except KeyError: deps.register(instrument_pricing_table_id="vector_de_precios_valmer")
    # register any other tables your page needs here
    st.session_state["_deps_bootstrapped"] = True

```


Always query platform objects through msc (e.g., msc.Asset, msc.PortfolioIndexAsset, msc.Trade, msc.Account).

Prefer cached queries for searches (@st.cache_data(show_spinner=False)).

Never hard‑code secrets/URLs; rely on the configured MainSequence client.

Name pages with numeric prefixes: 01_*.py, 02_*.py, etc., and keep page code self‑contained.

When plotting, reuse provided components (curve bump, overlay, positions table) and keep valuation date in st.session_state.

Gracefully handle missing data with get_or_none, st.info, st.warning, and download buttons for raw payloads.

Keep UI responsive with with st.spinner("…"): and small status chips (Engine HUD if available).

Checklist the AI should ask/confirm (or default)

Page title and goal

Which MS objects to retrieve (assets, portfolios, trades, accounts)

Which data_nodes to register (list of keys → identifiers)

Whether curves/pricing are needed (and for which indices/families)

Desired outputs (tables, charts, downloads)


## 14) Do’s & Don’ts

✅ Do import mainsequence.client as msc in every page/module that touches platform data.

✅ Do register data_nodes once per session and before any pricing/curve/portfolio operations.

✅ Do call run_page(PageConfig(...)) at the top of every page.

✅ Do guard platform calls and show actionable messages to users.

❌ Don’t bypass the scaffold (no st.set_page_config directly unless inside the scaffold).

❌ Don’t assume data nodes exist — register them explicitly and idempotently.

❌ Don’t rely on unstored UI state for valuation date or selections; persist in st.session_state.

## 15) Troubleshooting

“Asset/portfolio not found”: Use .get_or_none and surface the ID/UID you searched for. Confirm you’re registered against the right snapshot/table nodes.

“pricing_detail.instrument_dump missing”: Warn the user and short‑circuit cashflow rebuild; offer raw JSON download for support.

Mermaid graph too small: Increase the iframe height or compute it from len(nodes).

Theme not applied: Remove existing .streamlit/config.toml or set MS_APP_DIR and rerun so the scaffold can copy the packaged theme once.