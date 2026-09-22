"""PRISM: Profiling & Remote-sensing Interactive Spectra Monitor.

Interactive viewer for ground-based remote-sensing time-height products and
raw Doppler spectra, currently sourced from the Cloudnet data portal.

Run with:  panel serve src/prism/app.py --show
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn
import param
from bokeh.models import (
    AllLabels, CustomJSHover, CustomJSTickFormatter, FixedTicker, LinearAxis, LinearScale, Range1d,
)
from holoviews.plotting.util import process_cmap

from prism import cloudnet_client as cc
from prism import gridded_products as gp
from prism import mira_reader as mr
from prism import plot_meta as pm
from prism import rpg_reader as rr
from prism import settings_store as ss

pn.extension(raw_css=["""
    /* A stretch_both/percentage-sized Panel/Bokeh layout only has something
    real to stretch INTO if its ancestors actually span the viewport --
    the default HTML body is only as tall as its content, so without this
    the grid's sizing_mode="stretch_both" (see build_app) has no effect and
    it just falls back to a fixed default height. Deliberately scoped to
    just html/body, NOT a blanket rule on every Bokeh row/column -- those
    stay sized by their own explicit sizing_mode in Python (the top/bottom
    control bars are meant to stay their natural height, only the grid
    itself should claim the rest).
    */
    html, body { height: 100%; margin: 0; }
"""])
hv.extension("bokeh")
# Radar/height grids are piecewise- rather than perfectly-uniform (chirp
# boundaries, instrument-specific gate spacing). We deliberately render as a
# raster (hv.Image) rather than per-cell QuadMesh for speed, which needs a
# looser uniformity tolerance than the default.
hv.config.image_rtol = 0.5

N_MOMENT_PANELS = 3
DEFAULT_PANEL_VARIABLES = ["categorize:Z", "categorize:v", "categorize:width"]


def to_datetime64(x) -> np.datetime64:
    """Normalize a HoloViews Tap x-coordinate on a datetime axis, which can
    arrive as np.datetime64, datetime.datetime, or ms-since-epoch float
    depending on HoloViews/Bokeh version."""
    if isinstance(x, np.datetime64):
        return x
    if isinstance(x, dt.datetime):
        return np.datetime64(x)
    return np.datetime64(int(x), "ms")


def unix_seconds(t: np.datetime64) -> float:
    return (t - np.datetime64(0, "s")) / np.timedelta64(1, "s")


class AppState(param.Parameterized):
    site = param.String(default="hyytiala")
    day = param.Date(default=dt.date(2026, 2, 11))
    instrument = param.Selector(default="rpg-fmcw-94", objects=cc.RAW_SPECTRA_INSTRUMENTS)
    hour_index = param.Integer(default=8)  # UTC hour of day, 0-23
    selected_time = param.Number(default=None, allow_None=True)  # unix seconds
    selected_height = param.Number(default=None, allow_None=True)  # meters
    channel = param.Selector(default="co", objects=["co", "cross"])
    # Bumped once, explicitly, after a load has fully completed (data AND
    # variable selections in place). Every panel watches it, so bumping it
    # forces exactly one re-render in which _make_range_hook refits the
    # axes. Nothing else refits them, so clicks preserve the user's zoom.
    range_generation = param.Integer(default=0)

    def __init__(self, settings: ss.Settings, **params):
        super().__init__(**params)
        self.settings = settings
        cache_dir = settings.get("cache_dir")
        self.cache_dir = Path(cache_dir) if cache_dir else cc.DEFAULT_CACHE_DIR
        self.available_hours: list[cc.RemoteFile] = []
        self.catalog: list[gp.ProductVariable] = []
        self.spectra: rr.SpectraHour | None = None
        self.channel = settings.get("spectra_channel")
        # Which range_generation each Bokeh range object was last fitted for.
        self.range_keys: dict[str, int] = {}
        # Which (lo, hi) an auto_y range was last fit to, for _make_range_hook
        # to detect a genuine natural-extent change (e.g. RPG's chirp
        # boundary) vs. a plain re-render that shouldn't touch it again.
        self.auto_bounds: dict[str, tuple[float, float]] = {}
        # Per row-1 panel: the ONE FixedTicker/CustomJSTickFormatter pair its
        # colorbar is built with, kept so hooks can re-point them at the
        # current variable (see _panel_colorbar_models).
        self.colorbar_models: dict[int, tuple] = {}
        # Per row-1 panel: whether the image it last drew was log10-
        # transformed (see _moment_image). The hooks can't infer this from
        # the variable alone, because a manual color range with a
        # non-positive minimum forces a log-scale variable back to linear.
        self.panel_log: dict[int, bool] = {}
        last = settings.get("last_session")
        self.site = last["site"]
        self.day = dt.date.fromisoformat(last["day"])
        self.hour_index = last["hour"]
        # Older settings.json files predate the instrument selector.
        self.instrument = last.get("instrument", "rpg-fmcw-94")

    def refresh_site_day(self, on_progress=None, on_status=None):
        """Day-scoped: raw spectra file listing + moments/curtain-variable
        discovery. Cloudnet only publishes these as whole-day files, so this
        part can't be scoped to an hour -- but it's small (a listing call
        plus netCDFs on the order of MB, not the hundreds-of-MB spectra).
        Which single hour's spectra actually get downloaded/decoded is a
        separate, later step in load_hour()."""
        if on_status:
            on_status("Checking available hours and products...")
        self.available_hours = cc.list_raw_spectra_files(self.site, self.day, self.instrument)
        # Only download/scan the products the panels will actually show on
        # this Load: the defaults, plus whichever product each panel's OWN
        # persisted choice belongs to (so a returning user's saved variable
        # shows real data immediately instead of a stub). Every other
        # available product is listed but not fetched -- see
        # gp.discover()'s docstring and resolve_variable() below.
        eager = set(gp.DEFAULT_EAGER_PRODUCTS)
        for i in range(N_MOMENT_PANELS):
            stored_id = self.settings.get_panel(i)["variable"]
            if stored_id and ":" in stored_id:
                eager.add(stored_id.split(":", 1)[0])
        self.catalog = gp.discover(self.site, self.day, self.cache_dir,
                                    on_progress=on_progress, eager_products=eager)
        self.load_hour(on_progress=on_progress, on_status=on_status)

    def hours_with_data(self) -> set[int]:
        return {int(r.filename.split("_")[1][0:2]) for r in self.available_hours}

    def _remote_for_hour(self, hour: int) -> cc.RemoteFile | None:
        return next((r for r in self.available_hours if int(r.filename.split("_")[1][0:2]) == hour), None)

    def _decoder_for_instrument(self):
        return mr if self.instrument in ("mira-10", "mira-35") else rr

    def _ensure_hour_decoded(self, remote: cc.RemoteFile, on_progress=None, on_status=None) -> Path:
        """Downloads and decodes ONE hour's raw spectra file into its zarr
        cache, if not already there, and returns that cache path -- shared
        by load_hour() (the currently-selected hour) and
        cache_all_hours() (every other hour for this instrument/day). The
        raw file is deleted after a successful decode: only the much
        smaller processed zarr cache is kept, since ensure_decoded() never
        needs the raw file again once its _SUCCESS marker exists (checked
        here too, so a re-requested hour whose raw file was already
        deleted doesn't get re-downloaded just to satisfy this function's
        own plumbing). Decoding itself is dispatched by instrument --
        rpg_reader for RPG's LV0 binary format, mira_reader for METEK's
        znc/HDF5 format -- but both write the SAME zarr schema, so
        SpectraHour and everything downstream of it don't need to know or
        care which one ran."""
        decoder = self._decoder_for_instrument()
        local = cc.cache_path_for(remote, self.cache_dir)
        already_decoded = (decoder._zarr_cache_path(local, self.cache_dir) / "_SUCCESS").exists()
        if not already_decoded:
            if on_status:
                on_status(f"Downloading {remote.filename}...")
            local = cc.ensure_downloaded(remote, self.cache_dir, on_progress=on_progress)
            if on_status:
                on_status(f"Processing spectra ({self.instrument})...")
        zpath = decoder.ensure_decoded(local, self.cache_dir)
        if local.exists():
            local.unlink()
        return zpath

    def load_hour(self, on_progress=None, on_status=None):
        """Downloads and decodes exactly ONE hour's raw spectra file -- the
        one currently selected in hour_index -- never the whole day (see
        cache_all_hours() for that)."""
        remote = self._remote_for_hour(self.hour_index)
        if remote is None:
            self.spectra = None
            return
        zpath = self._ensure_hour_decoded(remote, on_progress=on_progress, on_status=on_status)
        self.spectra = rr.SpectraHour(zpath)
        mid = self.spectra.n_time // 2
        self.selected_time = float(self.spectra.time[mid])
        self.selected_height = float(self.spectra.height[
            self.spectra.nearest_range_index(float(np.median(self.spectra.height)))
        ])

    def cache_all_hours(self, on_progress=None, on_status=None) -> tuple[int, int]:
        """Downloads and decodes EVERY available hour's raw spectra for the
        current site/day/instrument, not just the selected one -- for the
        "Cache entire day" button. Never touches self.spectra/selected_time/
        selected_height, so it doesn't disturb whatever's currently on
        screen; each hour's zarr cache is simply left on disk for
        on_hour_change() to pick up instantly later. Returns (decoded,
        failed) counts -- a single hour's flaky/dropped download shouldn't
        abort caching the rest of the day, the same way discover() already
        tolerates one bad product file, especially since this loop is
        exactly the "bad connection" scenario most likely to hit one.

        Downloading hour i+1 starts in a background thread as soon as hour
        i's download finishes, so it runs WHILE hour i is being decoded
        (CPU-bound: reading + dB-quantizing every Doppler bin) instead of
        strictly after -- download is I/O-bound and barely touches the CPU
        the decode step needs, so by the time one hour's decode is done,
        the next one's file is often already sitting on disk ready to go,
        rather than only starting to download at that point.
        """
        decoder = self._decoder_for_instrument()
        pending = [r for r in self.available_hours
                   if not (decoder._zarr_cache_path(cc.cache_path_for(r, self.cache_dir), self.cache_dir)
                           / "_SUCCESS").exists()]
        if not pending:
            return 0, 0

        newly_decoded = 0
        failed = 0

        def _download(remote):
            return cc.ensure_downloaded(remote, self.cache_dir, on_progress=on_progress)

        with ThreadPoolExecutor(max_workers=1) as io_pool:
            next_future = io_pool.submit(_download, pending[0])
            for i, remote in enumerate(pending):
                if on_status:
                    on_status(f"Caching spectra: hour {i + 1}/{len(pending)} ({remote.filename})...")
                try:
                    local = next_future.result()
                except Exception:
                    failed += 1
                    local = None
                # Kick off the NEXT hour's download now, before this hour's
                # (CPU-bound) decode -- that's the actual overlap.
                if i + 1 < len(pending):
                    next_future = io_pool.submit(_download, pending[i + 1])
                if local is None:
                    continue
                try:
                    if on_status:
                        on_status(f"Processing spectra ({self.instrument}), hour {i + 1}/{len(pending)}...")
                    decoder.ensure_decoded(local, self.cache_dir)
                    if local.exists():
                        local.unlink()
                    newly_decoded += 1
                except Exception:
                    failed += 1
        return newly_decoded, failed

    def cache_all_products(self, on_progress=None, on_status=None) -> int:
        """Resolves every still-lazy (stub) product in the catalog -- for
        the "Cache entire day" button. Returns how many (product,
        instrument) pairs were actually resolved (vs. already real)."""
        pending = {(pv.product_id, pv.instrument_id): pv
                   for pv in self.catalog if pv.file_path is None}
        for n, pv in enumerate(pending.values()):
            if on_status:
                on_status(f"Caching products: {n + 1}/{len(pending)} ({pv.product_id}, "
                          f"{pv.instrument_id})...")
            self.catalog = gp.resolve_product(pv, self.catalog, self.cache_dir, on_progress=on_progress)
        return len(pending)

    def catalog_variable(self, catalog_id: str) -> gp.ProductVariable | None:
        return next((p for p in self.catalog if p.catalog_id == catalog_id), None)

    def resolve_variable(self, catalog_id: str | None) -> gp.ProductVariable | None:
        """Like catalog_variable(), but downloads and scans that variable's
        product FIRST if discover() only offered it as a stub (see
        gp.DEFAULT_EAGER_PRODUCTS / gp.resolve_product). Only the actual
        data-loading render functions (_moment_image/_moment_curve) need
        this -- everything else (colorbar/hover styling, categorical/log
        flags) only reads metadata that's already correct on a stub, so
        calling this there would just be a needless download trigger."""
        pv = self.catalog_variable(catalog_id)
        if pv is None or pv.file_path is not None:
            return pv
        self.catalog = gp.resolve_product(pv, self.catalog, self.cache_dir)
        return self.catalog_variable(catalog_id)

    @property
    def hour_time_bounds(self):
        if self.spectra is None:
            return None, None
        return (np.datetime64(int(self.spectra.time[0]), "s"),
                np.datetime64(int(self.spectra.time[-1]), "s"))

    @property
    def display_time_bounds(self):
        """Same as hour_time_bounds, but falls back to a placeholder window
        anchored at the currently SELECTED hour (not day-start) when no hour
        is loaded yet, so the very first (pre-Load) DynamicMap frame anchors
        almost exactly where real data will eventually land -- minimizing
        the placeholder/real mismatch regardless of which hour is picked."""
        t0, t1 = self.hour_time_bounds
        if t0 is None:
            hour_start = np.datetime64(self.day) + np.timedelta64(self.hour_index, "h")
            t0, t1 = hour_start, hour_start + np.timedelta64(1, "h")
        return t0, t1


def _to_axis_value(v):
    """Bokeh datetime axes are numeric underneath: milliseconds since epoch."""
    if isinstance(v, np.datetime64):
        return float(v.astype("datetime64[ms]").astype("int64"))
    if isinstance(v, dt.datetime):
        return v.timestamp() * 1000.0
    return float(v)


def _element_bounds(element, dim_name):
    """Full data extent of one dimension, as Bokeh axis coordinates."""
    try:
        lo, hi = element.range(dim_name)
    except Exception:
        return None
    if lo is None or hi is None:
        return None
    try:
        lo, hi = _to_axis_value(lo), _to_axis_value(hi)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _make_range_hook(state: AppState, x_dim: str | None, y_dim: str | None, auto_y: bool = False):
    """Take axis-range control away from HoloViews and do it ourselves.

    HoloViews recomputes shared axis ranges on EVERY DynamicMap update, so
    any click (which re-renders the spectrogram panels) snapped the shared
    time/height axes back to full extent, throwing away the user's zoom.
    Per-element options (framewise, xlim/ylim) can't prevent that -- the
    recomputation happens at the layout level. So the panels set
    apply_ranges=False (making HoloViews' own range updates a no-op) and
    this hook sets the ranges instead, keyed on state.range_generation: a
    range is only refitted when that is explicitly bumped after a completed
    load, so clicks leave the current view untouched.

    auto_y=True opts the y axis out of the generation rule, for axes whose
    NATURAL extent can change between frames without a generation bump: the
    single spectrum's power axis (different profile, different dB range
    every click) and the time spectrogram's velocity axis (RPG-FMCW-94's
    per-chirp Nyquist velocity, which can differ at the newly clicked
    height).

    Even so, auto_y does NOT mean "refit unconditionally on every frame" --
    it refits only when the newly computed bounds actually DIFFER from the
    last ones this hook fit that same Bokeh range object to (tracked in
    state.auto_bounds by rng.id). That distinction matters because
    shared_axes=True unifies a dimension's range across every plot that
    uses it, by NAME, regardless of whether it's that plot's x or y axis --
    confirmed empirically: the time spectrogram's "velocity" y_range is
    THE SAME Bokeh model as the range spectrogram's and single spectrum's
    "velocity" x_range. Those two are apply_ranges=False and correctly
    generation-gated, but auto_y unconditionally overwriting the SAME
    shared object on every click (since selecting a new height always
    re-renders the time spectrogram) silently threw away the user's zoom
    on the other two panels every single time -- for MIRA's single-chirp
    instruments the "natural" bounds never even change, so this was a pure
    regression with no benefit. Comparing against the last-fit bounds
    keeps the legitimate RPG chirp-change refit while leaving the shared
    range alone on every other click.
    """
    def hook(plot, element):
        generation = state.range_generation
        for handle, dim, always in (("x_range", x_dim, False), ("y_range", y_dim, auto_y)):
            rng = plot.handles.get(handle)
            if rng is None or dim is None:
                continue
            bounds = _element_bounds(element, dim)
            if bounds is None:
                continue
            if always:
                if state.auto_bounds.get(rng.id) == bounds:
                    continue
                state.auto_bounds[rng.id] = bounds
            elif state.range_keys.get(rng.id) == generation:
                continue
            lo, hi = bounds
            rng.start, rng.end = lo, hi
            # keep the Bokeh reset tool honest about where "home" is
            if hasattr(rng, "reset_start"):
                rng.reset_start, rng.reset_end = lo, hi
            state.range_keys[rng.id] = generation
    return hook


class FlyoutSelect(pn.reactive.ReactiveHTML):
    """A single control with a genuine flyout submenu: click to open,
    hover an instrument to reveal its variables beside it (pure CSS
    :hover, no click needed), click a variable to select it and close.
    Neither a native <select> (no submenus at all) nor
    Panel's own NestedSelect (two separately-rendered linked widgets, not
    one control that flies out) can do this -- built with ReactiveHTML,
    Panel's supported mechanism for custom HTML/CSS/JS with two-way Python
    binding, since no existing widget offers it.

    Three ReactiveHTML quirks worth remembering if this ever needs changes:
    - Inline `onclick="${script(...)}"` handlers do NOT work on nodes
      generated by a `{% for %}` loop (fails at runtime with "data.script is
      not a function") -- only on static template nodes. So the leaf/row
      clicks are handled by ONE delegated onclick on the static #menu div,
      using event.target.closest(...) to figure out what was actually
      clicked, rather than one handler per generated row/leaf.
    - A String param referenced as a node's entire text content (e.g.
      `${label}` alone in a labeled/id'd node) is misdetected as a "child"
      slot expecting a Panel component, raising a RuntimeError at class
      definition time. `_child_config = {"label": "literal"}` tells
      ReactiveHTML to treat it as literal reactive text instead.
    - A `{% for %}` loop is only ever expanded from the value `options` has
      at construction time -- reassigning `self.options` later does NOT
      re-render it (confirmed directly: a minimal reproduction showed the
      DOM staying frozen at the original loop output after the param
      changed). Since this app doesn't know the real catalog until after
      the first Load completes -- well after these widgets are built -- the
      menu is instead built with a plain (non-Jinja) empty #menu div, filled
      by a JS `_scripts` function that runs both on initial "render" and
      whenever "options" changes, so it works both times the same way.
    """

    options = param.Dict(default={})  # {instrument: [(catalog_id, label), ...]}
    value = param.String(default=None, allow_None=True)
    label = param.String(default="(no data)")

    _child_config = {"label": "literal"}

    _template = """
    <div id="root" class="flyout-root">
      <button id="toggle" class="flyout-toggle" onclick="${script('toggle')}" type="button">
        ${label} <span class="flyout-caret">&#9662;</span>
      </button>
      <div id="menu" class="flyout-menu" onclick="${script('menu_click')}"></div>
    </div>
    """

    _scripts = {
        "render": "self.build_menu()",
        "options": "self.build_menu()",
        "build_menu": """
            function esc(s) {
                const d = document.createElement('div');
                d.textContent = s;
                return d.innerHTML;
            }
            let out = '';
            for (const instrument in data.options) {
                const items = data.options[instrument];
                let leaves = '';
                for (const pair of items) {
                    const cid = pair[0];
                    const lbl = pair[1];
                    leaves += '<div class="flyout-leaf" data-value="' + esc(String(cid)) + '" data-label="' + esc(String(lbl)) + '">' + esc(String(lbl)) + '</div>';
                }
                out += '<div class="flyout-row"><div class="flyout-row-label">' + esc(instrument) + ' <span class="flyout-caret">&#9656;</span></div><div class="flyout-submenu">' + leaves + '</div></div>';
            }
            menu.innerHTML = out;
        """,
        "toggle": "menu.classList.toggle('open');",
        "menu_click": """
            const leaf = event.target.closest('.flyout-leaf');
            if (leaf) {
                data.value = leaf.dataset.value;
                data.label = leaf.dataset.label;
                menu.classList.remove('open');
                return;
            }
        """,
    }

    _stylesheets = ["""
    .flyout-root { position: relative; font-size: 13px; width: 100%; box-sizing: border-box; }
    .flyout-toggle {
        width: 100%; text-align: left; padding: 6px 10px; border: 1px solid #ccc;
        border-radius: 4px; background: white; cursor: pointer; box-sizing: border-box;
        display: flex; justify-content: space-between; align-items: center;
        font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .flyout-toggle:hover { border-color: #999; }
    .flyout-menu {
        display: none; position: absolute; top: 100%; left: 0; z-index: 1000;
        background: white; border: 1px solid #ccc; border-radius: 4px;
        min-width: 100%; box-shadow: 0 2px 6px rgba(0,0,0,0.15); margin-top: 2px;
    }
    .flyout-menu.open { display: block; }
    .flyout-row { position: relative; }
    .flyout-row-label {
        padding: 6px 10px; cursor: pointer; white-space: nowrap;
        display: flex; justify-content: space-between; align-items: center; gap: 8px;
    }
    .flyout-row-label:hover { background: #f0f0f0; }
    .flyout-submenu {
        display: none; position: absolute; left: 100%; top: 0;
        background: white; border: 1px solid #ccc; border-radius: 4px;
        min-width: 220px; max-height: 320px; overflow-y: auto;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .flyout-row:hover > .flyout-submenu { display: block; }
    .flyout-leaf { padding: 6px 10px; cursor: pointer; white-space: nowrap; }
    .flyout-leaf:hover { background: #f0f0f0; }
    """]


def _dropdown_label(pv: gp.ProductVariable) -> str:
    """Display text for the variable dropdown only -- pv.label itself stays
    plain (no dimension suffix) since it's reused for plot titles, the
    categorical legend, and the curve y-axis label, where it would be noise."""
    dims = "time, height" if pv.height_dim is not None else "time"
    return f"{pv.label} ({dims})"


def _grouped_options(catalog: list[gp.ProductVariable]) -> dict[str, list[tuple[str, str]]]:
    """{instrument: [(catalog_id, display_label), ...]} -- FlyoutSelect's
    native options format, and also the grouping used to decide which
    instrument row a variable falls under."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    # 2D (time, height) curtains before 1D (time)-only series within each
    # instrument, so the more commonly wanted moments/curtains show up
    # first rather than interleaved alphabetically with metadata variables.
    for pv in sorted(catalog, key=lambda p: (p.instrument_id, p.height_dim is None, p.label)):
        grouped.setdefault(pv.instrument_id, []).append((pv.catalog_id, _dropdown_label(pv)))
    return grouped or {"(no data)": [(None, "(no data)")]}


def _default_catalog_id(catalog: list[gp.ProductVariable], stored_id, fallback_id):
    ids = {pv.catalog_id for pv in catalog}
    if stored_id in ids:
        return stored_id
    if fallback_id in ids:
        return fallback_id
    return catalog[0].catalog_id if catalog else None


def _label_for_catalog_id(catalog: list[gp.ProductVariable], catalog_id) -> str:
    pv = next((p for p in catalog if p.catalog_id == catalog_id), None)
    return _dropdown_label(pv) if pv else "(no data)"


def _uses_log(pv: gp.ProductVariable | None, clim=None) -> bool:
    """Whether this variable would be drawn on a log10 scale. clim defaults
    to the variable's own plot_range; a manual range with a non-positive
    minimum can't be shown logarithmically at all. Hooks read
    state.panel_log[index] instead -- what was ACTUALLY drawn."""
    if pv is None or not pv.log_scale:
        return False
    lo, hi = clim if clim is not None else (pv.plot_range or (None, None))
    return lo is not None and hi is not None and lo > 0 and hi > 0


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """A BasicTicker-style 1/2/5-decade tick series for [lo, hi] -- needed
    because a row-1 colorbar's ticker object is frozen at build time (see
    _panel_colorbar_models), so even plain linear variables have to have
    their ticks computed here rather than left to Bokeh."""
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    raw = (hi - lo) / target
    mag = 10 ** np.floor(np.log10(raw))
    step = next((m * mag for m in (1, 2, 5) if raw <= m * mag), 10 * mag)
    first = np.ceil(lo / step) * step
    ticks = np.arange(first, hi + step * 0.5, step)
    return [float(t) for t in ticks if lo <= t <= hi]


def _make_colorbar_hook(state: AppState, index: int, choice: "FlyoutSelect"):
    """Re-points this panel's colorbar at whichever variable is now selected:
    class names for a categorical one, 1e-7/1e-6/... decades for a
    log-scaled one, plain numbers otherwise.

    It has to happen in a hook because a DynamicMap's colorbar_opts are
    applied only on its FIRST frame -- which here is the no-data placeholder
    -- and, worse, a Bokeh ColorBar view caches the ticker/formatter it was
    built with and ignores later assignments (see _panel_colorbar_models).
    So the panel's permanent ticker/formatter objects are MUTATED in place;
    that both survives the caching and triggers the repaint.

    The color MAPPER is deliberately left to HoloViews. Swapping
    glyph.color_mapper by hand updates the model but does not re-rasterize
    the image, leaving stale pixels (this is what made epsilon render solid
    black beneath a correct-looking log colorbar) -- log scaling is done by
    transforming the DATA in _moment_image instead.
    """
    def hook(plot, element):
        cb = plot.handles.get("colorbar")
        if cb is None:
            return
        ticker, formatter = state.colorbar_models[index]
        pv = state.catalog_variable(choice.value)
        # A time-only variable's image is just an invisible placeholder (see
        # _moment_image/_moment_curve) -- its colorbar would otherwise still
        # show, pointlessly, since colorbar presence can't be toggled off
        # structurally once a panel's first frame has one.
        cb.visible = pv is None or pv.height_dim is not None
        mapper = cb.color_mapper
        lo, hi = getattr(mapper, "low", None), getattr(mapper, "high", None)
        if lo is None or hi is None:
            # An auto-ranged variable (no plot_meta entry, color limits on
            # "auto") leaves the mapper's bounds unset and lets Bokeh take
            # them from the data -- which is exactly what the ticks have to
            # span too, so take them from the data here as well.
            lo, hi = _element_bounds(element, "value") or (None, None)
        if pv is not None and pv.is_categorical:
            names = [label.lstrip("_") for label, _ in pm.CATEGORICAL[pv.var_name]]
            formatter.args = dict(names=names)
            formatter.code = _TICK_LABEL_JS["categorical"]
            ticker.ticks = list(range(len(names)))
            cb.major_label_text_font_size = "8pt"
            # ColorBar.width takes "auto" or a pixel int (no percentages).
            # This is the bar itself, not the class labels beside it.
            cb.width = 30
        elif state.panel_log.get(index):
            # The data is log10-transformed, so the ticks are integers -7,
            # -6, ... -- one per decade, relabelled as powers of ten.
            formatter.args = dict(names=[])
            formatter.code = _TICK_LABEL_JS["log"]
            ticker.ticks = ([] if lo is None or hi is None else
                            list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1)))
            cb.major_label_text_font_size = "11px"
            cb.width = "auto"
        else:
            formatter.args = dict(names=[])
            formatter.code = _TICK_LABEL_JS["linear"]
            ticker.ticks = _nice_ticks(lo, hi)
            cb.major_label_text_font_size = "11px"
            cb.width = "auto"
    return hook


def _make_hover_hook(state: AppState, index: int, choice: "FlyoutSelect"):
    """Same fix as _make_colorbar_hook, for the hover tool: every categorical
    variable wants its tooltip to show a class name instead of a raw integer
    code, but a HoverTool object passed via .opts(tools=[...]) only ever
    takes effect on the first frame. So this mutates the ALREADY-ATTACHED
    HoverTool's tooltips/formatters directly on every frame instead. The
    first two tooltip entries (time, height) are left exactly as HoloViews
    originally set them up (which includes its own datetime formatting for
    "time") -- only the third ("value") entry is ever replaced.
    """
    captured: dict = {}

    def hook(plot, element):
        hover = plot.handles.get("hover")
        if hover is None:
            return
        if "base_tooltips" not in captured:
            captured["base_tooltips"] = list(hover.tooltips[:2])
            captured["base_formatters"] = dict(hover.formatters)
        pv = state.catalog_variable(choice.value)
        if pv is not None and pv.is_categorical:
            names = [label.lstrip("_") for label, _ in pm.CATEGORICAL[pv.var_name]]
            hover.tooltips = captured["base_tooltips"] + [("class", "@image")]
            hover.formatters = {**captured["base_formatters"], "@image": CustomJSHover(
                args=dict(names=names), code="return names[Math.round(value)] ?? 'unknown';")}
        elif state.panel_log.get(index):
            # _moment_image feeds log10(value) to the glyph so the colours
            # come out logarithmic; undo that here so the tooltip still
            # quotes the measured value rather than its exponent.
            hover.tooltips = captured["base_tooltips"] + [("value", "@image")]
            hover.formatters = {**captured["base_formatters"], "@image": CustomJSHover(
                code="return isFinite(value) ? Math.pow(10, value).toExponential(2) : 'n/a';")}
        else:
            hover.tooltips = captured["base_tooltips"] + [("value", "@image")]
            hover.formatters = dict(captured["base_formatters"])
    return hook


def build_moment_controls(state: AppState, index: int):
    """Widgets for one row-1 panel slot: variable choice + color-limit controls.
    Rendering itself happens in render_grid() so all six panels can share axes."""
    settings = state.settings
    panel_state = settings.get_panel(index)
    grouped = _grouped_options(state.catalog)
    default_id = _default_catalog_id(state.catalog, panel_state["variable"], DEFAULT_PANEL_VARIABLES[index])

    # width is responsive (not fixed), so this grows to fill whatever share
    # of the row it's given -- lining it up with its panel's column below
    # even as the window widens, instead of staying a fixed pixel width
    # while the (also-responsive) grid panels stretch to fill the rest.
    choice = FlyoutSelect(options=grouped, value=default_id,
                           label=_label_for_catalog_id(state.catalog, default_id),
                           height=32, sizing_mode="stretch_width", margin=(5, 5, 5, 5))

    auto_color = panel_state["color_limits"] == "auto"
    vmin_w = pn.widgets.FloatInput(name="color min", value=0.0, disabled=auto_color, width=90)
    vmax_w = pn.widgets.FloatInput(name="color max", value=1.0, disabled=auto_color, width=90)
    auto_w = pn.widgets.Checkbox(name="auto color", value=auto_color)
    reset_btn = pn.widgets.Button(name="Reset panel", button_type="light", width=100)

    def on_reset(event):
        settings.reset_panel(index)
        new_id = _default_catalog_id(state.catalog, None, DEFAULT_PANEL_VARIABLES[index])
        choice.value = new_id
        choice.label = _label_for_catalog_id(state.catalog, new_id)
        auto_w.value = True

    reset_btn.on_click(on_reset)

    def sync_categorical_disable(catalog_id):
        pv = state.catalog_variable(catalog_id)
        is_cat = bool(pv and pv.is_categorical)
        auto_w.disabled = is_cat
        vmin_w.disabled = is_cat or auto_w.value
        vmax_w.disabled = is_cat or auto_w.value

    sync_categorical_disable(choice.value)

    def on_auto_toggle(event):
        vmin_w.disabled = event.new
        vmax_w.disabled = event.new
        settings.set_panel(index, color_limits="auto" if event.new else [vmin_w.value, vmax_w.value])

    auto_w.param.watch(on_auto_toggle, "value")

    def on_limits_change(event):
        if not auto_w.value:
            settings.set_panel(index, color_limits=[vmin_w.value, vmax_w.value])

    vmin_w.param.watch(on_limits_change, "value")
    vmax_w.param.watch(on_limits_change, "value")

    def on_choice_change(event):
        settings.set_panel(index, variable=event.new)
        sync_categorical_disable(event.new)

    choice.param.watch(on_choice_change, "value")

    # Collapsed by default: a Card's header is itself the toggle (no extra
    # button needed), keeping the panel compact until color limits actually
    # need adjusting. Settings are stacked vertically (rather than one tight
    # row) so the card can stay narrow while still fitting everything without
    # clipping.
    options_card = pn.Card(
        auto_w, pn.Row(vmin_w, vmax_w), reset_btn,
        title="Options", collapsed=True, width=220, margin=(5, 0, 5, 0),
    )
    controls = pn.Row(choice, options_card, align="center", sizing_mode="stretch_width")

    def refresh(catalog: list[gp.ProductVariable]):
        """Repopulate the instrument/variable options from a freshly
        discovered catalog, restoring the stored preference."""
        nonlocal grouped
        grouped = _grouped_options(catalog)
        stored_id = settings.get_panel(index)["variable"]
        new_id = _default_catalog_id(catalog, stored_id, DEFAULT_PANEL_VARIABLES[index])
        choice.options = grouped
        choice.value = new_id
        choice.label = _label_for_catalog_id(catalog, new_id)

    return controls, choice, auto_w, vmin_w, vmax_w, refresh


_EMPTY_TIME = np.array([np.datetime64("2020-01-01"), np.datetime64("2020-01-01T00:00:01")])


def _moment_marker(state: AppState):
    """The click-position crosshair, as its OWN DynamicMap independent of the
    image, so a click only re-renders this lightweight Points layer rather
    than the whole image."""
    opts = dict(color="red", size=8, marker="x", apply_ranges=False)
    if state.selected_time is None:
        return hv.Points([], kdims=["time", "height"]).opts(**opts)
    return hv.Points(
        [(np.datetime64(int(state.selected_time), "s"), state.selected_height)],
        kdims=["time", "height"],
    ).opts(**opts)


def _panel_colorbar_models(state: AppState, index: int) -> dict:
    """The Bokeh ColorBar VIEW caches model.ticker/model.formatter when it is
    first built and never looks at them again (BaseColorBarView.initialize),
    so assigning a new ticker or formatter to an existing colorbar changes
    the model while the plot keeps drawing the old labels -- which is why a
    panel kept showing plain 0..10 numbers for target_classification.

    What the view DOES do is repaint whenever those originally-passed objects
    themselves emit a change. So every row-1 panel is built with one
    permanent FixedTicker + CustomJSTickFormatter (returned here as
    colorbar_opts), and _make_colorbar_hook re-points THOSE objects at the
    current variable instead of replacing them.
    """
    if index not in state.colorbar_models:
        state.colorbar_models[index] = (
            FixedTicker(ticks=[]),
            CustomJSTickFormatter(args=dict(names=[]), code=_TICK_LABEL_JS["linear"]),
        )
    ticker, formatter = state.colorbar_models[index]
    return dict(ticker=ticker, formatter=formatter, major_label_policy=AllLabels())


# Tick-label JS, swapped into the panel's permanent CustomJSTickFormatter by
# _make_colorbar_hook. "names" is always passed (empty unless categorical).
_TICK_LABEL_JS = {
    "categorical": "return names[Math.round(tick)] ?? '';",
    # Log-scale variables are plotted as log10(value) (see _moment_image), so
    # a tick of -6 has to read as 1e-6.
    "log": "var v = Math.pow(10, tick); return v.toExponential(0).replace('e+', 'e');",
    "linear": """
        var a = Math.abs(tick);
        if (a !== 0 && (a < 1e-3 || a >= 1e5)) return tick.toExponential(1).replace('e+', 'e');
        return String(parseFloat(tick.toPrecision(6)));
    """,
}


def _no_data_moment(state: AppState, title: str = "no data", index: int | None = None):
    """A DynamicMap must return the same element type on every frame -- so
    the placeholder must already be an hv.Image like real data uses, never a
    bare hv.Text (mixing types crashes HoloViews after the first real
    update). It must also use the SAME time axis as whatever real data is
    currently loaded (not an arbitrary fixed placeholder range), because the
    time axis is linked across panels -- otherwise a linked panel with real
    2026 data and one still showing "no data" with a fixed 2020 dummy axis
    stretch the shared axis to cover both, producing garbled ticks.

    Same reasoning for height: this can render for just ONE panel (e.g. that
    panel's variable has no data this hour) while its neighbors show a real
    curtain, and since only one panel's hook actually performs a given
    generation's range refit (see _make_range_hook), a sparse 2-point height
    axis here risks winning that refit and corrupting the shared axis with
    its oversized Bokeh cell-edge padding -- so reuse the real height bins
    whenever spectra are loaded, exactly like the time-only branch above.
    """
    t0, t1 = state.display_time_bounds
    height_arr = state.spectra.height if state.spectra is not None else np.array([0.0, 1.0])
    opts = dict(colorbar=True, responsive=True, title=title, apply_ranges=False,
                tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"])
    if index is not None:
        opts["colorbar_opts"] = _panel_colorbar_models(state, index)
        state.panel_log[index] = False
    return hv.Image(((t0, t1), height_arr, np.full((len(height_arr), 2), np.nan)),
                     kdims=["time", "height"], vdims=["value"]).opts(**opts)


def _moment_image(state: AppState, index: int, catalog_id, auto_color, vmin, vmax):
    """The image only -- no marker (see _moment_marker) -- and its streams
    (in build_app) deliberately do NOT include selected_time/selected_height,
    so clicking to select a point never re-triggers this and never resets
    the axes."""
    pv = state.resolve_variable(catalog_id)
    if pv is None or state.spectra is None:
        return _no_data_moment(state, index=index)
    t0, t1 = state.hour_time_bounds
    time_arr, height_arr, values = gp.load_curtain(pv, t0, t1)
    if len(time_arr) == 0:
        return _no_data_moment(state, f"{pv.label} (no data this hour)", index=index)

    is_timeseries = height_arr is None
    if is_timeseries:
        # Time-only products (e.g. MWR's LWP) are drawn by _moment_curve
        # instead, as a real line plot on its own right-hand axis (see
        # _make_curve_axis_hook) -- a "height" isn't a meaningful axis for
        # them at all. This image is just an invisible placeholder (see
        # _moment_curve for why the panel still needs one: a DynamicMap's
        # own frames must always be the same element type).
        # It reuses the real height BINS the curtain panels use (not just
        # their two endpoints) because only one of the three row-1 panels'
        # hooks actually performs a given generation's range refit (see
        # _make_range_hook) -- a too-sparse placeholder grid would still be
        # numerically correct at the endpoints, but Bokeh pads an hv.Image by
        # half a cell beyond its outermost coordinates, and with only 2 rows
        # spanning the whole height range that padding is enormous, which
        # corrupted the shared height axis when this panel's hook won the
        # refit. Reusing the identical bin array keeps the bounds and
        # padding indistinguishable from the real curtain panels'.
        height_arr = state.spectra.height if state.spectra is not None else np.array([0.0, 1000.0])
        values = np.full((len(time_arr), len(height_arr)), np.nan)

    title = f"{pv.label} ({pv.units})" if pv.units else pv.label
    opts = dict(colorbar=True, colorbar_opts=_panel_colorbar_models(state, index),
                tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"],
                responsive=True, title=title, apply_ranges=False,
                xlabel="time (UTC)", ylabel="" if is_timeseries else "height (m)")
    # colorbar PRESENCE is always True -- even for the invisible timeseries
    # placeholder, which _make_colorbar_hook hides via .visible instead --
    # because it's a structural Bokeh property fixed by this DynamicMap's
    # very first frame: toggling it per-variable would only ever reflect
    # whichever type was selected FIRST. Its ticks and labels are then set
    # per frame by _make_colorbar_hook.
    state.panel_log[index] = False
    if pv.is_categorical:
        colors = [c for _, c in pm.CATEGORICAL[pv.var_name]]
        opts.update(cmap=colors, clim=(-0.5, len(colors) - 0.5))
    else:
        if auto_color:
            clim = pv.plot_range or (None, None)
        else:
            clim = (vmin, vmax)
        if _uses_log(pv, clim):
            # Log-scale variables are rendered by pre-transforming the DATA
            # to log10 and keeping HoloViews' own LINEAR mapper -- NOT by
            # swapping in a Bokeh LogColorMapper. Two hard-won reasons:
            #   * holoviews' `logz` option (which would build a log mapper)
            #     is fixed by this DynamicMap's FIRST frame, which is the
            #     no-data placeholder, so it can never turn on later; and
            #   * swapping glyph.color_mapper by hand from a hook updates
            #     the MODEL but does not re-rasterize the image, leaving
            #     stale pixels from the previous mapping (this is what made
            #     epsilon render solid black while its colorbar showed a
            #     correct log scale).
            # A data change, by contrast, always repaints. The colorbar's
            # tick labels are then relabelled back to 1e-7/1e-6/... in
            # _make_colorbar_hook, and the hover tooltip likewise shows the
            # untransformed value, so the log10 stays an internal detail.
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.log10(np.where(values > 0, values, np.nan))
            clim = (float(np.log10(clim[0])), float(np.log10(clim[1])))
            state.panel_log[index] = True
        opts.update(cmap=pv.cmap, clim=clim)

    return hv.Image((time_arr, height_arr, values.T), kdims=["time", "height"], vdims=["value"]).opts(**opts)


def _make_curve_axis_hook(state: AppState, choice: "FlyoutSelect"):
    """Gives the row-1 time-only curve (e.g. MWR's LWP) its own right-hand
    y-axis, scaled to the curve's real data -- entirely independent of the
    shared "height" axis, so panning/zooming height never distorts it and
    its true value range is always visible without having to zoom all the
    way out.

    This is built BY HAND with Bokeh's own extra_y_ranges/add_layout, not
    via HoloViews' multi_y=True option (which is what the old version of
    this used). multi_y=True has to be set on the whole Overlay, and doing
    that silently disables the y coordinate of hv.streams.Tap for every
    layer in it ("Tap stream parameters ['y'] not yet supported with
    multi_y=True") -- exactly how every row-1 panel reports the clicked
    height to the rest of the app. That broke click-to-select-a-spectrum
    app-wide the one time this used multi_y. Building the extra range/axis
    directly on the shared Bokeh figure and re-pointing just this curve's
    own glyph renderer at it (renderer.y_range_name) achieves the same
    visual result without ever touching the Overlay's axis count, so Tap
    keeps working normally.

    One catch found only by triggering a SECOND frame (the first render
    alone doesn't show it): HoloViews' own ElementPlot._update_ranges caches
    `plot.extra_y_ranges`/`plot.extra_y_scales` into its handles once
    ANYTHING is present in extra_y_ranges, and on every later frame iterates
    that dict expecting a matching entry in extra_y_scales for each key --
    a dict Bokeh does NOT keep in sync with extra_y_ranges on its own, so
    the manual `fig.extra_y_ranges["ts_value"] = ...` line above must be
    paired with an explicit `fig.extra_y_scales["ts_value"] = LinearScale()`
    or the very next click crashes with KeyError('ts_value') deep inside
    HoloViews, even though the first frame renders fine.
    """
    def hook(plot, element):
        fig = plot.state
        pv = state.catalog_variable(choice.value)
        is_timeseries = pv is not None and pv.height_dim is None
        axis = next((a for a in fig.right if isinstance(a, LinearAxis)
                     and a.y_range_name == "ts_value"), None)
        if axis is None:
            fig.extra_y_ranges["ts_value"] = Range1d(start=0, end=1)
            fig.extra_y_scales["ts_value"] = LinearScale()
            axis = LinearAxis(y_range_name="ts_value")
            fig.add_layout(axis, "right")
        renderer = plot.handles.get("glyph_renderer")
        if renderer is not None:
            renderer.y_range_name = "ts_value"
        axis.visible = is_timeseries
        axis.axis_label = (f"{pv.label} ({pv.units})" if is_timeseries and pv.units
                            else pv.label if is_timeseries else "")
        rng = fig.extra_y_ranges.get("ts_value")
        if rng is None or not is_timeseries:
            return
        bounds = _element_bounds(element, "ts_value")
        if bounds is None:
            return
        lo, hi = bounds
        pad = (hi - lo) * 0.05 or 0.5
        rng.start, rng.end = lo - pad, hi + pad
    return hook


def _no_data_curve():
    return hv.Curve([], kdims=["time"], vdims=["ts_value"])


def _moment_curve(state: AppState, catalog_id):
    """The line-plot counterpart to _moment_image, for time-only variables
    (e.g. MWR's LWP): a real hv.Curve on its own right-hand y-axis (see
    _make_curve_axis_hook), since a 1D time series has no natural
    relationship to the shared "height" axis the other two elements use.
    Always part of the row-1 Overlay, but empty (draws nothing) whenever the
    current variable is a real 2D curtain instead -- this DynamicMap's own
    frames must always be the same element type (a Curve, just sometimes an
    empty one), exactly like image_dmap's own placeholder handling.

    Deliberately does NOT set apply_ranges=False: unlike the shared
    time/height axes, this curve's own value axis never needs zoom
    preserved across clicks (its streams don't include
    selected_time/selected_height, so a click never re-renders it at all --
    only an hour or variable change does, and both of those SHOULD refit
    this axis to the new data's real range).
    """
    pv = state.resolve_variable(catalog_id)
    if pv is None or pv.height_dim is not None or state.spectra is None:
        return _no_data_curve()
    t0, t1 = state.hour_time_bounds
    time_arr, _, values = gp.load_curtain(pv, t0, t1)
    if len(time_arr) == 0:
        return _no_data_curve()
    return hv.Curve((time_arr, values), kdims=["time"], vdims=["ts_value"]).opts(
        color="darkorange", line_width=2)


def build_spectra_controls(state: AppState):
    channel_toggle = pn.widgets.RadioButtonGroup(
        name="Channel (spectrograms)", options={"co-polar": "co", "cross-polar": "cross"}, value=state.channel
    )

    def on_channel(event):
        state.channel = event.new
        state.settings.set("spectra_channel", event.new)

    channel_toggle.param.watch(on_channel, "value")
    return channel_toggle


def _no_data_range_spectrogram():
    img = hv.Image((np.array([-1.0, 1.0]), np.array([0.0, 1.0]), np.full((2, 2), np.nan)),
                    kdims=["velocity", "height"], vdims=["power"]).opts(
        cmap="viridis", colorbar=True, tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"],
        apply_ranges=False, responsive=True, title="Range spectrogram (no data)",
        xlabel="Doppler velocity (m/s)", ylabel="height (m)")
    return hv.Overlay([img])


def _range_spectrogram(state: AppState, channel: str):
    """Content legitimately changes on every click (a different time's
    profile); the axes are managed by _make_range_hook, not by HoloViews.

    title/height/responsive/apply_ranges/xlabel/ylabel are set on each
    per-chirp Image itself, NOT on the returned Overlay -- an Overlay's own
    .opts() are tied to that specific Overlay object's id, and this
    function's caller multiplies the result by a marker-line DynamicMap
    (see _range_height_marker), which builds a brand-new Overlay on every
    frame and silently drops any option that lived only on the old one.
    Concretely, that previously lost the title AND apply_ranges=False
    (which is what kept a click from resetting the user's zoom) the moment
    the marker line was added -- HoloViews' own per-ELEMENT options survive
    that recombination, so they belong here instead, exactly like
    _moment_image/_time_spectrogram already do. Same reasoning applies to
    "tap": without it in `tools`, Bokeh never attaches a TapTool to this
    figure at all, so clicking here did nothing.
    """
    s = state.spectra
    if s is None or state.selected_time is None:
        return _no_data_range_spectrogram()
    t_idx = s.nearest_time_index(state.selected_time)
    segments = s.range_profile_segments(t_idx, channel)
    title = f"Range spectrogram ({channel})"
    images = [
        hv.Image((vel, s.height[rng_slice], block), kdims=["velocity", "height"], vdims=["power"]).opts(
            cmap="viridis", colorbar=True, tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"],
            apply_ranges=False, responsive=True, title=title,
            xlabel="Doppler velocity (m/s)", ylabel="height (m)")
        for rng_slice, vel, block in segments
    ]
    return hv.Overlay(images)


def _no_data_time_spectrogram(state: AppState):
    """Shares the "time" kdim with the moments/row1 panels, so it must span
    the same window; the axes themselves are managed by _make_range_hook."""
    t0, t1 = state.display_time_bounds
    return hv.Image(((t0, t1), np.array([-1.0, 1.0]), np.full((2, 2), np.nan)),
                     kdims=["time", "velocity"], vdims=["power"]).opts(
        cmap="viridis", colorbar=True, responsive=True,
        tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"],
        title="Time spectrogram (no data)", apply_ranges=False,
        xlabel="time (UTC)", ylabel="Doppler velocity (m/s)")


def _time_spectrogram(state: AppState, channel: str):
    """Shows the full loaded hour (not a narrow window around the selection)
    so its "time" axis has the same natural extent as the moments panels and
    can be genuinely axis-linked with them, rather than fighting over a
    shared range at two different intended zoom levels. "tap" must be in
    `tools` for clicking here to select a time at all -- without it Bokeh
    never attaches a TapTool to this figure."""
    s = state.spectra
    if s is None or state.selected_height is None:
        return _no_data_time_spectrogram(state)
    r_idx = s.nearest_range_index(state.selected_height)
    block = s.time_series(r_idx, 0, s.n_time, channel)
    vel = s.chirp.velocity_axis(r_idx)
    times = np.array([np.datetime64(int(x), "s") for x in s.time])
    return hv.Image((times, vel, block.T), kdims=["time", "velocity"], vdims=["power"]).opts(
        cmap="viridis", colorbar=True, tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"],
        responsive=True,
        title=f"Time spectrogram at {s.height[r_idx]:.0f} m ({channel})", apply_ranges=False,
        xlabel="time (UTC)", ylabel="Doppler velocity (m/s)",
    )


def _range_height_marker(state: AppState):
    """A red line across the range spectrogram at the currently selected
    height, so it's visible which profile row the single spectrum panel is
    showing. Always an HLine (never a differently-shaped placeholder) so
    this DynamicMap's frames stay one element type; when nothing is
    selected yet it's just made fully transparent rather than omitted.
    HLine has no "height"/"velocity" dim of its own, so overlaying it never
    perturbs _make_range_hook's element.range(...) computation (verified:
    only the Image layers contribute to that)."""
    y = state.selected_height if state.selected_height is not None else 0
    alpha = 1.0 if state.selected_height is not None else 0.0
    return hv.HLine(y).opts(color="red", line_width=1.5, line_alpha=alpha, apply_ranges=False)


def _time_selected_marker(state: AppState):
    """Same as _range_height_marker, but a vertical line on the time
    spectrogram at the currently selected time."""
    if state.selected_time is None:
        x, alpha = 0, 0.0
    else:
        x, alpha = np.datetime64(int(state.selected_time), "s"), 1.0
    return hv.VLine(x).opts(color="red", line_width=1.5, line_alpha=alpha, apply_ranges=False)


def _no_data_spectrum():
    empty_co = hv.Curve(([], []), kdims=["velocity"], vdims=["power"], label="co-polar").opts(color="steelblue")
    empty_cx = hv.Curve(([], []), kdims=["velocity"], vdims=["power"], label="cross-polar").opts(color="firebrick")
    return (empty_co * empty_cx).opts(
        hv.opts.Curve(responsive=True, tools=["hover"], apply_ranges=False),
        hv.opts.Overlay(title="Spectrum (no data)", legend_position="top_right", apply_ranges=False),
    )


def _spectrum_plot(state: AppState):
    """Always overlays co- and cross-polar spectra (no channel dependency here)."""
    s = state.spectra
    if s is None or state.selected_time is None or state.selected_height is None:
        return _no_data_spectrum()
    t_idx = s.nearest_time_index(state.selected_time)
    r_idx = s.nearest_range_index(state.selected_height)
    vel_co, db_co = s.spectrum(t_idx, r_idx, "co")
    vel_cx, db_cx = s.spectrum(t_idx, r_idx, "cross")
    label = (f"Spectrum @ {s.height[r_idx]:.0f} m, "
              f"{dt.datetime.utcfromtimestamp(int(s.time[t_idx])).strftime('%H:%M:%S')} UTC")
    curve_co = hv.Curve((vel_co, db_co), kdims=["velocity"], vdims=["power"], label="co-polar").opts(color="steelblue")
    curve_cx = hv.Curve((vel_cx, db_cx), kdims=["velocity"], vdims=["power"], label="cross-polar").opts(color="firebrick")
    return (curve_co * curve_cx).opts(
        hv.opts.Curve(responsive=True, tools=["hover"], apply_ranges=False),
        hv.opts.Overlay(title=label, legend_position="top_right", apply_ranges=False,
                         xlabel="Doppler velocity (m/s)", ylabel="power (dB)"),
    )


def build_app() -> pn.template.BaseTemplate:
    settings = ss.Settings()
    state = AppState(settings)

    try:
        site_options = {name: sid for sid, name in cc.list_sites()}
    except Exception:
        site_options = {"Hyytiälä": "hyytiala"}
    default_site_name = next((n for n, i in site_options.items() if i == state.site), next(iter(site_options)))

    site_select = pn.widgets.Select(name="", options=site_options, value=site_options[default_site_name],
                                     width=180, align="end")
    day_input = pn.widgets.DatePicker(name="", value=state.day, width=150, align="end")
    instrument_select = pn.widgets.Select(
        name="", options={"RPG-FMCW-94": "rpg-fmcw-94", "MIRA-10": "mira-10", "MIRA-35": "mira-35"},
        value=state.instrument, width=130, align="end")
    # Hour options are always 0-23, independent of what's actually loaded --
    # so the picker is immediately usable (no "empty until Load" moment) and
    # the user always knows loading is scoped to exactly one hour, never the
    # whole day. Labels get a "(no data)" suffix after Load reveals which
    # hours actually have a spectra file for this site/date.
    hour_select = pn.widgets.Select(
        name="", options={f"{h:02d} UTC": h for h in range(24)}, value=state.hour_index,
        width=150, align="end")
    prev_btn = pn.widgets.Button(name="< prev hour", width=100, align="end")
    next_btn = pn.widgets.Button(name="next hour >", width=100, align="end")
    load_btn = pn.widgets.Button(name="Load", button_type="primary", width=80, align="end")
    cache_day_btn = pn.widgets.Button(name="Cache entire day", button_type="success", width=130, align="end")
    reset_all_btn = pn.widgets.Button(name="Reset all settings", button_type="warning", width=140, align="end")
    clean_cache_btn = pn.widgets.Button(name="Clean cache", button_type="danger", width=100, align="end")
    status = pn.pane.Markdown("", sizing_mode="stretch_width")
    download_status = pn.pane.Markdown("", sizing_mode="stretch_width")

    def refresh_instrument_options():
        """Restrict the instrument picker to whatever actually has raw
        spectra for this site/day, discovered as part of this Load (see
        cc.list_available_instruments) -- mirrors refresh_hour_options()
        below, just filtering options instead of annotating them, since an
        instrument that's never present at this site is more clutter than
        the "(no data)" hours can still legitimately be picked to see."""
        available = cc.list_available_instruments(state.site, state.day)
        all_labels = {"RPG-FMCW-94": "rpg-fmcw-94", "MIRA-10": "mira-10", "MIRA-35": "mira-35"}
        opts = {k: v for k, v in all_labels.items() if v in available} or all_labels
        current = instrument_select.value
        instrument_select.options = opts
        instrument_select.value = current if current in opts.values() else next(iter(opts.values()))

    def refresh_hour_options():
        available = state.hours_with_data()
        current = hour_select.value
        opts = {
            (f"{h:02d} UTC" if h in available else f"{h:02d} UTC (no data)"): h
            for h in range(24)
        }
        hour_select.options = opts
        hour_select.value = current

    def load_status_message() -> str:
        if state.spectra is None:
            return (f"Loaded {len(state.catalog)} curtain variable(s) for the day, but no spectra "
                     f"for hour {state.hour_index:02d} UTC -- pick a different hour above.")
        return (f"Loaded hour {state.hour_index:02d} UTC ({len(state.available_hours)} hour(s) "
                 f"available that day), {len(state.catalog)} curtain variable(s).")

    # Background work (download/decode) runs in a thread via run_in_executor
    # so the event loop stays free to actually flush status-text updates to
    # the browser as they happen -- a plain blocking call inside a sync
    # Panel callback only reaches the browser as one patch at the very end,
    # no matter how many times you set .object during it.
    def _push(fn):
        doc = pn.state.curdoc
        if doc is not None:
            doc.add_next_tick_callback(fn)
        else:
            fn()

    _last_progress_report = {"filename": None, "mb": -1}

    def on_progress(filename: str, downloaded: int, total: int):
        # Throttle: only push a status update every ~5 MB (or on completion)
        # so a 300 MB file doesn't flood the browser with hundreds of pushes.
        mb = downloaded // (5 * 1024 * 1024)
        done = downloaded >= total
        if filename == _last_progress_report["filename"] and mb == _last_progress_report["mb"] and not done:
            return
        _last_progress_report["filename"] = filename
        _last_progress_report["mb"] = mb
        pct = 100 * downloaded / total if total else 100
        text = f"Downloading `{filename}`: {cc.format_bytes(downloaded)} / {cc.format_bytes(total)} ({pct:.0f}%)"
        _push(lambda: setattr(download_status, "object", text))

    def on_status(text: str):
        _push(lambda: setattr(status, "object", text))

    def show_cache_size():
        size = cc.cache_size_bytes(state.cache_dir)
        download_status.object = f"Cache: {cc.format_bytes(size)} at `{state.cache_dir}`"

    def set_controls_disabled(disabled: bool):
        for w in (load_btn, cache_day_btn, hour_select, prev_btn, next_btn,
                  site_select, day_input, instrument_select):
            w.disabled = disabled

    async def do_load(event=None):
        set_controls_disabled(True)
        status.object = "Loading..."
        state.site = site_select.value
        state.day = day_input.value
        state.instrument = instrument_select.value
        state.hour_index = hour_select.value
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: state.refresh_site_day(on_progress=on_progress, on_status=on_status))
        except ConnectionError as exc:
            status.object = f"⚠️ {exc}"
            set_controls_disabled(False)
            return

        def _finish_load():
            # Deferred via _push (add_next_tick_callback) rather than run
            # directly here: mutating certain widgets right after the
            # `await` above resumes this coroutine, outside Bokeh's
            # document-lock context, has crashed before with
            # "_pending_writes should be non-None...". Bundled into one
            # callback (rather than wrapping just one call) so
            # range_generation still bumps AFTER the variable selection
            # actually settles -- otherwise the axes could refit to stale
            # bounds one frame before the correct variable's data arrives.
            refresh_instrument_options()
            refresh_hour_options()
            refresh_variable_options()
            status.object = load_status_message()
            show_cache_size()
            settings.set_last_session(state.site, state.day.isoformat(), state.hour_index, state.instrument)
            state.range_generation += 1
            set_controls_disabled(False)

        _push(_finish_load)

    load_btn.on_click(do_load)

    async def on_hour_change(event):
        if event.new is None or not state.available_hours:
            return  # ignore the widget-construction event before first Load
        if (site_select.value != state.site or day_input.value != state.day
                or instrument_select.value != state.instrument):
            # The user changed site/date/instrument but hasn't clicked Load
            # yet -- state.site/day/instrument (and therefore
            # state.available_hours, which this quick-path relies on
            # instead of re-querying) still reflect the PREVIOUSLY loaded
            # combination. Without this guard, picking an hour here would
            # quietly download that hour for the OLD site/date instead of
            # the one now showing in the dropdowns. Require an explicit
            # Load to resync everything together.
            return
        set_controls_disabled(True)
        status.object = "Decoding hour..."
        state.hour_index = event.new
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: state.load_hour(on_progress=on_progress, on_status=on_status))
        except ConnectionError as exc:
            status.object = f"⚠️ {exc}"
            set_controls_disabled(False)
            return
        status.object = load_status_message()
        show_cache_size()
        settings.set_last_session(state.site, state.day.isoformat(), state.hour_index, state.instrument)
        state.range_generation += 1  # refit axes for the newly loaded hour
        set_controls_disabled(False)

    hour_select.param.watch(on_hour_change, "value")

    def step_hour(delta):
        def _cb(event):
            new_h = hour_select.value + delta
            if 0 <= new_h <= 23:
                hour_select.value = new_h
        return _cb

    prev_btn.on_click(step_hour(-1))
    next_btn.on_click(step_hour(1))

    async def do_cache_day(event=None):
        """Pre-fetches everything for the currently loaded site/day/
        instrument: every still-lazy product variable AND every available
        hour's raw spectra, not just what's on screen right now -- so a
        later click on any variable or hour picker is instant, no
        surprise download. Deliberately does NOT re-run discover()/re-read
        the dropdowns first: it caches exactly what's already loaded, the
        same combination Load last resolved."""
        set_controls_disabled(True)
        status.object = "Caching entire day..."
        loop = asyncio.get_running_loop()
        n_products = await loop.run_in_executor(
            None, lambda: state.cache_all_products(on_progress=on_progress, on_status=on_status))
        n_hours, n_failed = await loop.run_in_executor(
            None, lambda: state.cache_all_hours(on_progress=on_progress, on_status=on_status))

        def _finish_cache_day():
            refresh_variable_options()
            msg = f"Cached {n_products} product(s) and {n_hours} spectra hour(s) for {state.day.isoformat()}."
            if n_failed:
                msg += f" ({n_failed} hour(s) failed -- try again if your connection was interrupted.)"
            status.object = msg
            show_cache_size()
            set_controls_disabled(False)

        _push(_finish_cache_day)

    cache_day_btn.on_click(do_cache_day)

    def do_reset_all(event):
        settings.reset_all()
        # Location.reload is a Boolean param, not a method -- setting it
        # True is what actually tells the browser to reload (calling it as
        # reload() raises "'bool' object is not callable", since that's
        # exactly what the attribute already is).
        pn.state.location.reload = True

    reset_all_btn.on_click(do_reset_all)

    def do_clean_cache(event):
        status.object = "Clearing cache..."
        cc.clear_data_cache(state.cache_dir)
        status.object = "Cache cleared. Click Load to re-download."
        show_cache_size()

    clean_cache_btn.on_click(do_clean_cache)

    status.object = "Loading last-used site/date/hour..."
    show_cache_size()
    pn.state.onload(do_load)  # auto-load the last-used session; fast if cached

    moment_controls = [build_moment_controls(state, i) for i in range(N_MOMENT_PANELS)]
    channel_toggle = build_spectra_controls(state)

    def refresh_variable_options():
        """Widget options are built once from an empty catalog (before the
        first Load); repopulate them from the freshly discovered catalog."""
        for _, _choice, _auto, _vmin, _vmax, refresh in moment_controls:
            refresh(state.catalog)

    def on_tap_factory(i):
        def on_tap(x, y):
            if x is None or y is None:
                return
            state.selected_time = unix_seconds(to_datetime64(x))
            state.selected_height = float(y)
        return on_tap

    def on_tap_range(x, y):
        # Range spectrogram is velocity-vs-height at the CURRENTLY selected
        # time -- there's no time axis here to also update, unlike row1's
        # tap which reports both.
        if y is None:
            return
        state.selected_height = float(y)

    def on_tap_time(x, y):
        # Time spectrogram is time-vs-velocity at the CURRENTLY selected
        # height -- symmetric to on_tap_range above.
        if x is None:
            return
        state.selected_time = unix_seconds(to_datetime64(x))

    def param_stream(obj, name, unique_key):
        return hv.streams.Params(obj, [name], rename={name: unique_key})

    # Each panel is a real hv.DynamicMap (not a plain pn.bind function)
    # driven by hv.streams.Params watching the relevant widgets/state.
    # Combining DynamicMaps with `+`/hv.Layout keeps ONE persistent set of
    # Bokeh plot models that HoloViews updates in place and links shared
    # axes across -- rebuilding a whole hv.Layout from scratch on every
    # pn.bind call (the previous approach) raced Bokeh's model patching and
    # silently dropped updates after the first render.
    row1_dmaps = []
    for i, (_, choice, auto_w, vmin_w, vmax_w, _refresh) in enumerate(moment_controls):
        tap = hv.streams.Tap(x=None, y=None)

        def make_image_callback(idx=i, ch=choice, aw=auto_w, vn=vmin_w, vx=vmax_w):
            def _cb(**_kwargs):
                return _moment_image(state, idx, ch.value, aw.value, vn.value, vx.value)
            return _cb

        def make_curve_callback(ch=choice):
            def _cb(**_kwargs):
                return _moment_curve(state, ch.value)
            return _cb

        # Image, marker and curve are deliberately SEPARATE DynamicMaps: the
        # image/curve only react to hour/variable/color changes, the marker
        # reacts only to the click position, so a click re-renders as little
        # as possible. Axis ranges are owned by the hook (see _make_range_hook).
        image_dmap = hv.DynamicMap(make_image_callback(), streams=[
            tap,
            param_stream(choice, "value", f"var{i}"),
            param_stream(auto_w, "value", f"auto{i}"),
            param_stream(vmin_w, "value", f"vmin{i}"),
            param_stream(vmax_w, "value", f"vmax{i}"),
            hv.streams.Params(state, ["hour_index", "range_generation"]),
        ]).opts(hv.opts.Image(hooks=[_make_colorbar_hook(state, i, choice), _make_hover_hook(state, i, choice)]))
        marker_dmap = hv.DynamicMap(
            lambda **_kw: _moment_marker(state),
            streams=[hv.streams.Params(state, ["selected_time", "selected_height"])],
        )
        # A time-only variable (e.g. MWR's LWP) is drawn by this Curve on its
        # own right-hand y-axis, built by hand rather than via multi_y (see
        # _make_curve_axis_hook for why).
        curve_dmap = hv.DynamicMap(make_curve_callback(), streams=[
            param_stream(choice, "value", f"curve_var{i}"),
            hv.streams.Params(state, ["hour_index", "range_generation"]),
        ]).opts(hv.opts.Curve(hooks=[_make_curve_axis_hook(state, choice)]))
        tap.add_subscriber(on_tap_factory(i))
        row1_dmaps.append((image_dmap * marker_dmap * curve_dmap).opts(
            hv.opts.Overlay(apply_ranges=False,
                            hooks=[_make_range_hook(state, "time", "height")])))

    def _range_cb(**_kw):
        return _range_spectrogram(state, channel_toggle.value)

    def _time_cb(**_kw):
        return _time_spectrogram(state, channel_toggle.value)

    def _spectrum_cb(**_kw):
        return _spectrum_plot(state)

    # auto_y=True where the y axis SHOULD refit on every update: the time
    # spectrogram's velocity axis changes with the selected chirp, and the
    # spectrum's power axis would otherwise clip later spectra off-scale.
    # Each spectrogram's selection-marker line is its OWN DynamicMap (like
    # row1's _moment_marker) so moving the crosshair doesn't force a re-fetch
    # of the whole spectrogram image -- only _range_cb/_time_cb's own streams
    # (selected_time / selected_height respectively, since that picks WHICH
    # profile is shown) do that.
    range_marker_dmap = hv.DynamicMap(
        lambda **_kw: _range_height_marker(state),
        streams=[hv.streams.Params(state, ["selected_height"])],
    )
    time_marker_dmap = hv.DynamicMap(
        lambda **_kw: _time_selected_marker(state),
        streams=[hv.streams.Params(state, ["selected_time"])],
    )
    # Tap streams -- like row1's, added purely to route a click through to
    # state via on_tap_range/on_tap_time; they don't affect what's drawn
    # (the callback ignores its own x/y and just re-reads current state),
    # so a click here re-renders the same content, then range_marker_dmap/
    # time_marker_dmap and the OTHER panels react to the resulting state
    # change.
    tap_range = hv.streams.Tap(x=None, y=None)
    tap_range.add_subscriber(on_tap_range)
    tap_time = hv.streams.Tap(x=None, y=None)
    tap_time.add_subscriber(on_tap_time)
    range_dmap = (hv.DynamicMap(
        _range_cb,
        streams=[tap_range, param_stream(channel_toggle, "value", "ch_range"),
                 hv.streams.Params(state, ["hour_index", "selected_time", "range_generation"])],
    ) * range_marker_dmap).opts(hv.opts.Overlay(apply_ranges=False,
                            hooks=[_make_range_hook(state, "velocity", "height")]))
    time_dmap = (hv.DynamicMap(
        _time_cb,
        streams=[tap_time, param_stream(channel_toggle, "value", "ch_time"),
                 hv.streams.Params(state, ["hour_index", "selected_height", "range_generation"])],
    ) * time_marker_dmap).opts(hv.opts.Overlay(apply_ranges=False,
                          hooks=[_make_range_hook(state, "time", "velocity", auto_y=True)]))
    spectrum_dmap = hv.DynamicMap(
        _spectrum_cb,
        streams=[hv.streams.Params(state, ["hour_index", "selected_time", "selected_height", "range_generation"])],
    ).opts(hv.opts.Overlay(apply_ranges=False,
                            hooks=[_make_range_hook(state, "velocity", "power", auto_y=True)]))

    grid = hv.Layout(row1_dmaps + [range_dmap, time_dmap, spectrum_dmap]).cols(3).opts(shared_axes=True)
    # stretch_both (not stretch_width): the individual panels are already
    # responsive=True with no fixed height (see _moment_image et al.), which
    # makes each one stretch_both on its own -- but the CONTAINING pane also
    # needs to claim real vertical space for that to have anywhere to go,
    # rather than collapsing to Bokeh's default height and leaving the rest
    # of the browser window empty below it.
    grid_pane = pn.pane.HoloViews(grid, sizing_mode="stretch_both")

    top_bar = pn.Row(site_select, day_input, hour_select, prev_btn, next_btn, load_btn,
                      cache_day_btn, reset_all_btn, clean_cache_btn, align="end")
    # stretch_width + each child ALSO stretch_width (see build_moment_controls)
    # so the 3 dropdown groups split the row's width equally, matching the 3
    # equal-width columns of the grid panels below -- not just on narrow
    # screens where they happen to end up similarly sized by coincidence.
    controls_row = pn.Row(*[mc[0] for mc in moment_controls], align="end", sizing_mode="stretch_width")

    # stretch_both on the outer Column: top_bar/controls_row/the bottom
    # status row keep their own natural (fixed) height since they never set
    # a stretch_height/stretch_both sizing_mode, but grid_pane's is
    # stretch_both, so it's the one that expands to consume whatever
    # vertical space those fixed-height rows don't need -- filling the
    # window instead of leaving empty space below a fixed-height grid.
    return pn.Column(
        top_bar,
        controls_row,
        grid_pane,
        pn.Row(instrument_select, channel_toggle, status, download_status, align="center"),
        sizing_mode="stretch_both",
    )


if pn.state.served:
    build_app().servable(title="PRISM")
