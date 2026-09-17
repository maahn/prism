"""PRISM: Profiling & Remote-sensing Interactive Spectra Monitor.

Interactive viewer for ground-based remote-sensing time-height products and
raw Doppler spectra, currently sourced from the Cloudnet data portal.

Run with:  panel serve src/prism/app.py --show
"""
from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn
import param
from bokeh.models import (
    BasicTicker, BasicTickFormatter, CustomJSHover, CustomJSTickFormatter, FixedTicker,
    LinearColorMapper, LogColorMapper, LogTicker, LogTickFormatter,
)
from bokeh.models.widgets import Select as BokehSelect
from holoviews.plotting.util import process_cmap

from prism import cloudnet_client as cc
from prism import gridded_products as gp
from prism import mira_reader as mr
from prism import plot_meta as pm
from prism import rpg_reader as rr
from prism import settings_store as ss

pn.extension()
hv.extension("bokeh")
# Radar/height grids are piecewise- rather than perfectly-uniform (chirp
# boundaries, instrument-specific gate spacing). We deliberately render as a
# raster (hv.Image) rather than per-cell QuadMesh for speed, which needs a
# looser uniformity tolerance than the default.
hv.config.image_rtol = 0.5

N_MOMENT_PANELS = 3
DEFAULT_PANEL_VARIABLES = ["radar:Zh", "radar:v", "radar:width"]


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
        self.catalog = gp.discover(self.site, self.day, self.cache_dir, on_progress=on_progress)
        self.load_hour(on_progress=on_progress, on_status=on_status)

    def hours_with_data(self) -> set[int]:
        return {int(r.filename.split("_")[1][0:2]) for r in self.available_hours}

    def _remote_for_hour(self, hour: int) -> cc.RemoteFile | None:
        return next((r for r in self.available_hours if int(r.filename.split("_")[1][0:2]) == hour), None)

    def load_hour(self, on_progress=None, on_status=None):
        """Downloads and decodes exactly ONE hour's raw spectra file -- the
        one currently selected in hour_index -- never the whole day. The raw
        file is deleted after a successful decode: only the much smaller
        processed zarr cache is kept, since ensure_decoded() never needs the
        raw file again once its _SUCCESS marker exists (checked here too, so
        a re-selected hour whose raw file was already deleted doesn't get
        re-downloaded just to satisfy this function's own plumbing).
        Decoding itself is dispatched by instrument -- rpg_reader for RPG's
        LV0 binary format, mira_reader for METEK's znc/HDF5 format -- but
        both write the SAME zarr schema, so SpectraHour and everything
        downstream of it don't need to know or care which one ran."""
        remote = self._remote_for_hour(self.hour_index)
        if remote is None:
            self.spectra = None
            return
        decoder = mr if self.instrument in ("mira-10", "mira-35") else rr
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
        self.spectra = rr.SpectraHour(zpath)
        mid = self.spectra.n_time // 2
        self.selected_time = float(self.spectra.time[mid])
        self.selected_height = float(self.spectra.height[
            self.spectra.nearest_range_index(float(np.median(self.spectra.height)))
        ])

    def catalog_variable(self, catalog_id: str) -> gp.ProductVariable | None:
        return next((p for p in self.catalog if p.catalog_id == catalog_id), None)

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

    auto_y=True opts the y axis out of that rule, for axes that SHOULD
    refit every frame (the single spectrum's power axis, and the time
    spectrogram's velocity axis, which changes with the selected chirp).
    """
    def hook(plot, element):
        generation = state.range_generation
        for handle, dim, always in (("x_range", x_dim, False), ("y_range", y_dim, auto_y)):
            rng = plot.handles.get(handle)
            if rng is None or dim is None:
                continue
            if not always and state.range_keys.get(rng.id) == generation:
                continue
            bounds = _element_bounds(element, dim)
            if bounds is None:
                continue
            lo, hi = bounds
            rng.start, rng.end = lo, hi
            # keep the Bokeh reset tool honest about where "home" is
            if hasattr(rng, "reset_start"):
                rng.reset_start, rng.reset_end = lo, hi
            state.range_keys[rng.id] = generation
    return hook


class _CatalogChoice(param.Parameterized):
    """Mirrors the raw Bokeh Select's chosen catalog_id as a param, so the
    rest of the app (DynamicMap streams, settings persistence, categorical-
    legend lookup) has something to watch -- a plain Bokeh model's on_change
    callback isn't itself a param event."""
    catalog_id = param.String(default=None, allow_None=True)


def _grouped_options(catalog: list[gp.ProductVariable]) -> dict[str, list[tuple[str, str]]]:
    """{instrument: [(catalog_id, variable_label), ...]}, for a single
    native Bokeh Select rendered with real HTML <optgroup>s -- one genuine
    dropdown showing "instrument > variable" sub-groups, rather than two
    linked widgets or a custom flyout menu."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for pv in sorted(catalog, key=lambda p: (p.instrument_id, p.label)):
        grouped.setdefault(pv.instrument_id, []).append((pv.catalog_id, pv.label))
    return grouped or {"(no data)": [(None, "(no data)")]}


def _default_catalog_id(catalog: list[gp.ProductVariable], stored_id, fallback_id):
    ids = {pv.catalog_id for pv in catalog}
    if stored_id in ids:
        return stored_id
    if fallback_id in ids:
        return fallback_id
    return catalog[0].catalog_id if catalog else None


def _make_colorbar_hook(state: AppState, choice: "_CatalogChoice"):
    """Like _make_range_hook: a colorbar's formatter/ticker/width, and even
    the color MAPPER's type (linear vs log), are all fixed by this
    DynamicMap's very first frame and never re-applied by HoloViews' own
    .opts() on later frames -- switching a panel to a different variable
    afterwards leaves the previous one's colorbar/mapper setup in place. So
    this hook sets everything directly on the already-attached Bokeh
    ColorBar/glyph handles on every frame instead, always covering all three
    cases (target_classification, other log-scale, plain linear) so
    switching a panel between them can never leave stale state behind.

    For log-scale variables specifically: rather than pre-transforming the
    DATA to log10 and keeping a linear mapper (which would make hover
    tooltips show a meaningless log10 number instead of the real physical
    value), this swaps in a genuine Bokeh LogColorMapper on the real,
    untransformed data -- so the colorbar's log-scale ticks (1e-7, 1e-6, ...)
    and the hover tooltip's actual value both stay correct and consistent.
    """
    def hook(plot, element):
        cb = plot.handles.get("colorbar")
        glyph = plot.handles.get("glyph")
        cmapper = plot.handles.get("color_mapper")
        if cb is None:
            return
        pv = state.catalog_variable(choice.catalog_id)
        # A time-only variable's image is just an invisible placeholder (see
        # _moment_image/_moment_curve) -- its colorbar would otherwise still
        # show, pointlessly, since colorbar presence can't be toggled off
        # structurally once a panel's first frame has one.
        cb.visible = pv is None or pv.height_dim is not None
        if pv is not None and pv.is_categorical and pv.var_name == "target_classification":
            names = [label.lstrip("_") for label, _ in pm.CATEGORICAL[pv.var_name]]
            cb.ticker = FixedTicker(ticks=list(range(len(names))))
            cb.formatter = CustomJSTickFormatter(args=dict(names=names),
                                                  code="return names[Math.round(tick)] ?? '';")
            cb.major_label_text_font_size = "8pt"
            cb.width = 140
        else:
            cb.major_label_text_font_size = "11px"
            cb.width = 150
            has_bounds = cmapper is not None and cmapper.low is not None and cmapper.high is not None
            use_log = pv is not None and pv.log_scale and has_bounds and cmapper.low > 0
            if glyph is not None and has_bounds:
                # Rebuild the mapper every frame (not just when switching
                # to/from log-scale) so a panel that was PREVIOUSLY log-scale
                # never leaves a stale LogColorMapper attached after
                # switching to a variable that goes negative (which a log
                # mapper can't render at all).
                palette = process_cmap(pv.cmap if pv is not None else "viridis", ncolors=256)
                mapper_cls = LogColorMapper if use_log else LinearColorMapper
                mapper = mapper_cls(palette=palette, low=cmapper.low, high=cmapper.high)
                glyph.color_mapper = mapper
                cb.color_mapper = mapper
            if use_log:
                cb.ticker = LogTicker()
                cb.formatter = LogTickFormatter()
            else:
                cb.ticker = BasicTicker()
                cb.formatter = BasicTickFormatter()
    return hook


def _make_hover_hook(state: AppState, choice: "_CatalogChoice"):
    """Same fix as _make_colorbar_hook, for the hover tool: target_classification
    wants its tooltip to show a class name instead of a raw integer code, but a
    HoverTool object passed via .opts(tools=[...]) only ever takes effect on
    the first frame. So this mutates the ALREADY-ATTACHED HoverTool's
    tooltips/formatters directly on every frame instead. The first two
    tooltip entries (time, height) are left exactly as HoloViews originally
    set them up (which includes its own datetime formatting for "time") --
    only the third ("value") entry is ever replaced.
    """
    captured: dict = {}

    def hook(plot, element):
        hover = plot.handles.get("hover")
        if hover is None:
            return
        if "base_tooltips" not in captured:
            captured["base_tooltips"] = list(hover.tooltips[:2])
            captured["base_formatters"] = dict(hover.formatters)
        pv = state.catalog_variable(choice.catalog_id)
        if pv is not None and pv.is_categorical and pv.var_name == "target_classification":
            names = [label.lstrip("_") for label, _ in pm.CATEGORICAL[pv.var_name]]
            hover.tooltips = captured["base_tooltips"] + [("class", "@image")]
            hover.formatters = {**captured["base_formatters"], "@image": CustomJSHover(
                args=dict(names=names), code="return names[Math.round(value)] ?? 'unknown';")}
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

    # A raw Bokeh Select (not pn.widgets.Select, which stringifies a nested
    # dict instead of rendering it) so grouped options render as real HTML
    # <optgroup>s -- ONE native dropdown showing "instrument > variable"
    # sub-groups, rather than two linked widgets or a custom flyout menu.
    var_select = BokehSelect(options=grouped, value=default_id, width=280, height=32, margin=(5, 5, 5, 5))
    choice = _CatalogChoice(catalog_id=default_id)

    def _sync_choice(attr, old, new):
        choice.catalog_id = new

    var_select.on_change("value", _sync_choice)

    auto_color = panel_state["color_limits"] == "auto"
    vmin_w = pn.widgets.FloatInput(name="color min", value=0.0, disabled=auto_color, width=90)
    vmax_w = pn.widgets.FloatInput(name="color max", value=1.0, disabled=auto_color, width=90)
    auto_w = pn.widgets.Checkbox(name="auto color", value=auto_color)
    reset_btn = pn.widgets.Button(name="Reset panel", button_type="light", width=100)

    def on_reset(event):
        settings.reset_panel(index)
        var_select.value = _default_catalog_id(state.catalog, None, DEFAULT_PANEL_VARIABLES[index])
        auto_w.value = True

    reset_btn.on_click(on_reset)

    def sync_categorical_disable(catalog_id):
        pv = state.catalog_variable(catalog_id)
        is_cat = bool(pv and pv.is_categorical)
        auto_w.disabled = is_cat
        vmin_w.disabled = is_cat or auto_w.value
        vmax_w.disabled = is_cat or auto_w.value

    sync_categorical_disable(choice.catalog_id)

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

    choice.param.watch(on_choice_change, "catalog_id")

    # Collapsed by default: a Card's header is itself the toggle (no extra
    # button needed), keeping the panel compact until color limits actually
    # need adjusting. Settings are stacked vertically (rather than one tight
    # row) so the card can stay narrow while still fitting everything without
    # clipping.
    options_card = pn.Card(
        auto_w, pn.Row(vmin_w, vmax_w), reset_btn,
        title="Options", collapsed=True, width=220, margin=(5, 0, 5, 0),
    )
    var_pane = pn.pane.Bokeh(var_select, height=42, margin=(0, 0, 0, 0))
    controls = pn.Row(var_pane, options_card, align="center")

    def refresh(catalog: list[gp.ProductVariable]):
        """Repopulate the dropdown's options from a freshly discovered
        catalog, restoring the stored preference. Unlike Panel's own Select,
        a raw Bokeh Select doesn't auto-reassign .value when .options
        changes, so there's no risk of a transient wrong id overwriting the
        stored preference here."""
        nonlocal grouped
        grouped = _grouped_options(catalog)
        stored_id = settings.get_panel(index)["variable"]
        var_select.options = grouped
        var_select.value = _default_catalog_id(catalog, stored_id, DEFAULT_PANEL_VARIABLES[index])

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


def _no_data_moment(state: AppState, title: str = "no data"):
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
    return hv.Image(((t0, t1), height_arr, np.full((len(height_arr), 2), np.nan)),
                     kdims=["time", "height"], vdims=["value"]).opts(
        colorbar=True, responsive=True, height=260, title=title, apply_ranges=False,
        tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"])


def _moment_image(state: AppState, index: int, catalog_id, auto_color, vmin, vmax):
    """The image only -- no marker (see _moment_marker) -- and its streams
    (in build_app) deliberately do NOT include selected_time/selected_height,
    so clicking to select a point never re-triggers this and never resets
    the axes."""
    pv = state.catalog_variable(catalog_id)
    if pv is None or state.spectra is None:
        return _no_data_moment(state)
    t0, t1 = state.hour_time_bounds
    time_arr, height_arr, values = gp.load_curtain(pv, t0, t1)
    if len(time_arr) == 0:
        return _no_data_moment(state, f"{pv.label} (no data this hour)")

    is_timeseries = height_arr is None
    if is_timeseries:
        # Time-only products (e.g. MWR's LWP) are drawn by _moment_curve
        # instead, as a real line plot on its own y-axis -- a "height" isn't
        # a meaningful axis for them at all. This image is just an invisible
        # placeholder (see _moment_curve for why the panel still needs one:
        # a DynamicMap's own frames must always be the same element type).
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

    title = f"{pv.label} (time series)" if is_timeseries else pv.label
    opts = dict(colorbar=True, tools=["hover", "tap"], active_tools=["tap", "wheel_zoom"],
                responsive=True, height=260, title=title, apply_ranges=False,
                xlabel="time (UTC)", ylabel="" if is_timeseries else "height (m)")
    # colorbar PRESENCE is always True (even for categorical vars, which also
    # get the HTML legend, and for the invisible timeseries placeholder,
    # which _make_colorbar_hook hides via .visible instead) because it's a
    # structural Bokeh property fixed by this DynamicMap's very first frame
    # -- toggling it per-variable would only ever reflect whichever type was
    # selected FIRST. The color mapper's TYPE (linear vs log) has the exact
    # same limitation and can't be set here at all for the same reason --
    # see _make_colorbar_hook, the only place it can actually be changed
    # after the first frame.
    if pv.is_categorical:
        colors = [c for _, c in pm.CATEGORICAL[pv.var_name]]
        opts.update(cmap=colors, clim=(-0.5, len(colors) - 0.5))
    else:
        if auto_color:
            clim = pv.plot_range or (None, None)
        else:
            clim = (vmin, vmax)
        opts.update(cmap=pv.cmap, clim=clim)

    return hv.Image((time_arr, height_arr, values.T), kdims=["time", "height"], vdims=["value"]).opts(**opts)


def _make_curve_axis_hook(state: AppState, choice: "_CatalogChoice"):
    """The row-1 Curve (see _moment_curve) always draws its own right-hand
    y-axis (multi_y=True on the Overlay), even when it's the empty
    placeholder for a non-timeseries variable -- visually crowding the
    colorbar next to it for no reason. Hide that axis except when a
    time-only variable is actually active.

    Also fits that axis's range to the curve's actual data: like every
    other structural Bokeh property in this file, a multi_y extra range is
    only ever auto-fitted from this DynamicMap's first-ever frame (which is
    the empty placeholder, before any real data has loaded) and never
    refitted by HoloViews on later frames -- left alone, it stays stuck at
    Bokeh's Range1d default of (0, 1) forever, squashing the real curve
    into an invisible sliver near the bottom. Always refit (never zoom-
    preservation-gated like _make_range_hook) since this dmap's streams
    don't include selected_time/selected_height, so a click never
    re-renders it -- only a genuine hour/variable change does, and both of
    those should refit this axis to the new data's range anyway.
    """
    def hook(plot, element):
        axis = plot.handles.get("yaxis")
        pv = state.catalog_variable(choice.catalog_id)
        is_timeseries = pv is not None and pv.height_dim is None
        if axis is not None:
            axis.visible = is_timeseries
        fig = plot.handles.get("plot")
        extra_ranges = getattr(fig, "extra_y_ranges", None) if fig is not None else None
        rng = extra_ranges.get("ts_value") if extra_ranges is not None else None
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
    return hv.Curve([], kdims=["time"], vdims=["ts_value"]).opts(yaxis="right")


def _moment_curve(state: AppState, catalog_id):
    """The line-plot counterpart to _moment_image, for time-only variables
    (e.g. MWR's LWP): a real hv.Curve with its OWN y-axis on the right (via
    multi_y=True on the row-1 Overlay, since a 1D time series has no natural
    relationship to the shared "height" axis the other two elements use).
    Always part of the row-1 Overlay, but empty (draws nothing) whenever the
    current variable is a real 2D curtain instead -- this DynamicMap's own
    frames must always be the same element type (a Curve, just sometimes an
    empty one), exactly like image_dmap's own placeholder handling.

    Deliberately does NOT set apply_ranges=False: unlike the shared
    time/height axes, this curve's own value axis never needs zoom
    preserved across clicks (its streams don't include
    selected_time/selected_height, so a click never re-renders it at all --
    only an hour or variable change does, and both of those SHOULD refit
    this axis to the new data's real range, exactly like auto_y=True
    elsewhere in this file).
    """
    pv = state.catalog_variable(catalog_id)
    if pv is None or pv.height_dim is not None or state.spectra is None:
        return _no_data_curve()
    t0, t1 = state.hour_time_bounds
    time_arr, _, values = gp.load_curtain(pv, t0, t1)
    if len(time_arr) == 0:
        return _no_data_curve()
    ylabel = f"{pv.label} ({pv.units})" if pv.units else pv.label
    return hv.Curve((time_arr, values), kdims=["time"], vdims=["ts_value"]).opts(
        yaxis="right", color="darkorange", ylabel=ylabel)


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
        cmap="viridis", colorbar=True, tools=["hover"], apply_ranges=False)
    return hv.Overlay([img]).opts(responsive=True, height=280, apply_ranges=False,
                                   title="Range spectrogram (no data)")


def _range_spectrogram(state: AppState, channel: str):
    """Content legitimately changes on every click (a different time's
    profile); the axes are managed by _make_range_hook, not by HoloViews."""
    s = state.spectra
    if s is None or state.selected_time is None:
        return _no_data_range_spectrogram()
    t_idx = s.nearest_time_index(state.selected_time)
    segments = s.range_profile_segments(t_idx, channel)
    images = [
        hv.Image((vel, s.height[rng_slice], block), kdims=["velocity", "height"], vdims=["power"]).opts(
            cmap="viridis", colorbar=True, tools=["hover"], apply_ranges=False)
        for rng_slice, vel, block in segments
    ]
    return hv.Overlay(images).opts(
        responsive=True, height=280, title=f"Range spectrogram ({channel})", apply_ranges=False,
        xlabel="Doppler velocity (m/s)", ylabel="height (m)",
    )


def _no_data_time_spectrogram(state: AppState):
    """Shares the "time" kdim with the moments/row1 panels, so it must span
    the same window; the axes themselves are managed by _make_range_hook."""
    t0, t1 = state.display_time_bounds
    return hv.Image(((t0, t1), np.array([-1.0, 1.0]), np.full((2, 2), np.nan)),
                     kdims=["time", "velocity"], vdims=["power"]).opts(
        cmap="viridis", colorbar=True, responsive=True, height=280, tools=["hover"],
        title="Time spectrogram (no data)", apply_ranges=False,
        xlabel="time (UTC)", ylabel="Doppler velocity (m/s)")


def _time_spectrogram(state: AppState, channel: str):
    """Shows the full loaded hour (not a narrow window around the selection)
    so its "time" axis has the same natural extent as the moments panels and
    can be genuinely axis-linked with them, rather than fighting over a
    shared range at two different intended zoom levels."""
    s = state.spectra
    if s is None or state.selected_height is None:
        return _no_data_time_spectrogram(state)
    r_idx = s.nearest_range_index(state.selected_height)
    block = s.time_series(r_idx, 0, s.n_time, channel)
    vel = s.chirp.velocity_axis(r_idx)
    times = np.array([np.datetime64(int(x), "s") for x in s.time])
    return hv.Image((times, vel, block.T), kdims=["time", "velocity"], vdims=["power"]).opts(
        cmap="viridis", colorbar=True, tools=["hover"], responsive=True, height=280,
        title=f"Time spectrogram at {s.height[r_idx]:.0f} m ({channel})", apply_ranges=False,
        xlabel="time (UTC)", ylabel="Doppler velocity (m/s)",
    )


def _no_data_spectrum():
    empty_co = hv.Curve(([], []), kdims=["velocity"], vdims=["power"], label="co-polar").opts(color="steelblue")
    empty_cx = hv.Curve(([], []), kdims=["velocity"], vdims=["power"], label="cross-polar").opts(color="firebrick")
    return (empty_co * empty_cx).opts(
        hv.opts.Curve(responsive=True, height=280, tools=["hover"], apply_ranges=False),
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
        hv.opts.Curve(responsive=True, height=280, tools=["hover"], apply_ranges=False),
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
    reset_all_btn = pn.widgets.Button(name="Reset all settings", button_type="warning", width=140, align="end")
    clean_cache_btn = pn.widgets.Button(name="Clean cache", button_type="danger", width=100, align="end")
    status = pn.pane.Markdown("", sizing_mode="stretch_width")
    download_status = pn.pane.Markdown("", sizing_mode="stretch_width")

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
        for w in (load_btn, hour_select, prev_btn, next_btn, site_select, day_input, instrument_select):
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
            # refresh_variable_options() mutates a raw Bokeh Select (not a
            # Panel widget, which handles this safely on its own) -- doing
            # that directly here, after the `await` above resumes this
            # coroutine outside Bokeh's document-lock context, crashes with
            # "_pending_writes should be non-None...". Panel widgets (e.g.
            # hour_select below) don't need this, but _push's
            # add_next_tick_callback puts the raw-model mutation back under
            # a proper document lock. Bundled into one callback (rather than
            # wrapping just that one call) so range_generation still bumps
            # AFTER the variable selection actually settles -- otherwise the
            # axes could refit to stale bounds one frame before the correct
            # variable's data arrives.
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
    reset_all_btn.on_click(lambda e: (settings.reset_all(), pn.state.location.reload()))

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
                return _moment_image(state, idx, ch.catalog_id, aw.value, vn.value, vx.value)
            return _cb

        def make_curve_callback(ch=choice):
            def _cb(**_kwargs):
                return _moment_curve(state, ch.catalog_id)
            return _cb

        # Image, marker and curve are deliberately SEPARATE DynamicMaps: the
        # image/curve only react to hour/variable/color changes, the marker
        # reacts only to the click position, so a click re-renders as little
        # as possible. Axis ranges are owned by the hook (see _make_range_hook).
        image_dmap = hv.DynamicMap(make_image_callback(), streams=[
            tap,
            param_stream(choice, "catalog_id", f"var{i}"),
            param_stream(auto_w, "value", f"auto{i}"),
            param_stream(vmin_w, "value", f"vmin{i}"),
            param_stream(vmax_w, "value", f"vmax{i}"),
            hv.streams.Params(state, ["hour_index", "range_generation"]),
        ]).opts(hv.opts.Image(hooks=[_make_colorbar_hook(state, choice), _make_hover_hook(state, choice)]))
        marker_dmap = hv.DynamicMap(
            lambda **_kw: _moment_marker(state),
            streams=[hv.streams.Params(state, ["selected_time", "selected_height"])],
        )
        # A time-only variable (e.g. MWR's LWP) is drawn by this Curve on its
        # own right-hand y-axis (multi_y=True below) instead of forcing it
        # onto the shared "height" axis, which it has no natural relationship
        # to -- see _moment_curve.
        curve_dmap = hv.DynamicMap(make_curve_callback(), streams=[
            param_stream(choice, "catalog_id", f"curve_var{i}"),
            hv.streams.Params(state, ["hour_index", "range_generation"]),
        ]).opts(hv.opts.Curve(hooks=[_make_curve_axis_hook(state, choice)]))
        tap.add_subscriber(on_tap_factory(i))
        row1_dmaps.append((image_dmap * marker_dmap * curve_dmap).opts(
            hv.opts.Overlay(apply_ranges=False, multi_y=True,
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
    range_dmap = hv.DynamicMap(
        _range_cb,
        streams=[param_stream(channel_toggle, "value", "ch_range"),
                 hv.streams.Params(state, ["hour_index", "selected_time", "range_generation"])],
    ).opts(hv.opts.Overlay(apply_ranges=False,
                            hooks=[_make_range_hook(state, "velocity", "height")]))
    time_dmap = hv.DynamicMap(
        _time_cb,
        streams=[param_stream(channel_toggle, "value", "ch_time"),
                 hv.streams.Params(state, ["hour_index", "selected_height", "range_generation"])],
    ).opts(hv.opts.Image(apply_ranges=False,
                          hooks=[_make_range_hook(state, "time", "velocity", auto_y=True)]))
    spectrum_dmap = hv.DynamicMap(
        _spectrum_cb,
        streams=[hv.streams.Params(state, ["hour_index", "selected_time", "selected_height", "range_generation"])],
    ).opts(hv.opts.Overlay(apply_ranges=False,
                            hooks=[_make_range_hook(state, "velocity", "power", auto_y=True)]))

    grid = hv.Layout(row1_dmaps + [range_dmap, time_dmap, spectrum_dmap]).cols(3).opts(shared_axes=True)
    grid_pane = pn.pane.HoloViews(grid, sizing_mode="stretch_width")

    def legend_html(cat0, cat1, cat2):
        blocks = []
        for catalog_id in (cat0, cat1, cat2):
            pv = state.catalog_variable(catalog_id)
            if pv and pv.is_categorical:
                swatches = "".join(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color};margin-right:4px;border:1px solid #999;"></span>'
                    f'<span style="margin-right:12px;">{label}</span>'
                    for label, color in pm.legend_entries(pv.var_name)
                )
                blocks.append(f'<div style="margin-bottom:4px;"><b>{pv.label}:</b> {swatches}</div>')
        return "".join(blocks)

    legend_pane = pn.bind(legend_html,
                           moment_controls[0][1].param.catalog_id,
                           moment_controls[1][1].param.catalog_id,
                           moment_controls[2][1].param.catalog_id)

    top_bar = pn.Row(site_select, day_input, instrument_select, hour_select, prev_btn, next_btn, load_btn,
                      reset_all_btn, clean_cache_btn, align="end")
    controls_row = pn.Row(*[mc[0] for mc in moment_controls], align="end")

    return pn.Column(
        top_bar,
        controls_row, pn.pane.HTML(legend_pane),
        grid_pane,
        pn.Row(channel_toggle, status, download_status, align="center"),
        sizing_mode="stretch_width",
    )


if pn.state.served:
    build_app().servable(title="PRISM")
