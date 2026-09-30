# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PRISM: a Panel/HoloViews/Bokeh app (package `prism`, source in `src/prism/`) that shows linked time-height products and raw Doppler spectra from the Cloudnet data portal (RPG-FMCW-94, MIRA-10, MIRA-35).

## Commands

```bash
pip install -e ".[test]"   # editable install with test deps (needs netCDF/HDF5; conda-forge is easiest, see README)
prism                      # run the app (--port N, --no-browser)
pytest                     # fast offline unit tests (network tests excluded via addopts)
pytest tests/test_rpg_reader.py::test_name   # single test
pytest -m network          # opt-in: hits real Cloudnet API and downloads real files; don't run in a loop
```

No linter/formatter is configured. Cache: `~/.cache/prism` (`PRISM_CACHE`); settings: `~/.config/prism/settings.json` (`PRISM_SETTINGS`).

## Architecture

- `cli.py` -> `app.build_app()` builds the 6-panel Panel template. `app.py` (~1700 lines) holds everything UI: `AppState` (a `param.Parameterized` that all widgets read/write: selected hour, click point, catalog lookups) plus per-panel render functions and Bokeh hooks. Clicking a moments panel sets time/height on `AppState` and every other panel updates from it.
- `cloudnet_client.py`: Cloudnet API queries and cached downloads.
- Spectra readers (`rpg_reader.py`, `mira_reader.py`) decode instrument files into the **same zarr schema** (dB, uint8-quantized at 0.3 dB, one `byTime` and one `byRange` chunking for fast profile vs. time-series reads, plus a `ChirpTable`). This is what lets `SpectraHour` and all of `app.py` be instrument-agnostic; MIRA is modeled as a single-chirp table. A new instrument should write this schema rather than touch app.py.
- `rpg_reader._repair_truncated_tail` recovers the good prefix of RPG files that rpgpy rejects for corrupt trailing timestamps by scanning LV0 record layout directly.
- `gridded_products.py`: discovers time-height ("curtain") and time-only variables per site/day. Two-speed: `discover()` eagerly loads only products used by the current panels; all others become stub `ProductVariable`s from the hardcoded `PRODUCT_SCHEMA` and are resolved on demand (`resolve_product`, `AppState.resolve_variable`) when selected.
- `plot_meta.py`: default colormaps/ranges/legends per bare variable name, ported from cloudnetpy.
- `settings_store.py`: persisted per-panel state (`"auto"` means compute from the data).

## Gotchas

- `panel==1.9.4` and `bokeh==3.8.2` are pinned together deliberately; mismatched versions make click/tap events silently stop reaching the server. Bump both together and re-verify clicking.
- Several Bokeh properties (colorbar formatter/ticker/width, hover tooltips, linear-vs-log color mapper type) are frozen by a `DynamicMap`'s first frame and ignored by later `.opts()`. `_make_colorbar_hook` / `_make_hover_hook` mutate the Bokeh models directly each frame; use that pattern instead of `.opts()` for these.
- `tests/test_app_state.py` tests `AppState` only, deliberately not the Panel widget callbacks (use Playwright against a served app if wiring bugs need testing).
- Model heights are shown above sea level, and single-pol radars are forced to co-polar; cross-polar may be empty for some instruments (e.g. MIRA-10).
