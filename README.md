# PRISM

**P**rofiling & **R**emote-sensing **I**nteractive **S**pectra **M**onitor.

Interactive 6-panel viewer for ground-based remote-sensing time-height
products and raw Doppler spectra, currently sourced from the
[Cloudnet](https://cloudnet.fmi.fi/) data portal (RPG-FMCW-94, MIRA-10,
MIRA-35). Pick a site, date, instrument and hour; it downloads and caches the
raw spectra and processed products, then shows linked, zoomable time-height
panels plus range/time spectrograms and a single Doppler spectrum, all
connected by clicking anywhere on a moments panel.

## Install

Requires Python >= 3.10. `rpgpy` and `netCDF4` need a C compiler and the
netCDF/HDF5 libraries available at install time -- on a machine with conda
this is usually already the case; otherwise install those via your system
package manager or conda first.

```bash
git clone <this-repo-url>
cd cloudnetSpecViewer
pip install .
```

For development (editable install, so code edits take effect without
reinstalling):

```bash
pip install -e .
```

## Run

```bash
prism
```

This opens the viewer in your default browser. Data is downloaded from the
Cloudnet API on demand and cached locally (default `~/.cache/prism`, override
with the `PRISM_CACHE` env var). Settings (selected variables, color limits,
last site/date/instrument/hour) persist across restarts in
`~/.config/prism/settings.json` (override with `PRISM_SETTINGS`).

Options:

```bash
prism --port 8080     # fixed port instead of auto-picked
prism --no-browser    # print the URL instead of opening a tab
```

## Usage

1. Pick a **site**, **date** and **spectra instrument** (RPG-FMCW-94,
   MIRA-10 or MIRA-35), click **Load**. The hour dropdown fills in with only
   the hours that actually have raw Doppler spectra for that instrument.
   Not every instrument publishes raw spectra at every site -- MIRA-35, in
   particular, has only ever been observed publishing moments or PPI wind
   scans, never a vertical Doppler spectrum, across every site checked so
   far (Munich, Bucharest, Lindenberg, Melpitz).
2. Pick an hour; use the prev/next-hour buttons to step through the day.
3. Click anywhere on a moments panel (top row) to select a time/height point
   -- all other panels update to match, and axes are linked (zooming one
   time/height/velocity axis zooms every panel sharing that axis).
4. Each moments panel's dropdown lists every time-height curtain and
   time-only field Cloudnet published for that site/day, grouped by
   instrument (radar moments, target classification, lidar backscatter,
   model temperature, MWR liquid water path, etc.), not just a fixed list.
   A time-only field (like MWR's LWP) is drawn as a line plot on its own
   right-hand axis rather than forced onto the shared height axis.
5. The two spectrogram panels have a co-polar/cross-polar toggle; the single
   spectrum panel always shows both overlaid. Not every instrument publishes
   a cross-polar spectrum -- MIRA-10's archived spectra, for example, are
   co-polar only, so the cross-polar view is empty for that instrument.
6. **Clean cache** clears downloaded/decoded data (not your settings) if you
   want to free disk space or force a re-download.

## Development notes

`panel` and `bokeh` are version-pinned together in `pyproject.toml`: panel
bundles its own BokehJS build, and an installed `bokeh` newer than what that
panel version bundles makes the browser and server speak mismatched protocol
versions. Symptom: the app loads and looks fine, but clicks/taps silently do
nothing with no error anywhere. If you bump one, bump both together and
re-verify clicking works.

Several Bokeh model properties -- colorbar formatter/ticker/width, a hover
tool's tooltips, and a color mapper's linear-vs-log TYPE -- are fixed by a
`DynamicMap`'s first-ever rendered frame and are never re-applied by
HoloViews' own `.opts()` on later frames. Switching a panel's variable after
that first frame can't change any of these through the normal options API;
`app.py`'s `_make_colorbar_hook` / `_make_hover_hook` work around this by
mutating the already-attached Bokeh models directly on every frame via
HoloViews' `hooks=[...]` mechanism instead.
