# Tests

```bash
pip install -e ".[test]"
pytest
```

That runs the fast unit tests only (no network access, no downloads) --
they mock `rpgpy.read_rpg` and the Cloudnet API, and finish in a second or
two. This is what CI / a pre-commit check should run.

`tests/test_app_state.py` is the GUI-adjacent layer: it exercises
`AppState` (src/prism/app.py), which every widget in `build_app()` reads
and writes -- which hour a click resolves to, which catalog entry a
dropdown choice resolves to, stub-vs-resolved product lookups. It
deliberately does NOT drive the actual Panel widgets/callbacks
end-to-end: those are async closures wired to Bokeh's document-locked
event loop, and driving them without a real browser/server is heavy and
brittle for what it buys. If a bug ever turns out to be in the wiring
itself (a widget calling the wrong callback, say) rather than in
`AppState`, that's the point to reach for Playwright against a running
`panel serve` instead of extending this file.

## Network smoke test (opt-in)

`tests/test_network_smoke.py` hits the real Cloudnet API and downloads real
raw-spectra files for a handful of site/instrument/date combinations, then
runs them through the full decode pipeline. It is the only test that
exercises real-world filename/data quirks end-to-end -- both production
bugs fixed on 2026-09-25 (a `KeyError('HSpec')` crash on a single-pol RPG
radar, and raw files silently collapsing onto a single hour) were only
visible this way, not from unit tests of the surrounding logic.

It's marked `@pytest.mark.network` and excluded by default (see
`addopts` in `pyproject.toml`), because it downloads real files -- avoid
running it repeatedly / in a tight loop. Run it explicitly:

```bash
pytest -m network
```

Run it whenever a new site or instrument is added to the app, or
periodically to catch a site/instrument combination we haven't seen yet.
