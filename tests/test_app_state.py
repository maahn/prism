"""AppState is what every widget in app.py actually reads and writes --
site_select, hour_select, the variable dropdowns, and their callbacks are
all thin wiring around it (see build_app() in src/prism/app.py). Testing
build_app()'s closures directly would mean driving Panel's async widget
machinery (asyncio event loop scheduling, Bokeh document locks) with no
real browser or server, which is heavy and brittle for what it buys.
Testing AppState instead covers the actual decision logic behind the GUI
-- which hour a click resolves to, which catalog entry a dropdown choice
resolves to -- without any of that.
"""
import pytest

from prism import gridded_products as gp
from prism import mira_reader as mr
from prism import rpg_reader as rr
from prism.app import AppState
from prism.cloudnet_client import RemoteFile
from prism.settings_store import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(path=tmp_path / "settings.json")


def test_state_loads_last_session_from_settings(settings):
    settings.set_last_session("juelich", "2021-01-20", 14, "mira-10")
    state = AppState(settings)
    assert (state.site, state.day.isoformat(), state.hour_index, state.instrument) == (
        "juelich", "2021-01-20", 14, "mira-10")


def test_state_defaults_instrument_for_settings_predating_it(settings, tmp_path):
    # An older settings.json (written before the instrument selector existed)
    # has no "instrument" key in last_session -- must fall back to RPG rather
    # than raise a KeyError on every app start for a returning user.
    import json
    path = tmp_path / "old_settings.json"
    path.write_text(json.dumps({"last_session": {"site": "hyytiala", "day": "2024-02-16", "hour": 13}}))
    state = AppState(Settings(path=path))
    assert state.instrument == "rpg-fmcw-94"


def _remote(filename, instrument_id="rpg-fmcw-94"):
    return RemoteFile(uuid=filename, filename=filename, size=1, checksum="x",
                       download_url="http://example.invalid", instrument_id=instrument_id, kind="raw")


def test_hours_with_data_uses_hour_of_filename(settings):
    state = AppState(settings)
    state.available_hours = [
        _remote("joyrad94_20240110000001_P01_ZEN.lv0"),
        _remote("joyrad94_20240110130000_P01_ZEN.lv0"),
        _remote("joyrad94_20240110230000_P01_ZEN.lv0"),
    ]
    # This is the exact aggregation that silently collapsed to {20} for
    # Ny-Alesund before hour_of_filename's regex rewrite -- see
    # tests/test_cloudnet_client.py for the parsing fix itself.
    assert state.hours_with_data() == {0, 13, 23}


def test_remote_for_hour_picks_the_matching_file(settings):
    state = AppState(settings)
    r0 = _remote("joyrad94_20240110000001_P01_ZEN.lv0")
    r13 = _remote("joyrad94_20240110130000_P01_ZEN.lv0")
    state.available_hours = [r0, r13]
    assert state._remote_for_hour(13) is r13
    assert state._remote_for_hour(0) is r0
    assert state._remote_for_hour(5) is None  # no file for that hour


def test_decoder_for_instrument_dispatches_by_instrument(settings):
    state = AppState(settings)
    for instrument in ("mira-10", "mira-35"):
        state.instrument = instrument
        assert state._decoder_for_instrument() is mr
    state.instrument = "rpg-fmcw-94"
    assert state._decoder_for_instrument() is rr


def _pv(catalog_id, file_path=None, product_id="radar", var_name="Zh", instrument_id="rpg-fmcw-94"):
    return gp.ProductVariable(catalog_id=catalog_id, product_id=product_id, var_name=var_name,
                               label="radar: Radar reflectivity factor", units="dBZ",
                               file_path=file_path, instrument_id=instrument_id, height_dim="height")


def test_catalog_variable_looks_up_by_id_only(settings):
    state = AppState(settings)
    pv = _pv("radar:Zh#rpg-fmcw-94")
    state.catalog = [pv]
    assert state.catalog_variable("radar:Zh#rpg-fmcw-94") is pv
    assert state.catalog_variable("radar:v#rpg-fmcw-94") is None


def test_resolve_variable_leaves_a_real_entry_alone(settings, tmp_path):
    state = AppState(settings)
    pv = _pv("radar:Zh#rpg-fmcw-94", file_path=tmp_path / "real.nc")
    state.catalog = [pv]
    assert state.resolve_variable("radar:Zh#rpg-fmcw-94") is pv  # no gp.resolve_product call needed


def test_resolve_variable_downloads_a_stub_exactly_once(settings, tmp_path, monkeypatch):
    # A stub (file_path=None) is what discover() hands back for a product
    # that wasn't eagerly fetched -- resolve_variable is the one place that
    # should ever trigger a download for it (see its own docstring).
    stub = _pv("radar:Zh#rpg-fmcw-94", file_path=None)
    state = AppState(settings)
    state.catalog = [stub]

    resolved = _pv("radar:Zh#rpg-fmcw-94", file_path=tmp_path / "resolved.nc")
    calls = []

    def fake_resolve_product(pv, catalog, cache_dir, on_progress=None):
        calls.append(pv.catalog_id)
        return [resolved if p.catalog_id == pv.catalog_id else p for p in catalog]

    monkeypatch.setattr(gp, "resolve_product", fake_resolve_product)
    out = state.resolve_variable("radar:Zh#rpg-fmcw-94")
    assert out is resolved
    assert calls == ["radar:Zh#rpg-fmcw-94"]
    assert state.catalog_variable("radar:Zh#rpg-fmcw-94") is resolved  # catalog was updated in place


def test_resolve_variable_unknown_id_returns_none_without_resolving(settings, monkeypatch):
    state = AppState(settings)
    state.catalog = []
    monkeypatch.setattr(gp, "resolve_product", lambda *a, **k: pytest.fail("should not be called"))
    assert state.resolve_variable("does-not-exist") is None


def test_hour_time_bounds_none_until_spectra_loaded(settings):
    state = AppState(settings)
    assert state.spectra is None
    assert state.hour_time_bounds == (None, None)


def test_display_time_bounds_falls_back_to_selected_hour_placeholder(settings):
    import datetime as dt
    import numpy as np
    state = AppState(settings)
    state.day = dt.date(2024, 1, 10)
    state.hour_index = 13
    t0, t1 = state.display_time_bounds
    assert t0 == np.datetime64(dt.date(2024, 1, 10)) + np.timedelta64(13, "h")
    assert t1 - t0 == np.timedelta64(1, "h")


def test_load_hour_propagates_decode_failure_without_mutating_spectra(settings, monkeypatch):
    """A real Ny-Alesund file (2024-01-10, hour 11) has a corrupted embedded
    timestamp that rpgpy itself rejects with RPGFileError -- confirmed
    directly against the downloaded file, so it's a genuine upstream data
    problem, not something we can decode around. Before the app.py fix
    (do_load/on_hour_change catching only ConnectionError), this exception
    propagated all the way out of the async callback uncaught: Panel just
    logged it to the server console, and set_controls_disabled(False) never
    ran, leaving every control disabled and "Loading..." on screen until the
    page was reloaded. The fix depends on load_hour() actually letting this
    exception through (rather than swallowing it) so do_load/on_hour_change's
    now-broadened `except Exception` can catch it and recover -- this is
    the contract that regresses if load_hour ever grows a bare try/except
    of its own."""
    import rpgpy

    state = AppState(settings)
    state.available_hours = [_remote("joyrad94_20240110110000_P01_ZEN.lv0")]
    previous_spectra = object()
    state.spectra = previous_spectra  # simulate a previously-loaded hour

    def boom(*a, **k):
        raise rpgpy.RPGFileError(
            "Timestamp 940908380 is outside the expected range [726577200, 726580798].")

    monkeypatch.setattr(state, "_ensure_hour_decoded", boom)
    state.hour_index = 11

    with pytest.raises(rpgpy.RPGFileError):
        state.load_hour()

    # The prior hour's data must still be there for the UI to fall back to
    # (or at least not be silently cleared) -- not left half-updated.
    assert state.spectra is previous_spectra


def test_effective_channel_falls_back_to_co_without_a_cross_channel():
    # A saved "cross" preference on a single-pol radar (Ny-Alesund, Julich)
    # used to render completely blank spectrograms -- the cross arrays are
    # fully masked there.
    from prism.app import _effective_channel

    class Hour:
        def __init__(self, has_cross):
            self.has_cross = has_cross

    assert _effective_channel(Hour(False), "cross") == ("co", ", no cross")
    assert _effective_channel(Hour(True), "cross") == ("cross", "")
    assert _effective_channel(Hour(False), "co") == ("co", "")
