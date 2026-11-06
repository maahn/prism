"""What a hostile client or a misbehaving upstream can do to a running server.

The aim is that, in the worst case, someone can fill the disk with cached data
(which is what the app is for) and nothing else: no deleting or overwriting
anything outside the cache, no reading outside it, no taking the server down
by exhausting its memory. Each test pins one way that was closed.
"""
import threading
import time
from datetime import date

import pytest

from prism import cloudnet_client as cc
from prism import server_mode as sm
from prism import settings_store as ss
from prism.app import AppState


# --- path components ------------------------------------------------------

@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "/etc", "a\\b", "a\0b"])
def test_unsafe_path_components_are_rejected(bad):
    with pytest.raises(ValueError):
        cc.check_path_component(bad, "x")


@pytest.mark.parametrize("ok", ["ny-alesund", "20240110_ny-alesund_categorize.nc", "260211_080002_P09_ZEN.LV0",
                                "mirac-a_20210120000001_P01_ZEN.lv0", "20260916_200006.znc.gz"])
def test_real_names_are_accepted(ok):
    assert cc.check_path_component(ok, "x") == ok


def _remote(filename="20240110_ny-alesund_categorize.nc", instrument_id=None, kind="product"):
    return cc.RemoteFile(uuid="u", filename=filename, size=3, checksum="c", download_url="http://example.invalid",
                         instrument_id=instrument_id, kind=kind, product_id="categorize", day=date(2024, 1, 10))


@pytest.mark.parametrize("site", ["../../../tmp/evil", "/tmp/evil", "..", ""])
def test_a_site_cannot_escape_the_cache_or_the_data_archive(tmp_path, site):
    # With a data-archive pattern the site is substituted straight into the path.
    with pytest.raises(ValueError):
        cc.site_cache_dir(site, tmp_path)
    with pytest.raises(ValueError):
        cc.site_cache_dir(site, None, None, "/archive/{site_flat}/cloudnet/{product}/{year}/{file}")


@pytest.mark.parametrize("remote", [
    _remote("../../outside.nc"),
    _remote("/etc/passwd"),
    _remote("20240110_x_ZEN.LV0", instrument_id="../..", kind="raw"),
])
def test_a_hostile_api_response_cannot_pick_where_a_file_is_written(tmp_path, remote):
    with pytest.raises(ValueError):
        cc.cache_path_for(remote, tmp_path)


def test_decoded_store_name_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        cc.zarr_path_for(tmp_path / "raw" / "x.LV0", tmp_path, "../../../evil")


# --- download size cap ----------------------------------------------------

class _FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_a_server_sending_more_than_the_expected_size_cannot_fill_the_disk(tmp_path, monkeypatch):
    remote = _remote()  # the portal says 3 bytes
    monkeypatch.setattr(cc.requests, "get", lambda *a, **k: _FakeResponse([b"ab", b"c" * 10_000_000]))
    with pytest.raises(ConnectionError, match="more than the expected"):
        cc.ensure_downloaded(remote, tmp_path)
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in leftovers) <= 3  # nothing like the 10 MB offered


def test_an_exact_size_download_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(cc.requests, "get", lambda *a, **k: _FakeResponse([b"ab", b"c"]))
    assert cc.ensure_downloaded(_remote(), tmp_path).read_bytes() == b"abc"


# --- "Clean cache" ----------------------------------------------------------

def _build(monkeypatch, tmp_path, **kwargs):
    import panel as pn
    from prism import app

    monkeypatch.setattr(pn.state, "onload", lambda *a, **k: None)  # no server: skip the auto-load
    monkeypatch.setattr(cc, "list_sites", lambda: (_ for _ in ()).throw(RuntimeError("no network in tests")))
    monkeypatch.setattr(ss, "DEFAULT_SETTINGS_PATH", tmp_path / "settings.json")
    calls = []
    monkeypatch.setattr(cc, "clear_data_cache", lambda path: calls.append(path))
    layout = app.build_app(**kwargs)
    button = next(b for b in layout.select(pn.widgets.Button) if b.name == "Clean cache")
    return button, calls


def test_clean_cache_cannot_be_triggered_by_a_client_in_server_mode(monkeypatch, tmp_path):
    # The button is only hidden in server mode, and hiding is not access
    # control: a forged click used to delete the whole SHARED cache.
    button, calls = _build(monkeypatch, tmp_path, server_mode=True)
    assert button.visible is False
    button._process_events({"clicks": 1})
    assert calls == []


def test_clean_cache_cannot_be_triggered_on_a_configured_data_archive(monkeypatch, tmp_path):
    config = {"path_pattern": "/archive/{site_flat}/cloudnet/{product}/{year}/{file}"}
    button, calls = _build(monkeypatch, tmp_path, config=config)
    button._process_events({"clicks": 1})
    assert calls == []


def test_clean_cache_still_works_for_a_single_user_with_the_default_cache(monkeypatch, tmp_path):
    # the control: without it the two tests above could pass for the wrong reason
    button, calls = _build(monkeypatch, tmp_path)
    button._process_events({"clicks": 1})
    assert len(calls) == 1


# --- raw files in a data archive ---------------------------------------------

class _FakeDecoder:
    @staticmethod
    def _zarr_cache_path(local, cache_dir):
        return cache_dir / "spectra_zarr" / f"{local.stem}.zarr"

    @staticmethod
    def ensure_decoded(local, cache_dir, on_status=None):
        out = _FakeDecoder._zarr_cache_path(local, cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "_SUCCESS").write_text("ok")
        return out


def _state(tmp_path, monkeypatch, **settings_data):
    settings = ss.Settings(path=None, overrides=settings_data or None)
    state = AppState(settings)
    state.site, state.instrument = "hyytiala", "rpg-fmcw-94"
    monkeypatch.setattr(state, "_decoder_for_instrument", lambda: _FakeDecoder)
    return state


def test_a_raw_file_that_was_already_in_the_archive_is_not_deleted(tmp_path, monkeypatch):
    root = tmp_path / "archive" / "hyytiala"
    config = {"path_pattern": str(tmp_path / "archive" / "{site_flat}" / "{product}" / "{year}" / "{file}")}
    state = _state(tmp_path, monkeypatch, **config)
    remote = cc.RemoteFile(uuid="u", filename="260211_080002_P09_ZEN.LV0", size=4, checksum="c",
                           download_url="http://example.invalid", instrument_id="rpg-fmcw-94", kind="raw",
                           day=date(2026, 2, 11))
    raw = cc.cache_path_for(remote, root)
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"data")  # the archive's own copy, at exactly the path PRISM would use

    state._ensure_hour_decoded(remote)

    assert raw.read_bytes() == b"data"


def test_a_raw_file_in_prisms_own_cache_is_deleted_after_decoding(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch, cache_dir=str(tmp_path / "cache"))
    state.settings._data["cache_dir"] = None  # PRISM's own cache, not a configured archive
    monkeypatch.setattr(AppState, "cache_dir", property(lambda self: tmp_path / "cache"))
    remote = cc.RemoteFile(uuid="u", filename="260211_080002_P09_ZEN.LV0", size=4, checksum="c",
                           download_url="http://example.invalid", instrument_id="rpg-fmcw-94", kind="raw",
                           day=date(2026, 2, 11))
    raw = cc.cache_path_for(remote, tmp_path / "cache")
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"data")

    state._ensure_hour_decoded(remote)

    assert not raw.exists()


# --- memory -------------------------------------------------------------------

def test_decodes_are_limited_so_several_users_cannot_exhaust_the_memory():
    sm.set_decode_limit(1)
    running, peak, lock = 0, 0, threading.Lock()

    def decode():
        nonlocal running, peak
        with sm.decode_slot():
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with lock:
                running -= 1

    threads = [threading.Thread(target=decode) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak == 1

    sm.set_decode_limit(2)
    peak = 0
    threads = [threading.Thread(target=decode) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak == 2
    sm.set_decode_limit(sm.DEFAULT_MAX_DECODES)
