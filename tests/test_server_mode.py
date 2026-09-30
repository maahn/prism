"""Server mode (`prism --server`) shares one process and one cache directory
between many user sessions. These cover the pieces that make that safe:
idle-session expiry/teardown, in-memory per-session settings, and locking
around the shared cache. Like test_app_state.py, they test the logic, not the
Panel widget wiring (see tests/README.md).
"""
import threading
import time

import pytest

from prism import cloudnet_client as cc
from prism import server_mode as sm
from prism.app import AppState
from prism.cli import SERVER_DEFAULT_PORT, parse_args
from prism.settings_store import Settings


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


def _lifecycle(clock, timeout=30 * 60):
    released = []
    life = sm.SessionLifecycle(timeout, released.append, clock=clock)
    return life, released


def test_session_expires_after_idle_timeout(clock):
    life, released = _lifecycle(clock)
    clock.now += 30 * 60 - 1
    assert life.check() is False
    assert released == []
    clock.now += 1
    assert life.check() is True
    assert released == [True]  # released with expired=True: the browser is still there
    assert life.expired


def test_activity_resets_the_idle_clock(clock):
    life, released = _lifecycle(clock)
    for _ in range(5):  # 5 x 20 min = well past the limit, but never 30 min idle
        clock.now += 20 * 60
        life.touch()
        assert life.check() is False
    assert released == []


def test_destroyed_session_is_released_once_and_not_marked_expired(clock):
    life, released = _lifecycle(clock)
    life.close()  # what Bokeh's session-destroyed hook triggers
    life.close()
    clock.now += 3600
    life.check()  # a late idle check must not release a second time
    assert released == [False]
    assert not life.expired


def test_no_timeout_never_expires(clock):
    life, released = _lifecycle(clock, timeout=None)
    clock.now += 10 * 24 * 3600
    assert life.check() is False
    assert released == []
    life.close()
    assert released == [False]


def test_touch_after_close_is_ignored(clock):
    life, _ = _lifecycle(clock)
    life.close()
    before = life.last_active
    clock.now += 100
    life.touch()
    assert life.last_active == before


def test_failing_release_still_counts_the_session_as_closed(clock):
    def boom(expired):
        raise RuntimeError("release failed")

    before = sm.live_session_count()
    life = sm.SessionLifecycle(60, boom, clock=clock)
    assert sm.live_session_count() == before + 1
    life.close()  # must swallow the error
    assert sm.live_session_count() == before


def test_shared_dataset_cache_is_only_dropped_when_last_session_closes(clock, monkeypatch):
    cleared = []
    monkeypatch.setattr(sm.gp, "clear_dataset_cache", lambda: cleared.append(1))
    baseline = sm.live_session_count()
    a, _ = _lifecycle(clock)
    b, _ = _lifecycle(clock)
    a.close()
    if baseline == 0:
        assert cleared == []  # b is still using the shared datasets
    b.close()
    if baseline == 0:
        assert cleared == [1]


def test_settings_without_a_path_never_touch_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(path=None)
    s.set("spectra_channel", "cross")
    s.set_panel(0, variable="radar:Zh")
    s.set_last_session("juelich", "2021-01-20", 14, "mira-10")
    s.reset_all()
    assert list(tmp_path.iterdir()) == []


def test_in_memory_settings_are_independent_between_sessions():
    a, b = Settings(path=None), Settings(path=None)
    a.set_panel(0, variable="radar:Zh")
    a.set("spectra_channel", "cross")
    assert b.get_panel(0)["variable"] is None
    assert b.get("spectra_channel") == "co"


def test_release_drops_loaded_data():
    state = AppState(Settings(path=None))
    state.spectra = object()
    state.catalog = [object()]
    state.available_hours = [object()]
    state.range_keys["x"] = 1
    state.release()
    assert state.spectra is None and state.catalog == [] and state.available_hours == []
    assert state.range_keys == {}


def test_load_finishing_after_release_does_not_repopulate(monkeypatch, tmp_path):
    # The idle timer can fire while a background load is still running
    # (e.g. "Cache entire day"); its result must be discarded, not kept.
    state = AppState(Settings(path=None))
    state.available_hours = [cc.RemoteFile(
        uuid="u", filename="260216_130000_P09_ZEN.LV0", size=1, checksum="ab", download_url="x",
        instrument_id="rpg-fmcw-94", kind="raw")]

    def slow_decode(remote, **kwargs):
        state.release()  # session expires while the decode is in flight
        return tmp_path

    monkeypatch.setattr(state, "_ensure_hour_decoded", slow_decode)
    state.hour_index = 13
    state.load_hour()
    assert state.spectra is None


def test_path_lock_serializes_and_cleans_up():
    inside = []
    overlap = []

    def worker():
        with cc.path_lock("some/file"):
            inside.append(1)
            if len(inside) > 1:
                overlap.append(True)
            time.sleep(0.02)
            inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert overlap == []
    assert cc._path_locks == {}  # no entry left behind per file ever touched


def test_path_lock_is_reentrant():
    with cc.path_lock("a"):
        with cc.path_lock("a"):
            pass


def test_concurrent_sessions_download_a_file_once(tmp_path, monkeypatch):
    # Two sessions asking for the same hour must not both stream into the
    # same ".part" file.
    payload = b"x" * 5000
    remote = cc.RemoteFile(uuid="u", filename="f.lv0", size=len(payload), checksum="ab" * 16,
                           download_url="http://example.invalid/f", instrument_id="rpg-fmcw-94", kind="raw")
    calls = []

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            time.sleep(0.05)  # widen the window in which a second caller would collide
            yield payload

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(cc.requests, "get", fake_get)
    results = []
    threads = [threading.Thread(target=lambda: results.append(cc.ensure_downloaded(remote, tmp_path)))
               for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1
    assert len(results) == 4 and all(p.read_bytes() == payload for p in results)


def test_cli_defaults_to_local_mode():
    args = parse_args([])
    assert not args.server and args.idle_timeout == 30 and args.allow_websocket_origin == []


def test_cli_server_options():
    args = parse_args(["--server", "--idle-timeout", "5", "--allow-websocket-origin", "a.org",
                       "--allow-websocket-origin", "b.org:80", "--address", "127.0.0.1"])
    assert args.server and args.idle_timeout == 5 and args.address == "127.0.0.1"
    assert args.allow_websocket_origin == ["a.org", "b.org:80"]
    assert SERVER_DEFAULT_PORT == 5006


def test_cli_rejects_negative_idle_timeout():
    with pytest.raises(SystemExit):
        parse_args(["--server", "--idle-timeout", "-1"])
