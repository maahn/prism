"""Shared data directories: per-site roots, <product>/<YYYY>/ layout, reuse of
existing archives (flat or yearly), and the --config file."""
import json
from datetime import date
from pathlib import Path

import pytest

from prism import cloudnet_client as cc
from prism import settings_store as ss


def _product(filename, size=10, product_id="categorize", instrument_id=None, day=date(2024, 1, 5)):
    return cc.RemoteFile(uuid="u", filename=filename, size=size, checksum="c", download_url="http://x.invalid",
                         instrument_id=instrument_id, kind="product", product_id=product_id,
                         site="hyytiala", day=day)


def test_site_cache_dir_variants(tmp_path):
    assert cc.site_cache_dir("hyytiala", tmp_path) == tmp_path / "sites" / "hyytiala"
    pattern = str(tmp_path / "{site_flat}" / "cloudnet") + "/{product}/{year}/{file}"
    assert cc.site_cache_dir("hyytiala", None, None, pattern) == tmp_path / "hyytiala" / "cloudnet"
    assert cc.site_cache_dir("ny-alesund", None, None, pattern) == tmp_path / "nyalesund" / "cloudnet"
    # site_dirs wins over the pattern, for folders that don't follow it
    assert cc.site_cache_dir("ny-alesund", None, {"ny-alesund": "/x/{product}/{year}/{file}"}, pattern) == Path("/x")
    assert cc.site_cache_dir("mosaic", None, None, str(tmp_path / "{site}") + "/{product}/{year}/{file}") == tmp_path / "mosaic"
    dirs = {"ny-alesund": str(tmp_path / "nyaalesund" / "cloudnet") + "/{product}/{year}/{file}"}
    assert cc.site_cache_dir("ny-alesund", None, dirs) == tmp_path / "nyaalesund" / "cloudnet"
    assert cc.site_cache_dir("hyytiala", tmp_path, dirs) == tmp_path / "sites" / "hyytiala"


@pytest.mark.parametrize("filename, expected", [
    ("20240101_hyytiala_categorize.nc", "categorize"),
    ("20240101_ny-alesund_rpg-fmcw-94_f7d44dc8.nc", "rpg-fmcw-94"),
    ("20240101_ny-alesund_hatpro-multi.nc", "hatpro-multi"),
    ("20240110_ny-alesund_ecmwf.nc", "ecmwf"),
    ("20231222_hyytiala_weather-station_80082867.nc", "weather-station"),
])
def test_product_dir_name_matches_existing_archive_folders(filename, expected):
    assert cc.product_dir_name(_product(filename)) == expected


def test_cache_path_always_has_year_folder(tmp_path):
    p = cc.cache_path_for(_product("20240105_hyytiala_categorize.nc"), tmp_path)
    assert p == tmp_path / "categorize" / "2024" / "20240105_hyytiala_categorize.nc"
    raw = cc.RemoteFile(uuid="u", filename="240105_130000_P09_ZEN.LV0", size=1, checksum="c",
                        download_url="x", instrument_id="rpg-fmcw-94", kind="raw", day=date(2024, 1, 5))
    assert cc.cache_path_for(raw, tmp_path) == tmp_path / "raw" / "rpg-fmcw-94" / "2024" / raw.filename
    assert cc.zarr_path_for(cc.cache_path_for(raw, tmp_path), tmp_path, "240105_130000_P09_ZEN") == (
        tmp_path / "spectra_zarr" / "rpg-fmcw-94" / "2024" / "240105_130000_P09_ZEN.zarr")


@pytest.mark.parametrize("relpath", [
    "rpg-fmcw-94/20240105_hyytiala_rpg-fmcw-94.nc",          # flat folder
    "rpg-fmcw-94/2024/20240105_hyytiala_rpg-fmcw-94.nc",     # yearly folder
    "my_own_name/2024/20240105_hyytiala_rpg-fmcw-94.nc",     # hand-named folder
])
def test_existing_archive_file_is_reused_without_download(tmp_path, monkeypatch, relpath):
    existing = tmp_path / relpath
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"x" * 10)
    monkeypatch.setattr(cc.requests, "get", lambda *a, **k: pytest.fail("must not download"))
    remote = _product("20240105_hyytiala_rpg-fmcw-94.nc", product_id="radar", instrument_id="rpg-fmcw-94")
    assert cc.ensure_downloaded(remote, tmp_path) == existing


def test_existing_file_with_wrong_size_is_not_reused(tmp_path):
    f = tmp_path / "categorize" / "20240105_hyytiala_categorize.nc"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x" * 3)  # truncated
    assert cc.find_existing(_product("20240105_hyytiala_categorize.nc"), tmp_path) is None


def test_load_config_validates(tmp_path):
    good = tmp_path / "c.json"
    good.write_text(json.dumps({"site_dirs": {"hyytiala": "/data/h/{product}/{year}/{file}"}}))
    assert ss.load_config(good) == {"site_dirs": {"hyytiala": "/data/h/{product}/{year}/{file}"}}
    for bad in ('{"cache": "x"}', '[]', '{"site_dirs": {"a": 1}}', '{"site_dirs": {"a": "/data/a"}}', "not json"):
        f = tmp_path / "bad.json"
        f.write_text(bad)
        with pytest.raises(ValueError):
            ss.load_config(f)
    with pytest.raises(ValueError):
        ss.load_config(tmp_path / "missing.json")


def test_config_overrides_settings_file_and_is_not_persisted(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"cache_dir": "/from/file", "spectra_channel": "cross"}))
    s = ss.Settings(path, overrides={"site_dirs": {"hyytiala": "/shared/{product}/{year}/{file}"}})
    assert s.get("site_dirs") == {"hyytiala": "/shared/{product}/{year}/{file}"}
    assert s.get("cache_dir") == "/from/file"
    assert s.data_dirs_configured
    s.set("spectra_channel", "co")
    assert "site_dirs" not in json.loads(path.read_text())
    assert not ss.Settings(path=None).data_dirs_configured


def test_keep_existing_never_replaces_an_archive_file(tmp_path, monkeypatch):
    f = tmp_path / "categorize" / "2024" / "20240105_hyytiala_categorize.nc"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"old-version")  # size differs from the portal's (10)
    monkeypatch.setattr(cc.requests, "get", lambda *a, **k: pytest.fail("must not download"))
    remote = _product("20240105_hyytiala_categorize.nc")
    assert cc.ensure_downloaded(remote, tmp_path, keep_existing=True) == f
    assert f.read_bytes() == b"old-version"


def test_outdated_file_is_replaced_by_default(tmp_path, monkeypatch):
    f = tmp_path / "categorize" / "2024" / "20240105_hyytiala_categorize.nc"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"old-version")  # 11 bytes; the portal says 10

    class Resp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield b"x" * 10

    monkeypatch.setattr(cc.requests, "get", lambda *a, **k: Resp())
    assert cc.ensure_downloaded(_product("20240105_hyytiala_categorize.nc"), tmp_path) == f
    assert f.read_bytes() == b"x" * 10


@pytest.mark.parametrize("pattern", ["/d/{site}/cloudnet", "/d/{product}/{site}/{product}/{year}/{file}", "/d/{nope}/{product}/{year}/{file}"])
def test_load_config_rejects_bad_path_pattern(tmp_path, pattern):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"path_pattern": pattern}))
    with pytest.raises(ValueError):
        ss.load_config(f)
