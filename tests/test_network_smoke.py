"""End-to-end sweep across real site/instrument/date combinations. This is
the test that would actually have caught both 2026-09-25 bugs (the
KeyError('HSpec') crash on a single-pol RPG radar at Julich, and the
"only 20 UTC" hour-collision at Ny-Alesund): both were invisible to the
unit tests because the unit tests only exercise logic we already knew to
be suspicious. Opt-in only (@pytest.mark.network) -- it hits the real
Cloudnet API and downloads real files, so it's not run by default (see
tests/README.md). Run it whenever a new site/instrument is added, or
periodically to catch a case we haven't seen yet.
"""
import datetime

import pytest
import zarr

from prism import cloudnet_client as cc
from prism import mira_reader as mr
from prism import rpg_reader as rr

pytestmark = pytest.mark.network

# (site, day, instrument) -- chosen to cover every filename convention and
# polarization mode this app has actually hit in production so far.
CASES = [
    ("juelich", datetime.date(2021, 1, 20), "rpg-fmcw-94"),      # single-pol, run-together filename
    ("ny-alesund", datetime.date(2024, 1, 10), "rpg-fmcw-94"),   # run-together filename, dual-pol
    ("hyytiala", datetime.date(2024, 2, 16), "rpg-fmcw-94"),     # separated YYMMDD filename, dual-pol STSR
    ("munich", datetime.date(2026, 9, 16), "mira-10"),           # separated YYYYMMDD filename
]


@pytest.mark.parametrize("site,day,instrument", CASES)
def test_hours_are_not_collapsed(site, day, instrument):
    files = cc.list_raw_spectra_files(site, day, instrument)
    if not files:
        pytest.skip(f"no {instrument} raw spectra published for {site} {day}")
    hours = {cc.hour_of_filename(f.filename) for f in files}
    # A radar restart can legitimately produce two files starting in the same
    # hour (confirmed for real at Julich 2021-01-20, hour 10), so this isn't
    # "one file per hour" -- but real files span a whole day, so they must
    # spread across most of it. This is exactly the assertion that would have
    # failed for Ny-Alesund before the hour_of_filename fix (24 files all
    # "collapsing" onto hour 20 alone).
    assert len(hours) >= min(len(files), 12), (
        f"{site} {day}: {len(files)} files but only {len(hours)} distinct hours -- "
        "hour_of_filename is likely misparsing this site's filename format"
    )


@pytest.mark.parametrize("site,day,instrument", CASES)
def test_first_hour_decodes_without_raising(site, day, instrument, tmp_path):
    files = cc.list_raw_spectra_files(site, day, instrument)
    if not files:
        pytest.skip(f"no {instrument} raw spectra published for {site} {day}")
    remote = files[0]
    local = cc.ensure_downloaded(remote, cache_dir=tmp_path / "raw")

    decoder = rr if instrument == "rpg-fmcw-94" else mr
    out = decoder.ensure_decoded(local, tmp_path / "decoded")
    assert (out / "_SUCCESS").exists()


def test_known_corrupted_file_is_recovered_not_lost(tmp_path):
    """Pins a real, confirmed-corrupted file: Ny-Alesund 2024-01-10 hour 11
    has a corrupted tail -- the last 64 of 1787 profiles fail rpgpy's own
    timestamp check (checksum verified against the Cloudnet API, so this is
    upstream data corruption, not a download bug on our side). rpgpy itself
    offers no way to decode around it (it exposes no option to skip the
    check, and dies on the FIRST bad timestamp it hits), so
    rpg_reader._repair_truncated_tail reverse-engineers rpgpy's own LV0
    record layout well enough to find exactly where the file desyncs,
    truncates there, and hands rpgpy a patched copy -- recovering the good
    1723 profiles instead of losing the whole hour (CloudnetPy's own
    rpg.py, by contrast, just drops a file like this entirely; see the
    comment in tests/test_rpg_reader.py's corruption-recovery section for
    why we chose differently here).

    The exact counts (1723 kept, 1787 total) are pinned to THIS file, not
    computed -- a future rpgpy behavior change (looser or stricter
    validation) would change them, and this test failing is exactly the
    signal that our own truncation math needs re-checking against reality.
    """
    files = cc.list_raw_spectra_files("ny-alesund", datetime.date(2024, 1, 10), "rpg-fmcw-94")
    remote = next((f for f in files if cc.hour_of_filename(f.filename) == 11), None)
    if remote is None:
        pytest.skip("expected hour-11 file no longer published for this site/day")
    local = cc.ensure_downloaded(remote, cache_dir=tmp_path / "raw")

    statuses = []
    out = rr.ensure_decoded(local, tmp_path / "decoded", on_status=statuses.append)

    assert (out / "_SUCCESS").exists()
    assert any("corrupted after profile 1723/1787" in s for s in statuses)
    store = zarr.open_group(str(out), mode="r")
    assert dict(store.attrs["truncated_from_corruption"]) == {"kept_samples": 1723, "expected_samples": 1787}
    assert store["co_byTime"].shape[0] == 1723
