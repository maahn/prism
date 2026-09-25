"""Filename parsing is the part of cloudnet_client.py most likely to
silently break on a site/instrument we haven't seen before -- there's no
fixed spec for these filenames, just whatever convention each institution
happened to pick. Every case below is a REAL filename observed from the
live API (see the comment on each), not a guess.
"""
from prism.cloudnet_client import _is_raw_spectra_file, hour_of_filename


def test_hour_of_filename_separated_yymmdd():
    # Hyytiälä / Bucharest RPG-FMCW-94: YYMMDD_HHMMSS_<program>_ZEN.LV0
    assert hour_of_filename("260211_000000_P09_ZEN.LV0") == 0
    assert hour_of_filename("260211_080002_P09_ZEN.LV0") == 8
    assert hour_of_filename("260211_230059_P06_ZEN.LV0") == 23


def test_hour_of_filename_separated_yyyymmdd():
    # Munich MIRA-10: YYYYMMDD_HHMMSS.znc.gz
    assert hour_of_filename("20260916_000004.znc.gz") == 0
    assert hour_of_filename("20260916_200006.znc.gz") == 20


def test_hour_of_filename_run_together_with_instrument_prefix():
    # Ny-Ålesund and Jülich RPG-FMCW-94: <instrument>_YYYYMMDDHHMMSS_<program>_ZEN.lv0
    # -- no underscore between the date and the time. A prior version of
    # hour_of_filename (filename.split("_")[1][0:2]) read "20" here every
    # time -- the START OF THE YEAR, not the hour -- so every file that day
    # collided under "20 UTC" and the other 23 hours looked missing.
    assert hour_of_filename("joyrad94_20240110000001_P01_ZEN.lv0") == 0
    assert hour_of_filename("joyrad94_20240110010000_P01_ZEN.lv0") == 1
    assert hour_of_filename("joyrad94_20240110230000_P01_ZEN.lv0") == 23
    assert hour_of_filename("mirac-a_20210120000001_P01_ZEN.lv0") == 0
    assert hour_of_filename("mirac-a_20210120130000_P01_ZEN.lv0") == 13


def test_hour_of_filename_unrecognized_raises():
    import pytest
    with pytest.raises(ValueError):
        hour_of_filename("not_a_timestamp_at_all.txt")


def test_is_raw_spectra_file_rpg():
    assert _is_raw_spectra_file("rpg-fmcw-94", "260211_080002_P09_ZEN.LV0")
    assert _is_raw_spectra_file("rpg-fmcw-94", "joyrad94_20240110010000_P01_ZEN.lv0")  # lowercase extension
    assert not _is_raw_spectra_file("rpg-fmcw-94", "260211_080002_P09_ZEN.LV1")  # moments-only, not full spectra


def test_is_raw_spectra_file_mira():
    assert _is_raw_spectra_file("mira-10", "20260916_200006.znc.gz")
    assert _is_raw_spectra_file("mira-35", "20260916_200006.znc.gz")
    assert not _is_raw_spectra_file("mira-35", "20260916_200006_windppi.znc.gz")  # PPI scan, no vertical profile
