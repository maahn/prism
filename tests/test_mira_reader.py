"""decide_db_window() is the one place mira_reader.py has to guess a
per-file constant from the data itself (see its own docstring for why:
MIRA's units sit ~75 dB below RPG's, so quantizing against RPG's fixed
window pinned 99.7% of every spectrum to a single code). It's the kind of
thing that looks fine on the one file it was tuned against and silently
misbehaves on the next.
"""
import numpy as np

from prism import mira_reader as mr
from prism import rpg_reader as rr


def test_decide_db_window_spans_real_dynamic_range():
    rng = np.random.default_rng(0)
    # Realistic MIRA-ish linear power: mostly noise floor (~1e-13) with a
    # cloud signal a few orders of magnitude above it.
    sample = rng.uniform(1e-13, 1e-11, 5000)
    sample = np.concatenate([sample, rng.uniform(1e-9, 1e-7, 200)])

    offset, scale = mr.decide_db_window(sample)
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(sample)
    codes = rr._encode_db(sample, offset, scale)

    # The window should cover the data's real spread, not collapse
    # everything to one or two codes the way the RPG-calibrated default
    # window did for MIRA (see mira_reader.decide_db_window's docstring).
    assert codes.max() - codes[codes != rr.MASK_CODE].min() > 100
    # And it shouldn't blow the 8-bit budget: the top of the range should
    # land near, not far past, code 255.
    assert 200 <= codes.max() <= 255


def test_decide_db_window_handles_all_nonpositive_input():
    # Every profile is fully masked/dropped (e.g. a corrupted or all-zero
    # chunk) -- must fall back cleanly rather than raise or return NaN.
    offset, scale = mr.decide_db_window(np.zeros(100))
    assert offset == rr.DB_OFFSET
    assert scale == rr.DB_SCALE


def test_zarr_cache_path_strips_both_known_suffixes():
    from pathlib import Path
    p = mr._zarr_cache_path(Path("20260916_200006.znc.gz"), Path("/cache"))
    assert p == Path("/cache/spectra_zarr/20260916_200006.zarr")
