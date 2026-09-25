"""ensure_decoded()'s co/cross derivation branches on which of
HSpec/ReVHSpec RPG's own read_rpg() actually returned -- which depends on
the radar's polarization mode (its DualPol header flag), not on anything
we control. A single-polarization radar (DualPol==0, no HSpec/ReVHSpec at
all -- confirmed against a real Jülich file) used to crash with
KeyError('HSpec') the first time load_hour() tried to decode one, because
the STSR-only formula assumed those keys always exist. rpgpy itself isn't
exercised here (it needs a real .LV0 file and a C extension) -- these tests
monkeypatch rpgpy.read_rpg to return the same SHAPE of header/data dict for
each polarization mode and check ensure_decoded()'s own branching logic,
which is the part that actually broke.
"""
import numpy as np
import pytest
import zarr

from prism import rpg_reader as rr


def _make_header_data(n_time=3, n_range=2, n_doppler=4, dual_pol_stsr=False,
                       ldr_mode=False):
    rng = np.random.default_rng(0)
    tot = rng.uniform(1e-8, 1e-6, (n_time, n_range, n_doppler)).astype("float32")
    header = {
        "RngOffs": np.array([0]),
        "SpecN": np.array([n_doppler]),
        "velocity_vectors": np.zeros((1, n_doppler), dtype="float32"),
        "RAlts": np.linspace(100, 200, n_range),
        "DualPol": 2 if dual_pol_stsr else (1 if ldr_mode else 0),
    }
    data = {
        "TotSpec": tot,
        "Time": np.arange(n_time, dtype="int64"),
    }
    if dual_pol_stsr:
        data["HSpec"] = rng.uniform(1e-9, 1e-7, tot.shape).astype("float32")
        data["ReVHSpec"] = rng.uniform(-1e-8, 1e-8, tot.shape).astype("float32")
    elif ldr_mode:
        data["HSpec"] = rng.uniform(1e-9, 1e-7, tot.shape).astype("float32")
    # single-pol (DualPol==0): neither key present, matching the real
    # Jülich file's rpgpy.read_rpg() output exactly.
    return header, data


@pytest.mark.parametrize("dual_pol_stsr,ldr_mode", [
    (True, False),   # DualPol==2: HSpec + ReVHSpec both present
    (False, True),   # DualPol==1: HSpec present, no ReVHSpec
    (False, False),  # DualPol==0: neither present -- this is the crash case
])
def test_ensure_decoded_handles_every_polarization_mode(tmp_path, monkeypatch, dual_pol_stsr, ldr_mode):
    header, data = _make_header_data(dual_pol_stsr=dual_pol_stsr, ldr_mode=ldr_mode)
    monkeypatch.setattr(rr.rpgpy, "read_rpg", lambda path: (header, dict(data)))

    lv0_path = tmp_path / "fake_260101_000000_P01_ZEN.LV0"
    lv0_path.write_bytes(b"not a real LV0 file -- read_rpg is mocked above")
    cache_dir = tmp_path / "cache"

    out = rr.ensure_decoded(lv0_path, cache_dir)  # must not raise

    assert (out / "_SUCCESS").exists()
    store = zarr.open_group(str(out), mode="r")
    co = store["co_byTime"][:]
    cross = store["cross_byTime"][:]
    assert co.shape == cross.shape == data["TotSpec"].shape
    assert co.dtype == cross.dtype == np.uint8

    if dual_pol_stsr or ldr_mode:
        # A real cross channel exists -- it should carry real (non-fully-masked) data.
        assert (cross != rr.MASK_CODE).any()
    else:
        # Single-pol: no cross channel at all, so it must be masked
        # throughout rather than silently showing fabricated co-channel
        # data under the "cross-polar" toggle.
        assert (cross == rr.MASK_CODE).all()

    # co always has real (non-fully-masked) data regardless of mode.
    assert (co != rr.MASK_CODE).any()


def test_encode_decode_db_roundtrip():
    linear = np.array([1e-7, 1e-6, 1e-5, 0.0, np.nan, -1.0], dtype="float32")
    codes = rr._encode_db(linear)
    assert codes.dtype == np.uint8
    # Non-positive/non-finite power has no valid dB value -- must be masked,
    # not silently turned into some arbitrary quantized code.
    assert codes[3] == rr.MASK_CODE  # 0.0
    assert codes[4] == rr.MASK_CODE  # nan
    assert codes[5] == rr.MASK_CODE  # negative

    decoded = rr._decode_db(codes)
    valid = codes != rr.MASK_CODE
    # Quantization error must stay well under one step (0.3 dB default).
    assert np.allclose(10 * np.log10(linear[valid]), decoded[valid], atol=rr.DB_SCALE)
    assert np.all(np.isnan(decoded[~valid]))


def test_encode_db_parallel_matches_sequential():
    rng = np.random.default_rng(1)
    linear = rng.uniform(1e-8, 1e-3, (37, 5, 6)).astype("float32")  # odd size: uneven thread split
    sequential = rr._encode_db(linear)
    parallel = rr._encode_db_parallel(linear, max_workers=4)
    assert np.array_equal(sequential, parallel)
