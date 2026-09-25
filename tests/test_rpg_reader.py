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
import struct

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


# --- Corrupted-file tail recovery ---------------------------------------
#
# A real Ny-Alesund file (2024-01-10, hour 11) has a corrupted tail: rpgpy's
# own timestamp check rejects the last 64 of 1787 profiles as garbage
# (confirmed by downloading and inspecting it directly -- checksum matched
# what Cloudnet publishes, so it's upstream data corruption, not a download
# bug). CloudnetPy's own answer to this (see its rpg.py: catches
# RPGFileError per-file and just drops the whole file) is to accept total
# data loss; _repair_truncated_tail instead recovers the good prefix by
# reverse-engineering rpgpy's own LV0 record layout (from its data.pyx) well
# enough to find exactly where the file desyncs, truncate there, and patch
# the header's own sample count down to match.
#
# These tests build a hand-crafted sample-records region (NOT a real,
# rpgpy-parseable header -- see the "IMPORTANT" note below) so
# _scan_sample_records/_find_corruption_boundary/_repair_truncated_tail's
# own boundary-finding logic can be tested without a real, multi-GB RPG
# file. The full pipeline against a REAL corrupted file (rpgpy actually
# succeeding on the repaired output) is instead covered by the opt-in
# tests/test_network_smoke.py, since that needs bytes only rpgpy's own
# format truly produces.

def _fake_header(n_levels=2, n_doppler=4, start_time=1000, stop_time=1010):
    return {
        "RAltN": n_levels, "TAltN": 0, "HAltN": 0, "DualPol": 0, "CompEna": 0,
        "AntiAlias": 0, "RngOffs": np.array([0]), "SpecN": np.array([n_doppler]),
        "StartTime": start_time, "StopTime": stop_time, "HeaderLen": 16,
    }


def _build_fake_records(header, times) -> bytes:
    """The sample-records region of an LV0-like file for a
    compression=0/DualPol=0/AntiAlias=0 header -- exactly what
    _scan_sample_records expects to walk. IMPORTANT: this is NOT a
    real/rpgpy-parseable file (the header bytes are arbitrary padding) --
    it only needs to satisfy our OWN _scan_sample_records, which is what's
    under test here."""
    n_levels = int(header["RAltN"])
    n_dummy = 3 + int(header["TAltN"]) + 2 * int(header["HAltN"]) + n_levels
    n_points = rr._n_points_per_level(header)
    out = bytearray()
    for t in times:
        out += struct.pack("<i", 0)  # SampBytes (unused by our code)
        out += struct.pack("<I", t)  # Time
        out += struct.pack("<i", 0)  # MSec
        out += struct.pack("<b", 0)  # QF
        out += bytes(4 * 17)  # RR..PCT (10 + 7 floats)
        out += bytes(n_dummy * 4)
        out += bytes(n_levels * 4)  # SLv
        out += bytes([1] * n_levels)  # is_data: every level present
        for alt_ind in range(n_levels):
            out += struct.pack("<i", 0)  # per-level unconditional int
            out += bytes(int(n_points[alt_ind]) * 4)  # TotSpec block
    return bytes(out)


def _write_fake_file(tmp_path, header, times):
    path = tmp_path / "fake.lv0"
    header_len = header["HeaderLen"]
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 0))  # FileCode (unused by our code)
        f.write(struct.pack("<i", header_len))
        f.write(bytes(header_len))
        f.write(struct.pack("<i", len(times)))
        f.write(_build_fake_records(header, times))
    return path


def test_find_corruption_boundary_recovers_contiguous_tail(tmp_path):
    header = _fake_header()
    times = [1000, 1002, 1004, 1006, 1008, 1010, 99999, 88888]  # last 2 out of range
    path = _write_fake_file(tmp_path, header, times)
    result = rr._find_corruption_boundary(path, header)
    assert result is not None
    n_good, _corruption_offset, n_total = result
    assert (n_good, n_total) == (6, 8)


def test_find_corruption_boundary_refuses_isolated_bad_sample(tmp_path):
    # A single stray bad timestamp surrounded by good ones on both sides --
    # deliberately NOT the same situation as a corrupted tail: there's no
    # reliable way to know what the correct value should have been, so this
    # must refuse to guess rather than silently drop or fabricate one
    # profile.
    header = _fake_header()
    times = [1000, 1002, 99999, 1006, 1008, 1010]
    path = _write_fake_file(tmp_path, header, times)
    assert rr._find_corruption_boundary(path, header) is None


def test_find_corruption_boundary_refuses_when_nothing_good(tmp_path):
    header = _fake_header()
    times = [99999, 88888, 77777]  # corrupted from the very first sample
    path = _write_fake_file(tmp_path, header, times)
    assert rr._find_corruption_boundary(path, header) is None


def test_repair_truncated_tail_produces_a_well_formed_prefix(tmp_path):
    header = _fake_header()
    times = [1000, 1002, 1004, 1006, 1008, 1010, 99999, 88888]
    path = _write_fake_file(tmp_path, header, times)

    repaired = rr._repair_truncated_tail(path, header)
    assert repaired is not None
    tmp_out, n_good, n_total = repaired
    try:
        assert (n_good, n_total) == (6, 8)
        with open(tmp_out, "rb") as f:
            f.seek(4)
            (header_len,) = struct.unpack("<i", f.read(4))
            assert header_len == header["HeaderLen"]
            f.seek(header_len, 1)
            (n_samples,) = struct.unpack("<i", f.read(4))
            assert n_samples == n_good  # header's own sample count was patched down
            recovered = [t for _, t, _ in rr._scan_sample_records(f, header, n_samples)]
        assert recovered == times[:6]  # exactly the good prefix, nothing fabricated
    finally:
        tmp_out.unlink()


def test_repair_truncated_tail_none_when_unrecoverable(tmp_path):
    header = _fake_header()
    times = [1000, 99999, 1004]  # isolated bad sample -- not a recoverable signature
    path = _write_fake_file(tmp_path, header, times)
    assert rr._repair_truncated_tail(path, header) is None
