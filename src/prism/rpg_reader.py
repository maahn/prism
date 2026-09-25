"""Decode RPG-FMCW-94 LV0 Doppler-spectra files and cache them for fast random access.

A raw hour of spectra decodes to ~1.7 GB per channel (time x range x doppler-bin,
float32) -- too large to reload on every click. We decode once with rpgpy,
convert to dB and quantize to uint8 (0.3 dB steps -- well below radar
calibration uncertainty), and write two zarr copies of each channel at
different chunkings so that both a range-profile-at-one-time query and a
time-series-at-one-range query hit a single chunk:

- "byTime"  chunks = (1, n_range, n_doppler)  -- fast range-profile reads
- "byRange" chunks = (n_time, 1, n_doppler)   -- fast time-series reads
"""
from __future__ import annotations

import os
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rpgpy
import rpgpy.header
import zarr

# RPG instrument software timestamps are seconds since 2001-01-01T00:00:00Z,
# not the Unix epoch. Convert once at decode time so everything downstream
# (moments, other Cloudnet products) can compare on plain Unix seconds.
RPG_EPOCH_OFFSET = 978307200

# dB quantization: code 0 is reserved for "no signal"/masked. Codes 1..255
# cover DB_OFFSET .. DB_OFFSET + 254*DB_SCALE. These defaults suit the
# RPG-FMCW-94, whose measured spectral dynamic range is about -69..-9 dB, at
# 0.3 dB steps (max quantization error 0.15 dB, far below radar calibration
# uncertainty).
#
# They are only DEFAULTS: instruments report spectral power in their own
# units, and METEK's MIRA sits roughly 75 dB lower, so writers pass their own
# window (see mira_reader.decide_db_window) and record it in the zarr's
# attributes. Caches written before that existed carry no attributes, hence
# the fallback in db_window_of().
DB_OFFSET = -75.0
DB_SCALE = 0.3
MASK_CODE = 0


def _encode_db(linear_power: np.ndarray, offset: float = DB_OFFSET, scale: float = DB_SCALE) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(linear_power)
    code = np.round((db - offset) / scale)
    code = np.clip(code, 1, 255)
    valid = np.isfinite(db) & (linear_power > 0)
    return np.where(valid, code, MASK_CODE).astype("uint8")


def _encode_db_parallel(linear_power: np.ndarray, offset: float = DB_OFFSET, scale: float = DB_SCALE,
                         max_workers: int | None = None) -> np.ndarray:
    """Same result as _encode_db, split across threads along the time axis
    (axis 0). Worth doing specifically because it's threads, not processes:
    numpy's own log10/clip/where release the GIL for large arrays (measured
    ~2.6x wall-clock speedup on 8 threads over a real spectra-sized array),
    so this gets real multi-core use on data that's ALREADY resident in
    memory, with none of the pickling/IPC cost a process pool would add for
    arrays this large -- RPG's whole-hour array is loaded in one shot by
    rpgpy, unlike MIRA's per-chunk netCDF reads (see mira_reader.ensure_decoded,
    which parallelizes across PROCESSES instead, because HDF5 reads do NOT
    parallelize under threads -- verified empirically, threaded reads were
    slower than sequential due to the underlying library's internal lock).
    """
    n = linear_power.shape[0]
    max_workers = max_workers or min(os.cpu_count() or 4, 8)
    if n < 2 or max_workers < 2:
        return _encode_db(linear_power, offset, scale)
    bounds = np.array_split(np.arange(n), min(max_workers, n))
    out = np.empty(linear_power.shape, dtype="uint8")

    def _run(idx):
        lo, hi = idx[0], idx[-1] + 1
        out[lo:hi] = _encode_db(linear_power[lo:hi], offset, scale)

    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        list(pool.map(_run, bounds))
    return out


def _decode_db(codes: np.ndarray, offset: float = DB_OFFSET, scale: float = DB_SCALE) -> np.ndarray:
    db = codes.astype("float32") * scale + offset
    db[codes == MASK_CODE] = np.nan
    return db


def db_window_of(store) -> tuple[float, float]:
    """The (offset, scale) a decoded zarr was quantized with."""
    return (float(store.attrs.get("db_offset", DB_OFFSET)),
            float(store.attrs.get("db_scale", DB_SCALE)))


@dataclass(frozen=True)
class ChirpTable:
    """Per-chirp-sequence range offsets and Doppler-velocity axes.

    RPG-FMCW-94 splits the profile into a few chirp sequences; each covers a
    contiguous block of range gates but has its own Nyquist velocity and
    number of Doppler bins, so the velocity axis is NOT uniform across range.

    IMPORTANT: a chirp with fewer bins than the storage width (1024) is
    CENTER-padded, not left-aligned -- e.g. a 512-bin chirp's real data lives
    in array columns 256:768, not 0:512. Naive `[..., :n_bins]` slicing
    silently returns a zero/garbage spectrum with the wrong velocity axis for
    every such range gate. We rely on rpgpy's own `velocity_vectors` (which
    already encodes this padding correctly) rather than reconstructing the
    axis from Nyquist velocity and bin count.
    """
    range_offsets: np.ndarray  # first range-gate index of each chirp
    n_bins: np.ndarray  # doppler bins per chirp
    velocity_vectors: np.ndarray  # (n_chirp, n_doppler_storage), zero-padded

    def chirp_for_range_index(self, range_idx: int) -> int:
        return int(np.searchsorted(self.range_offsets, range_idx, side="right") - 1)

    def valid_slice(self, chirp_idx: int) -> slice:
        """Column range within the storage width holding this chirp's real data."""
        n_storage = self.velocity_vectors.shape[1]
        n = int(self.n_bins[chirp_idx])
        lo = (n_storage - n) // 2
        return slice(lo, lo + n)

    def velocity_axis(self, range_idx: int) -> np.ndarray:
        c = self.chirp_for_range_index(range_idx)
        return self.velocity_vectors[c, self.valid_slice(c)]


def _zarr_cache_path(lv0_path: Path, cache_dir: Path) -> Path:
    return cache_dir / "spectra_zarr" / f"{lv0_path.stem}.zarr"


def _n_points_per_level(header: dict) -> np.ndarray:
    """How many Doppler bins are stored for each range gate -- mirrors
    rpgpy's own (private) _get_n_samples in data.pyx, which is a pure
    function of header fields (chirp boundaries and per-chirp bin counts),
    not of anything actually written per-sample."""
    array = np.ones(int(header["RAltN"]), dtype=int)
    sub_arrays = np.split(array, np.asarray(header["RngOffs"])[1:])
    for sub_array, scale in zip(sub_arrays, header["SpecN"]):
        sub_array *= scale
    return np.concatenate(sub_arrays)


def _scan_sample_records(f, header: dict, n_samples: int):
    """Replicates rpgpy's data.pyx per-sample LV0 byte layout closely enough
    to walk sample records in pure Python WITHOUT decoding the spectral
    data -- just seeking past it -- so a corrupted file's exact byte offset
    can be found without needing the C reader to succeed first.

    Yields (sample_index, time_value, record_start_offset) for each sample.
    Returns (stops iterating) early if a read comes up short, which the
    caller treats as "can't safely characterize this file any further".

    Mirrors every compression (0/1/2) and polarization (0/1/2) branch of
    _read_rpg_l0, but has only been exercised end-to-end against a real
    compression=0, single-polarization file (a corrupted Ny-Alesund hour --
    see tests/test_network_smoke.py). The other branches are written from
    the data.pyx source, not verified against a real file of that kind.
    _repair_truncated_tail's own self-consistency requirement (every
    sample from the first bad one through EOF must also be bad) is the
    guard against silently trusting a mis-parsed offset from an untested
    branch: a coding mistake here would make the "good" prefix look
    smaller or larger than it really is, not band-aid over it.
    """
    n_levels = int(header["RAltN"])
    polarization = int(header["DualPol"])
    compression = int(header["CompEna"])
    anti_alias = int(header["AntiAlias"])
    n_dummy = 3 + int(header["TAltN"]) + 2 * int(header["HAltN"]) + n_levels
    if polarization > 0:
        n_dummy += n_levels
    n_points = _n_points_per_level(header)

    for sample in range(n_samples):
        rec_start = f.tell()
        prefix = f.read(8)  # SampBytes (unused here), Time
        if len(prefix) < 8:
            return
        (time_val,) = struct.unpack("<I", prefix[4:8])
        yield sample, time_val, rec_start

        f.seek(4 + 1 + 4 * 17, 1)  # MSec, QF, RR..PCT (10 + 7 floats)
        f.seek(n_dummy * 4, 1)
        f.seek(n_levels * 4, 1)  # SLv
        if polarization > 0:
            f.seek(n_levels * 4, 1)  # SLh
        is_data = f.read(n_levels)
        if len(is_data) < n_levels:
            return

        for alt_ind in range(n_levels):
            if is_data[alt_ind] != 1:
                continue
            f.seek(4, 1)  # unconditional per-level int preceding spectral data
            n = int(n_points[alt_ind])
            if compression == 0:
                n_channels = 4 if polarization > 0 else 1  # Tot(+H+ReVH+ImVH)
                f.seek(n * 4 * n_channels, 1)
                continue
            nb_byte = f.read(1)
            if not nb_byte:
                return
            n_blocks = nb_byte[0]
            idx_bytes = f.read(4 * n_blocks)
            if len(idx_bytes) < 4 * n_blocks:
                return
            min_ind = struct.unpack(f"<{n_blocks}h", idx_bytes[: 2 * n_blocks])
            max_ind = struct.unpack(f"<{n_blocks}h", idx_bytes[2 * n_blocks:])
            block_points = sum(hi - lo + 1 for lo, hi in zip(min_ind, max_ind))
            n_channels = 1  # TotSpec
            if polarization > 0:
                n_channels += 3  # HSpec, ReVHSpec, ImVHSpec
            if compression == 2:
                n_channels += 3  # RefRat, CorrCoeff, DiffPh
            if compression == 2 and polarization == 2:
                n_channels += 2  # SLDR, SCorrCoeff (block-shaped)
            f.seek(block_points * 4 * n_channels, 1)
            if compression == 2 and polarization == 2:
                f.seek(4 + 4, 1)  # KDP, DiffAtt (one scalar each, per level)
            f.seek(4, 1)  # TotNoisePow
            if polarization > 0:
                f.seek(4, 1)  # HNoisePow
            if anti_alias == 1:
                f.seek(1 + 4, 1)  # AliasMsk, MinVel


def _find_corruption_boundary(lv0_path: Path, header: dict) -> tuple[int, int, int] | None:
    """Finds where a file that failed rpgpy's own timestamp check first goes
    bad. Returns (n_good_samples, corruption_byte_offset, n_total_samples)
    ONLY when every single sample from the first bad one through the end of
    the file is also out of range -- the exact signature confirmed on a
    real corrupted Ny-Alesund file (2024-01-10, hour 11: the last 64 of
    1787 profiles were garbage, everything before was fine). That signature
    means the file's tail is genuinely gone, not that one stray value needs
    guessing at.

    Returns None if nothing is salvageable (corruption at sample 0), the
    failure doesn't match that all-bad-to-EOF signature (a single bad
    timestamp surrounded by good ones is a DIFFERENT situation this
    deliberately does not attempt -- there'd be no reliable way to know
    what the correct value should have been), or the scan itself runs into
    something it can't interpret.
    """
    start_time, stop_time = int(header["StartTime"]), int(header["StopTime"])
    try:
        with open(lv0_path, "rb") as f:
            f.seek(4)  # FileCode
            (header_len,) = struct.unpack("<i", f.read(4))
            f.seek(header_len, 1)
            (n_samples,) = struct.unpack("<i", f.read(4))

            first_bad = None
            bad_count = 0
            for sample, time_val, rec_start in _scan_sample_records(f, header, n_samples):
                if not (start_time <= time_val <= stop_time):
                    if first_bad is None:
                        first_bad = (sample, rec_start)
                    bad_count += 1
    except (struct.error, IndexError, ValueError, OSError):
        return None

    if first_bad is None or first_bad[0] == 0:
        return None
    bad_sample, corruption_offset = first_bad
    if bad_count != n_samples - bad_sample:
        return None
    return bad_sample, corruption_offset, n_samples


def _repair_truncated_tail(lv0_path: Path, header: dict) -> tuple[Path, int, int] | None:
    """If the file matches _find_corruption_boundary's narrow recovery
    signature, writes a truncated copy (everything up to the corruption,
    with the header's own sample count patched down to match) to a temp
    file and returns (temp_path, n_good_samples, n_total_samples) -- the
    caller is responsible for deleting it. Returns None if unrecoverable."""
    found = _find_corruption_boundary(lv0_path, header)
    if found is None:
        return None
    n_good, corruption_offset, n_total = found
    n_samples_field_offset = 8 + int(header["HeaderLen"])

    fd, tmp_name = tempfile.mkstemp(suffix=".lv0")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with open(lv0_path, "rb") as src, open(tmp_path, "wb") as dst:
        remaining = corruption_offset
        while remaining > 0:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
        dst.seek(n_samples_field_offset)
        dst.write(struct.pack("<i", n_good))
    return tmp_path, n_good, n_total


def ensure_decoded(lv0_path: Path, cache_dir: Path, on_status: Callable[[str], None] | None = None) -> Path:
    """Decode an LV0 file into a chunked zarr store, if not already cached."""
    out = _zarr_cache_path(lv0_path, cache_dir)
    done_marker = out / "_SUCCESS"
    if done_marker.exists():
        return out

    recovered = None  # (n_good, n_total) if a corrupted tail was truncated
    try:
        header, data = rpgpy.read_rpg(str(lv0_path))
    except rpgpy.RPGFileError:
        probe_header, _ = rpgpy.header.read_rpg_header(str(lv0_path))
        # NOTE: read_rpg_header's own returned file-position has been
        # observed to be wrong on a real corrupted file (it reported EOF
        # instead of the true post-header offset) -- _find_corruption_
        # boundary/_scan_sample_records deliberately never rely on it,
        # recomputing the post-header offset the same way data.pyx does
        # (FileCode + HeaderLen + seek), using this dict only for field
        # values (StartTime, RAltN, DualPol, etc.), which matched reality.
        repaired = _repair_truncated_tail(lv0_path, probe_header)
        if repaired is None:
            raise
        tmp_path, n_good, n_total = repaired
        try:
            header, data = rpgpy.read_rpg(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        recovered = (n_good, n_total)
        if on_status:
            on_status(f"⚠️ {lv0_path.name} is corrupted after profile {n_good}/{n_total} -- "
                      f"recovered the first {n_good}, discarded the rest.")

    # Only a dual-pol STSR (simultaneous transmit/receive) radar -- RPG's
    # own header flags this as DualPol==2 -- exports HSpec and ReVHSpec at
    # all. For those, TotSpec is H + V + the cross term, NOT H + V alone,
    # so the V-channel ("co", by the same convention CloudnetPy's own
    # moments use, e.g. Zh) is TotSpec - HSpec - 2*Re(ReVHSpec), not just
    # TotSpec - HSpec. Missing that cross-correlation term is a small
    # correction in the well-detected core of a spectrum (median ~0.1 dB
    # against real data) but grows to several dB near the noise floor,
    # exactly where a careful analysis (e.g. spectral LDR/SLDR, per
    # Myagkov/RPG's own published method) would be most sensitive to it.
    # The channel labeling itself (HSpec = cross, the derived V-ish
    # quantity = co) was separately verified against Cloudnet's published
    # `ldr` moment (HSpec / co matched to within ~0.9 dB MAD) and is
    # unaffected by this correction.
    #
    # A single-polarization radar (DualPol==0 -- confirmed against a real
    # Jülich file, whose `data` dict has no HSpec/ReVHSpec/HNoisePow keys
    # at all) has no cross-channel to derive: TotSpec there already IS the
    # co-channel power on its own, so decoding it through the STSR formula
    # crashed with a KeyError instead of just... not having an LDR/SLDR
    # product for that instrument, which is the real, unremarkable
    # situation. A DualPol==1 (LDR mode, alternating H/V transmission
    # rather than simultaneous) radar isn't verified against a real file
    # here, but per RPG's own docs it exports HSpec without ReVHSpec --
    # there's no complex covariance to subtract in that mode, so TotSpec is
    # used as co directly, same as the single-pol case, just with a real
    # HSpec for cross instead of a masked one.
    has_h = "HSpec" in data
    has_cross_term = "ReVHSpec" in data
    if has_h and has_cross_term:
        co_db = _encode_db_parallel(data["TotSpec"] - data["HSpec"] - 2 * data["ReVHSpec"])
        cross_db = _encode_db_parallel(data["HSpec"])
    else:
        co_db = _encode_db_parallel(data["TotSpec"])
        cross_db = _encode_db_parallel(data["HSpec"]) if has_h else np.full_like(co_db, MASK_CODE)
    time_unix = data["Time"].astype("int64") + RPG_EPOCH_OFFSET
    del data

    n_time, n_range, n_doppler = co_db.shape
    store = zarr.open_group(str(out), mode="w")
    store.create_array("time", data=time_unix, chunks=(n_time,))
    store.create_array("range_offsets", data=np.asarray(header["RngOffs"]))
    store.create_array("n_bins", data=np.asarray(header["SpecN"]))
    store.create_array("velocity_vectors", data=np.asarray(header["velocity_vectors"]))
    store.create_array("height", data=np.asarray(header["RAlts"]))

    for name, codes in (("co", co_db), ("cross", cross_db)):
        store.create_array(f"{name}_byTime", data=codes, chunks=(1, n_range, n_doppler))
        store.create_array(f"{name}_byRange", data=codes, chunks=(n_time, 1, n_doppler))

    store.attrs["source_file"] = lv0_path.name
    if recovered is not None:
        n_good, n_total = recovered
        # Recorded persistently (not just via on_status, which only reaches
        # whoever happened to be watching the live status text) so a
        # truncated hour's incompleteness is inspectable later from the
        # cache alone -- see SpectraHour.truncated_from_corruption.
        store.attrs["truncated_from_corruption"] = {"kept_samples": n_good, "expected_samples": n_total}
    done_marker.write_text("ok")
    return out


class SpectraHour:
    """Fast random-access reader for one decoded hour of Doppler spectra."""

    def __init__(self, zarr_path: Path):
        self._store = zarr.open_group(str(zarr_path), mode="r")
        self._db_offset, self._db_scale = db_window_of(self._store)
        self.time = self._store["time"][:]  # unix seconds, int64
        self.height = self._store["height"][:]
        self.chirp = ChirpTable(
            range_offsets=self._store["range_offsets"][:],
            n_bins=self._store["n_bins"][:],
            velocity_vectors=self._store["velocity_vectors"][:],
        )
        self.n_time, self.n_range, self.n_doppler = self._store["co_byTime"].shape
        # Present only for an hour recovered from a corrupted raw file (see
        # ensure_decoded/_repair_truncated_tail) -- {"kept_samples",
        # "expected_samples"} -- so callers can show a persistent warning
        # rather than relying on whoever happened to see the one-off
        # on_status message at decode time.
        self.truncated_from_corruption: dict | None = self._store.attrs.get("truncated_from_corruption")

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return _decode_db(codes, self._db_offset, self._db_scale)

    def nearest_time_index(self, unix_time: float) -> int:
        return int(np.searchsorted(self.time, unix_time))

    def nearest_range_index(self, height_m: float) -> int:
        return int(np.argmin(np.abs(self.height - height_m)))

    def spectrum(self, time_idx: int, range_idx: int, channel: str = "co") -> tuple[np.ndarray, np.ndarray]:
        """Return (velocity_axis, power_db) for one time/range point."""
        c = self.chirp.chirp_for_range_index(range_idx)
        sl = self.chirp.valid_slice(c)
        codes = self._store[f"{channel}_byRange"][time_idx, range_idx, sl]
        vel = self.chirp.velocity_axis(range_idx)
        return vel, self._decode(codes)

    def range_profile_segments(self, time_idx: int, channel: str = "co"):
        """Spectrum-vs-range at one time step (dB), split by chirp segment.

        Doppler resolution and Nyquist velocity change per chirp, so this
        cannot be a single dense (range, doppler) image. Returns one
        (range_slice, velocity_axis, block_db) tuple per chirp, where block
        has shape (n_ranges_in_chirp, n_bins_in_chirp).
        """
        arr = self._store[f"{channel}_byTime"]
        offsets = list(self.chirp.range_offsets) + [self.n_range]
        segments = []
        for c in range(len(offsets) - 1):
            start, stop = offsets[c], offsets[c + 1]
            sl = self.chirp.valid_slice(c)
            codes = arr[time_idx, start:stop, sl]
            vel = self.chirp.velocity_axis(start)
            segments.append((slice(start, stop), vel, self._decode(codes)))
        return segments

    def time_series(self, range_idx: int, t_start: int, t_stop: int, channel: str = "co") -> np.ndarray:
        """Spectrum-vs-time image (dB) at one range gate: shape (n_times, doppler_for_this_chirp)."""
        c = self.chirp.chirp_for_range_index(range_idx)
        sl = self.chirp.valid_slice(c)
        codes = self._store[f"{channel}_byRange"][t_start:t_stop, range_idx, sl]
        return self._decode(codes)
