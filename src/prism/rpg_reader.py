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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rpgpy
import zarr

# RPG instrument software timestamps are seconds since 2001-01-01T00:00:00Z,
# not the Unix epoch. Convert once at decode time so everything downstream
# (moments, other Cloudnet products) can compare on plain Unix seconds.
RPG_EPOCH_OFFSET = 978307200

# dB quantization: code 0 is reserved for "no signal"/masked. Codes 1..255
# cover DB_OFFSET .. DB_OFFSET + 254*DB_SCALE. Measured spectral dynamic range
# is about -69..-9 dB, so this leaves comfortable headroom at 0.3 dB steps
# (max quantization error 0.15 dB, far below radar calibration uncertainty).
DB_OFFSET = -75.0
DB_SCALE = 0.3
MASK_CODE = 0


def _encode_db(linear_power: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(linear_power)
    code = np.round((db - DB_OFFSET) / DB_SCALE)
    code = np.clip(code, 1, 255)
    valid = np.isfinite(db) & (linear_power > 0)
    return np.where(valid, code, MASK_CODE).astype("uint8")


def _decode_db(codes: np.ndarray) -> np.ndarray:
    db = codes.astype("float32") * DB_SCALE + DB_OFFSET
    db[codes == MASK_CODE] = np.nan
    return db


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


def ensure_decoded(lv0_path: Path, cache_dir: Path) -> Path:
    """Decode an LV0 file into a chunked zarr store, if not already cached."""
    out = _zarr_cache_path(lv0_path, cache_dir)
    done_marker = out / "_SUCCESS"
    if done_marker.exists():
        return out

    header, data = rpgpy.read_rpg(str(lv0_path))

    # RPG's "TotSpec" is co+cross combined power, not co-channel alone; true
    # co-channel power is TotSpec - HSpec (verified against the
    # Cloudnet-derived LDR: reconstructing HSpec / (TotSpec - HSpec) matches
    # the published `ldr` product to within ~0.9 dB MAD).
    co_db = _encode_db(data["TotSpec"] - data["HSpec"])
    cross_db = _encode_db(data["HSpec"])
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
    done_marker.write_text("ok")
    return out


class SpectraHour:
    """Fast random-access reader for one decoded hour of Doppler spectra."""

    def __init__(self, zarr_path: Path):
        self._store = zarr.open_group(str(zarr_path), mode="r")
        self.time = self._store["time"][:]  # unix seconds, int64
        self.height = self._store["height"][:]
        self.chirp = ChirpTable(
            range_offsets=self._store["range_offsets"][:],
            n_bins=self._store["n_bins"][:],
            velocity_vectors=self._store["velocity_vectors"][:],
        )
        self.n_time, self.n_range, self.n_doppler = self._store["co_byTime"].shape

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
        return vel, _decode_db(codes)

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
            segments.append((slice(start, stop), vel, _decode_db(codes)))
        return segments

    def time_series(self, range_idx: int, t_start: int, t_stop: int, channel: str = "co") -> np.ndarray:
        """Spectrum-vs-time image (dB) at one range gate: shape (n_times, doppler_for_this_chirp)."""
        c = self.chirp.chirp_for_range_index(range_idx)
        sl = self.chirp.valid_slice(c)
        codes = self._store[f"{channel}_byRange"][t_start:t_stop, range_idx, sl]
        return _decode_db(codes)
