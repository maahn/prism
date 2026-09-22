"""Decode METEK MIRA (mira-10, mira-35) Doppler-spectra ".znc.gz" files and
cache them in the SAME zarr schema rpg_reader.py uses, so SpectraHour and the
whole app.py query/rendering layer work unchanged regardless of which
instrument produced the data.

Unlike RPG-FMCW-94's multiple chirp sequences (each with its own Nyquist
velocity and bin count), MIRA's znc format uses a single uniform Doppler axis
for every range gate -- so it's represented as a ChirpTable with exactly one
"chirp" spanning the whole profile, reusing that machinery unmodified.

Two format quirks, both verified against a real Munich mira-10 file:
- Doppler bins are stored in native FFT order (0, +bin, ... +Nyquist, then
  -Nyquist, ... -bin), not ascending -- needs reordering before it's a sane
  axis to plot against, exactly like RPG's chirp center-padding needed its
  own unpacking.
- SPCco/SPCcx ("Doppler Spectrum Co/Cross-Channel") are linear power despite
  the file's own "db: 1" attribute flag (which turned out to just mean
  "this variable's plotting convention is dB scaled", not "already in dB") --
  confirmed by inspecting real values (~1e-13 to 1e-7, spanning many decades).
"""
from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from prism import rpg_reader as rr
from prism.rpg_reader import MASK_CODE, _encode_db

# Decoding one hour materializes ~5 GB (uncompressed) per channel at MIRA-10's
# 4096 Doppler bins; process in time chunks so peak memory stays under ~1 GB
# instead of loading the whole hour into memory at once.
_TIME_CHUNK = 50
# Measured on a real Munich mira-10 hour: reading one 50-profile chunk from
# the netCDF file takes ~0.7s, but _encode_db-ing it takes ~3s PER channel
# (~6s for co+cross together) -- that dominant cost is pure numpy math on
# data already in memory, which should parallelize beautifully, but this
# process reads the SAME shared netCDF/HDF5 file per chunk, and h5netcdf's
# underlying library serializes concurrent reads through an internal lock:
# measured threaded reads were actually SLOWER than sequential ones. Real
# separate PROCESSES each get their own independent HDF5 state and don't
# fight that lock, so chunks are parallelized across processes here (see
# _decode_one_chunk) rather than threads -- unlike rpg_reader's
# _encode_db_parallel, which threads because RPG's data is already fully
# resident in memory with no per-chunk file access to contend over.
_MAX_DECODE_WORKERS = min(os.cpu_count() or 4, 8)
# co_byRange/cross_byRange are chunked (n_time, 1, n_doppler) -- one zarr
# chunk PER RANGE GATE, spanning the full time axis -- so they need to be
# populated in RANGE-aligned blocks, not time-aligned ones (see
# _decode_one_range_block). This many range gates per block, matching
# _TIME_CHUNK's rough memory budget per worker.
_RANGE_CHUNK = 10


def decide_db_window(sample_linear: np.ndarray) -> tuple[float, float]:
    """Pick the (offset, scale) for uint8 dB quantization from a sample of a
    file's own spectral powers.

    rpg_reader's defaults (-75 dB floor, 0.3 dB steps) are calibrated for the
    RPG-FMCW-94's units. MIRA reports the same quantity about 75 dB lower
    (~1e-13..1e-7 linear, i.e. -130..-70 dB), so quantizing it against the RPG
    window pinned 99.7% of every spectrum to code 1 -- the app drew perfectly
    flat, featureless spectrograms with a ~1 dB total range. Deriving the
    window from the data keeps the full dynamic range for any instrument's
    units, at the price of a scale that varies slightly between files (which
    is why it is stored per file; see rpg_reader.db_window_of).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(sample_linear[sample_linear > 0])
    db = db[np.isfinite(db)]
    if db.size == 0:
        return float(rr.DB_OFFSET), float(rr.DB_SCALE)
    # A low percentile rather than the minimum: a handful of near-zero powers
    # would otherwise stretch the window over empty decades and waste most of
    # the 255 codes on noise nobody looks at.
    lo, hi = float(np.percentile(db, 0.1)), float(db.max())
    scale = max((hi - lo) / 254.0, 0.01)
    return lo - scale, scale


def _decode_one_chunk(znc_path: str, start: int, stop: int, order: np.ndarray,
                       db_offset: float, db_scale: float, has_cross: bool):
    """Runs in a WORKER PROCESS (see _MAX_DECODE_WORKERS): opens its own
    handle onto the already-extracted .znc file -- xarray/h5netcdf Dataset
    objects can't be pickled across a process boundary, so each worker
    needs its own -- reads and encodes one time-chunk, and returns the
    result for the main process to write to zarr. Keeping the actual zarr
    writes in ONE process (see ensure_decoded) avoids a real correctness
    hazard: co_byRange/cross_byRange's chunking spans the FULL time axis
    per range gate, so writing two different chunks' time-slices into it
    concurrently would race (each write is a read-modify-write of the
    whole underlying chunk file) -- reading/encoding in parallel and then
    writing sequentially back in the caller sidesteps that entirely.
    """
    with xr.open_dataset(znc_path, decode_times=False) as ds:
        co_db = _encode_db(ds["SPCco"].isel(time=slice(start, stop)).values[:, :, order],
                            db_offset, db_scale)
        if has_cross:
            cross_db = _encode_db(ds["SPCcx"].isel(time=slice(start, stop)).values[:, :, order],
                                   db_offset, db_scale)
        else:
            # This instrument/configuration doesn't export a cross-channel
            # spectrum -- leave it fully masked rather than fabricating
            # data; the app already renders a masked channel as "no data"
            # cleanly.
            cross_db = np.full_like(co_db, MASK_CODE)
    return start, stop, co_db, cross_db


def _decode_one_range_block(znc_path: str, r0: int, r1: int, order: np.ndarray,
                             db_offset: float, db_scale: float, has_cross: bool):
    """Same idea as _decode_one_chunk, but slices along the RANGE axis
    instead of time -- used to populate co_byRange/cross_byRange. Its zarr
    chunking is (n_time, 1, n_doppler): ONE chunk per range gate, spanning
    the full time axis. Populating it from TIME-sliced chunks (like
    co_byTime) would mean every single time-chunk write touches EVERY
    range gate's chunk file, and a partial write to a zarr chunk means
    reading the whole existing chunk, patching in the new slice, and
    writing the whole thing back -- that read-modify-write, repeated once
    per time-chunk times every range chunk, measured as the actual
    dominant cost (~10 minutes for one hour), dwarfing the encode step by
    an order of magnitude even after parallelizing that. Slicing along
    range instead means each write lands on complete, non-overlapping
    chunks, so no read-modify-write is needed at all.
    """
    with xr.open_dataset(znc_path, decode_times=False) as ds:
        co_db = _encode_db(ds["SPCco"].isel(range=slice(r0, r1)).values[:, :, order],
                            db_offset, db_scale)
        if has_cross:
            cross_db = _encode_db(ds["SPCcx"].isel(range=slice(r0, r1)).values[:, :, order],
                                   db_offset, db_scale)
        else:
            cross_db = np.full_like(co_db, MASK_CODE)
    return r0, r1, co_db, cross_db


def _zarr_cache_path(znc_gz_path: Path, cache_dir: Path) -> Path:
    name = znc_gz_path.name
    for suffix in (".gz", ".znc"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return cache_dir / "spectra_zarr" / f"{name}.zarr"


def ensure_decoded(znc_gz_path: Path, cache_dir: Path) -> Path:
    """Decode a gzip-compressed MIRA znc file into a chunked zarr store, if
    not already cached."""
    out = _zarr_cache_path(znc_gz_path, cache_dir)
    done_marker = out / "_SUCCESS"
    if done_marker.exists():
        return out

    with tempfile.NamedTemporaryFile(suffix=".znc") as tmp:
        with gzip.open(znc_gz_path, "rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp.flush()
        with xr.open_dataset(tmp.name, decode_times=False) as ds:
            doppler = ds["doppler"].values
            order = np.argsort(doppler)  # native FFT order -> ascending velocity
            velocity = doppler[order]
            time_unix = ds["time"].values.astype("int64")
            height = ds["range"].values.astype("float64")
            has_cross = "SPCcx" in ds.data_vars
            n_time, n_range, n_doppler = ds["SPCco"].shape
            # A few profiles spread across the hour are enough to size the
            # quantization window, and avoid reading the whole (multi-GB)
            # hour twice.
            probe = ds["SPCco"].isel(time=slice(None, None, max(1, n_time // 8))).values
            db_offset, db_scale = decide_db_window(probe)
            del probe

            store = zarr.open_group(str(out), mode="w")
            store.create_array("time", data=time_unix, chunks=(n_time,))
            store.create_array("range_offsets", data=np.array([0]))
            store.create_array("n_bins", data=np.array([n_doppler]))
            store.create_array("velocity_vectors", data=velocity[np.newaxis, :])
            store.create_array("height", data=height)

            arrays = {}
            for name in ("co", "cross"):
                arrays[f"{name}_byTime"] = store.create_array(
                    f"{name}_byTime", shape=(n_time, n_range, n_doppler),
                    dtype="uint8", chunks=(1, n_range, n_doppler))
                arrays[f"{name}_byRange"] = store.create_array(
                    f"{name}_byRange", shape=(n_time, n_range, n_doppler),
                    dtype="uint8", chunks=(n_time, 1, n_doppler))

            # Two SEPARATE passes -- one time-aligned (for *_byTime), one
            # range-aligned (for *_byRange) -- because those two arrays'
            # zarr chunkings are transposed relative to each other,
            # and only a write aligned with an array's OWN chunking avoids
            # an expensive read-modify-write (see _decode_one_range_block).
            # Both passes re-read the source file (once per axis) rather
            # than sharing one pass's results -- more total I/O, but far
            # less than the read-modify-write cost it replaces.
            time_bounds = [(s, min(s + _TIME_CHUNK, n_time)) for s in range(0, n_time, _TIME_CHUNK)]
            with ProcessPoolExecutor(max_workers=min(_MAX_DECODE_WORKERS, len(time_bounds))) as pool:
                futures = [pool.submit(_decode_one_chunk, tmp.name, start, stop, order,
                                        db_offset, db_scale, has_cross)
                           for start, stop in time_bounds]
                for future in as_completed(futures):
                    start, stop, co_db, cross_db = future.result()
                    arrays["co_byTime"][start:stop] = co_db
                    arrays["cross_byTime"][start:stop] = cross_db

            range_bounds = [(r, min(r + _RANGE_CHUNK, n_range)) for r in range(0, n_range, _RANGE_CHUNK)]
            with ProcessPoolExecutor(max_workers=min(_MAX_DECODE_WORKERS, len(range_bounds))) as pool:
                futures = [pool.submit(_decode_one_range_block, tmp.name, r0, r1, order,
                                        db_offset, db_scale, has_cross)
                           for r0, r1 in range_bounds]
                for future in as_completed(futures):
                    r0, r1, co_db, cross_db = future.result()
                    arrays["co_byRange"][:, r0:r1, :] = co_db
                    arrays["cross_byRange"][:, r0:r1, :] = cross_db

    store.attrs["source_file"] = znc_gz_path.name
    store.attrs["db_offset"] = db_offset
    store.attrs["db_scale"] = db_scale
    done_marker.write_text("ok")
    return out
