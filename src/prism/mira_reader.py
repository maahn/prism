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

# Reading the raw file is CHUNKED (rather than one bulk read like rpgpy's
# for RPG) because MIRA's .znc is HDF5 with the SPCco/SPCcx variables
# gzip-compressed internally (confirmed via h5py: chunks=(1, 175, 2048),
# compression="gzip") -- there's no equivalent "read everything in one
# native call" path the way rpgpy.read_rpg() has for RPG's raw, uncompressed
# LV0 binary; every read has to decompress whichever HDF5 chunks it
# touches. Process in time chunks so a single worker's peak memory (the
# float32 chunk it reads, before quantizing to uint8) stays modest.
_TIME_CHUNK = 50
# Measured on a real Munich mira-10 hour: reading+decompressing one
# 50-profile chunk takes ~0.7s, but _encode_db-ing it takes ~3s PER channel
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

            # ONE pass over the source file, time-chunked and parallelized
            # across processes (see _decode_one_chunk). co_byRange/
            # cross_byRange used to be populated by a SECOND pass, sliced
            # along range instead of time to match that array's own zarr
            # chunking (n_time, 1, n_doppler) and avoid a read-modify-write
            # -- correct, but it meant reading and decompressing the
            # source file twice in full, and decompression is most of
            # what makes MIRA's decode slow to begin with (see
            # rpg_reader.py's module docstring / _TIME_CHUNK's comment
            # above for why: gzip-compressed HDF5, unlike RPG's raw
            # binary). Since the DECODED (uint8) data is 4x smaller than
            # what was just read (uint8 vs the source's float32), this
            # keeps the full decoded arrays in memory instead -- typically
            # a couple GB for one hour -- and populates co_byRange/
            # cross_byRange from THAT afterwards with plain numpy slicing,
            # no second read at all. That's exactly how rpg_reader.py
            # already populates both layouts from one in-memory result;
            # MIRA just needs the chunked read first to build that result,
            # since it has no equivalent of rpgpy's one-shot whole-file
            # read.
            co_full = np.empty((n_time, n_range, n_doppler), dtype="uint8")
            cross_full = np.empty((n_time, n_range, n_doppler), dtype="uint8")
            time_bounds = [(s, min(s + _TIME_CHUNK, n_time)) for s in range(0, n_time, _TIME_CHUNK)]
            with ProcessPoolExecutor(max_workers=min(_MAX_DECODE_WORKERS, len(time_bounds))) as pool:
                futures = [pool.submit(_decode_one_chunk, tmp.name, start, stop, order,
                                        db_offset, db_scale, has_cross)
                           for start, stop in time_bounds]
                for future in as_completed(futures):
                    start, stop, co_db, cross_db = future.result()
                    arrays["co_byTime"][start:stop] = co_db
                    arrays["cross_byTime"][start:stop] = cross_db
                    co_full[start:stop] = co_db
                    cross_full[start:stop] = cross_db

            # A full-array write still lands on complete, non-overlapping
            # zarr chunks (each of co_byRange's chunks spans the FULL time
            # axis for one range gate, and this write covers the full time
            # axis for every range gate at once), so no read-modify-write
            # here either.
            arrays["co_byRange"][:] = co_full
            arrays["cross_byRange"][:] = cross_full
            del co_full, cross_full

    store.attrs["source_file"] = znc_gz_path.name
    store.attrs["db_offset"] = db_offset
    store.attrs["db_scale"] = db_scale
    done_marker.write_text("ok")
    return out
