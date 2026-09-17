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
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from prism.rpg_reader import MASK_CODE, _encode_db

# Decoding one hour materializes ~5 GB (uncompressed) per channel at MIRA-10's
# 4096 Doppler bins; process in time chunks so peak memory stays under ~1 GB
# instead of loading the whole hour into memory at once.
_TIME_CHUNK = 50


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

            for start in range(0, n_time, _TIME_CHUNK):
                stop = min(start + _TIME_CHUNK, n_time)
                co_db = _encode_db(ds["SPCco"].isel(time=slice(start, stop)).values[:, :, order])
                if has_cross:
                    cross_db = _encode_db(ds["SPCcx"].isel(time=slice(start, stop)).values[:, :, order])
                else:
                    # This instrument/configuration doesn't export a
                    # cross-channel spectrum -- leave it fully masked rather
                    # than fabricating data; the app already renders a
                    # masked channel as "no data" cleanly.
                    cross_db = np.full_like(co_db, MASK_CODE)
                arrays["co_byTime"][start:stop] = co_db
                arrays["co_byRange"][start:stop] = co_db
                arrays["cross_byTime"][start:stop] = cross_db
                arrays["cross_byRange"][start:stop] = cross_db

    store.attrs["source_file"] = znc_gz_path.name
    done_marker.write_text("ok")
    return out
