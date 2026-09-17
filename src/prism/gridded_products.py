"""Auto-discover and load every time-height ("curtain") variable Cloudnet
publishes for a site/day -- radar moments, target classification, lidar
backscatter, categorize-derived model temperature, etc. -- whatever actually
exists for that site, instead of a hardcoded list of radar moments.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

from . import cloudnet_client as cc
from . import plot_meta

# Products worth scanning for curtain (time-height) or time-only variables.
CANDIDATE_PRODUCTS = {
    "radar", "categorize", "classification", "lidar",
    "iwc", "lwc", "drizzle", "der", "ier", "epsilon-radar",
    "mwr",  # time-only products, e.g. liquid water path (lwp)
}

# A variable qualifies as a "curtain" if its dims look like (time, <height>).
HEIGHT_DIM_NAMES = {"height", "range"}


@dataclass(frozen=True)
class ProductVariable:
    catalog_id: str  # "<product_id>:<var_name>", used as the dropdown value
    product_id: str
    var_name: str
    label: str  # human-readable, e.g. "radar: Radar reflectivity factor"
    units: str
    file_path: Path
    instrument_id: str  # physical instrument, or "<product_id> (combined)"
    height_dim: str | None  # None for a time-only (1D) variable, e.g. mwr lwp
    is_categorical: bool = False
    cmap: str = "viridis"
    plot_range: tuple[float, float] | None = None
    log_scale: bool = False


def discover(site: str, day: date, cache_dir: Path = cc.DEFAULT_CACHE_DIR, on_progress=None) -> list[ProductVariable]:
    """List every curtain (time-height) and time-only variable available for
    this site/day, downloading (and caching) each candidate product file as
    needed.

    on_progress(filename, downloaded_bytes, total_bytes), if given, is
    forwarded to each download so the caller can drive a status line.
    """
    remotes = [r for r in cc.list_product_files(site, day) if r.product_id in CANDIDATE_PRODUCTS]
    catalog: list[ProductVariable] = []
    for remote in remotes:
        path = cc.ensure_downloaded(remote, cache_dir, on_progress=on_progress)
        try:
            ds = xr.open_dataset(path)
        except Exception:
            continue
        # Multi-instrument sites can publish the same product (e.g. "radar")
        # from more than one instrument -- group by the actual instrument so
        # the variable picker can show "instrument > variable" instead of a
        # flat list. Instrument-agnostic (derived/combined) products like
        # classification or categorize have no instrument of their own.
        instrument = remote.instrument_id or f"{remote.product_id} (combined)"
        for var_name, da in ds.data_vars.items():
            if "time" not in da.dims:
                continue
            height_dim = next((d for d in da.dims if d in HEIGHT_DIM_NAMES), None)
            if height_dim is not None and da.ndim == 2:
                pass  # a genuine (time, height) curtain
            elif da.ndim == 1 and da.dims == ("time",):
                height_dim = None  # time-only series, e.g. mwr lwp
            else:
                continue
            label = f"{remote.product_id}: {da.attrs.get('long_name', var_name)}"
            is_categorical = var_name in plot_meta.CATEGORICAL
            cont = plot_meta.CONTINUOUS.get(var_name)
            catalog.append(ProductVariable(
                catalog_id=f"{remote.product_id}:{var_name}",
                product_id=remote.product_id,
                var_name=var_name,
                label=label,
                units=da.attrs.get("units", ""),
                file_path=path,
                instrument_id=instrument,
                height_dim=height_dim,
                is_categorical=is_categorical,
                cmap=cont.cmap if cont else "viridis",
                plot_range=cont.plot_range if cont else None,
                log_scale=cont.log_scale if cont else False,
            ))
        ds.close()
    return sorted(catalog, key=lambda p: p.label)


@lru_cache(maxsize=32)
def _load_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def load_curtain(pv: ProductVariable, t_start=None, t_stop=None):
    """Return (time[datetime64], height[m] or None, values) sliced to an
    optional time window (e.g. the currently selected hour). height is None
    and values is 1D (time,) for a time-only variable (pv.height_dim is
    None), e.g. mwr's lwp -- the caller decides how to display that."""
    ds = _load_dataset(str(pv.file_path))
    da = ds[pv.var_name]
    if t_start is not None and t_stop is not None:
        da = da.sel(time=slice(t_start, t_stop))
    if pv.height_dim is None:
        return da["time"].values, None, da.values
    height = ds[pv.height_dim].values
    # height/range can itself be time-varying in some products; if so, use
    # the first profile as a static axis (adequate for a single-hour view).
    if height.ndim > 1:
        height = height[0]
    return da["time"].values, np.asarray(height), da.values


def nearest_index(coord: np.ndarray, value) -> int:
    return int(np.argmin(np.abs(coord.astype("int64") - np.asarray(value).astype("int64")))) \
        if np.issubdtype(coord.dtype, np.datetime64) else int(np.argmin(np.abs(coord - value)))
