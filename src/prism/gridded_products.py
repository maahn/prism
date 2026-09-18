"""Auto-discover and load every time-height ("curtain") variable Cloudnet
publishes for a site/day -- radar moments, target classification, lidar
backscatter, categorize-derived model temperature, etc. -- whatever actually
exists for that site, instead of a hardcoded list of radar moments.

discover() is deliberately two-speed: it downloads and scans only the
EAGER products (the ones the default/persisted panels actually show) so
Load isn't stuck downloading a dozen products' files before the first
pixel appears. Every other available product gets a STUB ProductVariable
built from PRODUCT_SCHEMA below -- a hardcoded table of what CloudnetPy
publishes for each product, good enough to populate the variable dropdown
immediately. Picking a stub variable triggers resolve_product() to
download and scan that ONE file on demand (see AppState.resolve_variable
in app.py), after which it behaves exactly like an eagerly-discovered one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
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

# Site/instrument housekeeping fields that are always constant over an hour
# (position, hardware calibration, temperatures inside the enclosure, ...).
# Hardcoded rather than detected at runtime -- the runtime check (was: load
# the full array, test np.all(values == values[0])) only works on a file
# we've already downloaded, which is exactly what discover() now tries to
# avoid doing for every product up front. Kept deliberately conservative:
# only names that are NEVER scientifically interesting to plot as a time
# series are listed here, e.g. zenith_angle is left OUT because at least
# one real instrument (a scanning microwave radiometer) genuinely varies it.
CONSTANT_VARIABLE_NAMES = {
    "altitude", "latitude", "longitude",
    "wavelength", "lidar_wavelength", "radar_frequency",
    "calibration_interval", "program_number", "sample_duration",
    "voltage", "pc_temperature", "receiver_temperature",
    "transmitter_temperature", "if_power", "tpow", "transmitted_power",
    "time_ms",
}


@dataclass(frozen=True)
class VarSchema:
    label_suffix: str  # long_name equivalent, e.g. "Radar reflectivity factor"
    units: str
    has_height: bool
    is_categorical: bool = False


# Hardcoded fallback variable list per product, used to build STUB catalog
# entries for a product that hasn't been downloaded yet. Compiled from real
# CloudnetPy output (radar: RPG-FMCW-94/MIRA-10/MIRA-35; lidar: CL61D/CHM15k;
# mwr: HATPRO) across several sites -- see discover()'s docstring. A stub's
# fields are only ever a starting point: resolve_product() always prefers
# whatever the real file says once it's actually downloaded, so an outdated
# or missing schema entry here degrades to "not offered until scanned",
# never to wrong data being shown.
PRODUCT_SCHEMA: dict[str, dict[str, VarSchema]] = {
    "radar": {
        "Zh": VarSchema("Radar reflectivity factor", "dBZ", True),
        "v": VarSchema("Doppler velocity", "m s-1", True),
        "width": VarSchema("Spectral width", "m s-1", True),
        "ldr": VarSchema("Linear depolarisation ratio", "dB", True),
        "sldr": VarSchema("Slanted linear depolarisation ratio", "dB", True),
        "SNR": VarSchema("Signal-to-noise ratio", "dB", True),
        "kurtosis": VarSchema("Kurtosis of spectra", "1", True),
        "skewness": VarSchema("Skewness of spectra", "1", True),
        "nyquist_velocity": VarSchema("Nyquist velocity", "m s-1", True),
        "rho_hv": VarSchema("Correlation coefficient", "1", True),
        "srho_hv": VarSchema("Slanted correlation coefficient", "1", True),
        "zdr": VarSchema("Differential reflectivity", "dB", True),
        "kdp": VarSchema("Specific differential phase shift", "rad km-1", True),
        "phi_dp": VarSchema("Differential phase", "rad", True),
        "phi_cx": VarSchema("Co-cross-channel differential phase", "rad", True),
        "rho_cx": VarSchema("Co-cross-channel correlation coefficient", "1", True),
        "differential_attenuation": VarSchema("Differential attenuation", "dB km-1", True),
        "lwp": VarSchema("Liquid water path", "kg m-2", False),
        "rainfall_rate": VarSchema("Rainfall rate", "m s-1", False),
        "air_pressure": VarSchema("Air pressure", "Pa", False),
        "air_temperature": VarSchema("Air temperature", "K", False),
        "brightness_temperature": VarSchema("Brightness temperature", "K", False),
        "relative_humidity": VarSchema("Relative humidity", "1", False),
        "wind_speed": VarSchema("Wind speed", "m s-1", False),
        "wind_direction": VarSchema("Wind direction", "degree", False),
        "quality_flag": VarSchema("Quality flag", "1", False),
    },
    "categorize": {
        "Z": VarSchema("Radar reflectivity factor", "dBZ", True),
        "Z_error": VarSchema("Error in radar reflectivity factor", "dB", True),
        "v": VarSchema("Doppler velocity", "m s-1", True),
        "v_sigma": VarSchema("Standard deviation of mean Doppler velocity", "m s-1", True),
        "width": VarSchema("Spectral width", "m s-1", True),
        "ldr": VarSchema("Linear depolarisation ratio", "dB", True),
        "beta": VarSchema("Attenuated backscatter coefficient", "sr-1 m-1", True),
        "Tw": VarSchema("Wet-bulb temperature", "K", True),
        "insect_prob": VarSchema("Insect probability", "1", True),
        "category_bits": VarSchema("Target categorization bits", "1", True),
        "quality_bits": VarSchema("Data quality bits", "1", True),
        "nyquist_velocity": VarSchema("Nyquist velocity", "m s-1", True),
        "radar_gas_atten": VarSchema("Two-way radar attenuation due to atmospheric gases", "dB", True),
        "radar_liquid_atten": VarSchema("Two-way radar attenuation due to liquid water", "dB", True),
        "radar_rain_atten": VarSchema("Two-way radar attenuation due to rain", "dB", True),
        "radar_melting_atten": VarSchema("Two-way radar attenuation due to melting ice", "dB", True),
        "lwp": VarSchema("Liquid water path", "kg m-2", False),
        "lwp_error": VarSchema("Error in liquid water path", "kg m-2", False),
        "rain_detected": VarSchema("Rain detected", "1", False),
        "rainfall_rate": VarSchema("Rainfall rate", "m s-1", False),
        "synop_WaWa": VarSchema("Synop code WaWa", "1", False),
    },
    "classification": {
        "target_classification": VarSchema("Target classification", "1", True, is_categorical=True),
        "detection_status": VarSchema("Radar and lidar detection status", "1", True, is_categorical=True),
        "radar_attenuation_status": VarSchema("Radar attenuation status", "1", True, is_categorical=True),
        "signal_source_status": VarSchema("Signal source status", "1", True, is_categorical=True),
        "cloud_base_height_agl": VarSchema("Height of cloud base above ground level", "m", False),
        "cloud_base_height_amsl": VarSchema("Height of cloud base above mean sea level", "m", False),
        "cloud_top_height_agl": VarSchema("Height of cloud top above ground level", "m", False),
        "cloud_top_height_amsl": VarSchema("Height of cloud top above mean sea level", "m", False),
        "cloud_top_height_status": VarSchema("Cloud top height quality status", "1", False),
        "rain_detected": VarSchema("Rain detected", "1", False),
    },
    "lidar": {
        "beta": VarSchema("Attenuated backscatter coefficient", "sr-1 m-1", True),
        "beta_raw": VarSchema("Attenuated backscatter coefficient", "sr-1 m-1", True),
        "beta_smooth": VarSchema("Attenuated backscatter coefficient", "sr-1 m-1", True),
        "depolarisation": VarSchema("Lidar volume linear depolarisation ratio", "1", True),
        "depolarisation_raw": VarSchema("Lidar volume linear depolarisation ratio", "1", True),
    },
    "mwr": {
        "lwp": VarSchema("Liquid water path", "kg m-2", False),
        "iwv": VarSchema("Integrated water vapour", "kg m-2", False),
        "zenith_angle": VarSchema("Zenith angle", "degree", False),
    },
    "epsilon-radar": {
        "epsilon": VarSchema("Dissipation rate of turbulent kinetic energy", "m2 s-3", True),
        "epsilon_error": VarSchema("Absolute error in dissipation rate of turbulent kinetic energy", "m2 s-3", True),
    },
    "iwc": {
        "iwc": VarSchema("Ice water content", "kg m-3", True),
        "iwc_error": VarSchema("Random error in ice water content", "dB", True),
        "iwc_retrieval_status": VarSchema("Ice water content retrieval status", "1", True, is_categorical=True),
    },
    "lwc": {
        "lwc": VarSchema("Liquid water content", "kg m-3", True),
        "lwc_error": VarSchema("Random error in liquid water content, one standard deviation", "dB", True),
        "lwc_retrieval_status": VarSchema("Liquid water content retrieval status", "1", True, is_categorical=True),
        "lwp": VarSchema("Liquid water path", "kg m-2", False),
        "lwp_error": VarSchema("Error in liquid water path", "kg m-2", False),
    },
    "drizzle": {
        "Do": VarSchema("Drizzle median diameter", "m", True),
        "Do_error": VarSchema("Random error in drizzle median diameter", "dB", True),
        "S": VarSchema("Lidar backscatter-to-extinction ratio", "sr", True),
        "S_error": VarSchema("Random error in lidar backscatter-to-extinction ratio", "dB", True),
        "beta_corr": VarSchema("Lidar backscatter correction factor", "1", True),
        "drizzle_N": VarSchema("Drizzle number concentration", "m-3", True),
        "drizzle_N_error": VarSchema("Random error in drizzle number concentration", "dB", True),
        "drizzle_lwc": VarSchema("Drizzle liquid water content", "kg m-3", True),
        "drizzle_lwc_error": VarSchema("Random error in drizzle liquid water content", "dB", True),
        "drizzle_lwf": VarSchema("Drizzle liquid water flux", "kg m-2 s-1", True),
        "drizzle_lwf_error": VarSchema("Random error in drizzle liquid water flux", "dB", True),
        "drizzle_retrieval_status": VarSchema("Drizzle parameter retrieval status", "1", True, is_categorical=True),
        "mu": VarSchema("Drizzle droplet size distribution shape parameter", "1", True),
        "mu_error": VarSchema("Random error in drizzle droplet size distribution shape parameter", "dB", True),
        "v_air": VarSchema("Vertical air velocity", "m s-1", True),
        "v_drizzle": VarSchema("Drizzle droplet fall velocity", "m s-1", True),
        "v_drizzle_error": VarSchema("Random error in drizzle droplet fall velocity", "dB", True),
    },
    "der": {
        "der": VarSchema("Droplet effective radius", "m", True),
        "der_error": VarSchema("Absolute error in droplet effective radius", "m", True),
        "der_retrieval_status": VarSchema("Droplet effective radius retrieval status", "1", True, is_categorical=True),
        "der_scaled": VarSchema("Droplet effective radius (scaled to LWP)", "m", True),
        "der_scaled_error": VarSchema("Absolute error in droplet effective radius (scaled to LWP)", "m", True),
        "N_scaled": VarSchema("Cloud droplet number concentration", "m-3", True),
    },
    "ier": {
        "ier": VarSchema("Ice effective radius", "m", True),
        "ier_error": VarSchema("Random error in ice effective radius", "m", True),
        "ier_retrieval_status": VarSchema("Ice effective radius retrieval status", "1", True, is_categorical=True),
    },
}

# Only these products are downloaded/scanned on the initial Load; everything
# else in CANDIDATE_PRODUCTS is offered from PRODUCT_SCHEMA as a stub and
# fetched lazily. Kept in sync with DEFAULT_PANEL_VARIABLES in app.py.
DEFAULT_EAGER_PRODUCTS = {"categorize"}


@dataclass(frozen=True)
class ProductVariable:
    catalog_id: str  # "<product_id>:<var_name>", used as the dropdown value
    product_id: str
    var_name: str
    label: str  # human-readable, e.g. "radar: Radar reflectivity factor"
    units: str
    file_path: Path | None  # None until resolve_product() downloads it
    instrument_id: str  # physical instrument, or "<product_id> (combined)"
    height_dim: str | None  # None for a time-only (1D) variable, e.g. mwr lwp
    is_categorical: bool = False
    cmap: str = "viridis"
    plot_range: tuple[float, float] | None = None
    log_scale: bool = False
    remote: cc.RemoteFile | None = None  # set on stubs, for lazy resolution


def discover(
    site: str, day: date, cache_dir: Path = cc.DEFAULT_CACHE_DIR, on_progress=None,
    eager_products: set[str] = DEFAULT_EAGER_PRODUCTS,
) -> list[ProductVariable]:
    """List every curtain (time-height) and time-only variable available for
    this site/day.

    Only products in `eager_products` are actually downloaded and scanned
    here (their real variables, from the file itself). Every other
    available product gets STUB entries built from PRODUCT_SCHEMA -- no
    network access -- so the dropdown can offer them immediately; picking
    one resolves it on demand via resolve_product().

    on_progress(filename, downloaded_bytes, total_bytes), if given, is
    forwarded to each eager download so the caller can drive a status line.
    """
    remotes = [r for r in cc.list_product_files(site, day) if r.product_id in CANDIDATE_PRODUCTS]
    catalog: list[ProductVariable] = []
    for remote in remotes:
        if remote.product_id in eager_products:
            catalog.extend(_scan_remote(remote, cache_dir, on_progress=on_progress))
        else:
            catalog.extend(_stub_remote(remote))
    catalog = _disambiguate_labels(catalog)
    # 2D (time, height) curtains before 1D (time)-only series, so the more
    # commonly wanted moments/curtains aren't buried under LWP-style
    # metadata variables (site altitude, latitude, etc.) that also qualify
    # as "time-only" and would otherwise interleave alphabetically.
    return sorted(catalog, key=lambda p: (p.height_dim is None, p.label))


def resolve_product(pv: ProductVariable, catalog: list[ProductVariable],
                     cache_dir: Path = cc.DEFAULT_CACHE_DIR, on_progress=None) -> list[ProductVariable]:
    """Download and scan the one product `pv` belongs to (a no-op if it's
    already real), replacing every stub entry for that (product, instrument)
    pair in `catalog` with the real thing -- called the first time the user
    picks a not-yet-downloaded variable from the dropdown. Real data always
    wins over PRODUCT_SCHEMA's guess, so a stale or missing schema entry can
    only ever mean "not offered until picked", never wrong data."""
    if pv.file_path is not None or pv.remote is None:
        return catalog
    fresh = _scan_remote(pv.remote, cache_dir, on_progress=on_progress)
    kept = [p for p in catalog
            if not (p.product_id == pv.product_id and p.instrument_id == pv.instrument_id)]
    return _disambiguate_labels(kept + fresh)


def _catalog_id_for(product_id: str, var_name: str, instrument: str) -> str:
    """Uniquely identifies one variable for the dropdown/settings/lookups.

    Derived/combined products (categorize, classification, drizzle, iwc,
    lwc, der, ier) publish at most one file per product per day, so the
    plain "product:var" id is already unique. Physical-instrument products
    (radar, lidar, mwr) can have MORE than one instrument publishing the
    same product at once -- e.g. Munich runs epsilon-radar through both
    mira-10 and mira-35 -- so "radar:Zh" from one collided with the other's,
    and catalog_variable() lookups (keyed only by this id) would silently
    resolve to whichever one happened to come first in the list, regardless
    of which instrument's submenu the user actually picked. Always
    including the real instrument in the id here (never only when a
    collision happens to be detected) keeps it collision-proof AND stable
    across sessions/sites: the alternative -- suffixing only on an observed
    collision -- would give the same variable a different id depending on
    which OTHER instruments happen to be present that day, breaking
    settings.json's persisted choice unpredictably.
    """
    if instrument.endswith(" (combined)"):
        return f"{product_id}:{var_name}"
    return f"{product_id}:{var_name}#{instrument}"


def _scan_remote(remote: cc.RemoteFile, cache_dir: Path, on_progress=None) -> list[ProductVariable]:
    """Download (if needed), open, and scan one product file into real
    ProductVariable entries."""
    try:
        path = cc.ensure_downloaded(remote, cache_dir, on_progress=on_progress)
        ds = xr.open_dataset(path)
    except Exception:
        # A single flaky/oversized download (or a file that fails to parse)
        # shouldn't abort discovery of every OTHER product for this
        # site/day -- skip it and keep going, the same way a missing hour
        # or variable degrades gracefully elsewhere in this app rather than
        # blocking the whole load.
        return []
    # Multi-instrument sites can publish the same product (e.g. "radar")
    # from more than one instrument -- group by the actual instrument so
    # the variable picker can show "instrument > variable" instead of a
    # flat list. Instrument-agnostic (derived/combined) products like
    # classification or categorize have no instrument of their own.
    instrument = remote.instrument_id or f"{remote.product_id} (combined)"
    out: list[ProductVariable] = []
    for var_name, da in ds.data_vars.items():
        if "time" not in da.dims or var_name in CONSTANT_VARIABLE_NAMES:
            continue
        height_dim = next((d for d in da.dims if d in HEIGHT_DIM_NAMES), None)
        if height_dim is not None and da.ndim == 2:
            pass  # a genuine (time, height) curtain
        elif da.ndim == 1 and da.dims == ("time",):
            height_dim = None  # time-only series, e.g. mwr lwp
        else:
            continue
        # No dimension suffix here -- this label is reused for plot titles,
        # the categorical legend, and the curve y-axis label, where it
        # would just be noise. app._dropdown_label() appends
        # "(time, height)"/"(time)" only for the dropdown's own display
        # text.
        label = f"{remote.product_id}: {da.attrs.get('long_name', var_name)}"
        is_categorical = var_name in plot_meta.CATEGORICAL
        cont = plot_meta.CONTINUOUS.get(var_name)
        out.append(ProductVariable(
            catalog_id=_catalog_id_for(remote.product_id, var_name, instrument),
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
    return out


def _stub_remote(remote: cc.RemoteFile) -> list[ProductVariable]:
    """Build catalog entries for a product from PRODUCT_SCHEMA, without
    downloading anything. Falls back to an empty list for a product this
    app doesn't have a hardcoded schema for yet -- it simply won't appear
    in the dropdown until PRODUCT_SCHEMA gains an entry, same as any
    variable this app has never seen would previously go undiscovered."""
    schema = PRODUCT_SCHEMA.get(remote.product_id)
    if not schema:
        return []
    instrument = remote.instrument_id or f"{remote.product_id} (combined)"
    out = []
    for var_name, s in schema.items():
        cont = plot_meta.CONTINUOUS.get(var_name)
        out.append(ProductVariable(
            catalog_id=_catalog_id_for(remote.product_id, var_name, instrument),
            product_id=remote.product_id,
            var_name=var_name,
            label=f"{remote.product_id}: {s.label_suffix}",
            units=s.units,
            file_path=None,
            instrument_id=instrument,
            height_dim="height" if s.has_height else None,
            is_categorical=s.is_categorical,
            cmap=cont.cmap if cont else "viridis",
            plot_range=cont.plot_range if cont else None,
            log_scale=cont.log_scale if cont else False,
            remote=remote,
        ))
    return out


def _disambiguate_labels(catalog: list[ProductVariable]) -> list[ProductVariable]:
    """CloudnetPy sometimes gives several distinct variables of the same
    product the exact same long_name -- e.g. the lidar product's beta_raw
    (non-screened), beta (SNR-screened) and beta_smooth (screened + Gaussian-
    smoothed) all carry long_name "Attenuated backscatter coefficient"; the
    only place that distinction lives is each variable's own `comment`
    attribute, which this app never reads, and its var_name. Without this,
    the dropdown -- and every plot title, hover, and curve label built from
    ProductVariable.label -- would show three identical, unpickable entries.
    So append the variable's own name wherever a label collides with
    another's within the same instrument."""
    counts = Counter((pv.instrument_id, pv.label) for pv in catalog)
    return [
        replace(pv, label=f"{pv.label} [{pv.var_name}]") if counts[(pv.instrument_id, pv.label)] > 1 else pv
        for pv in catalog
    ]


@lru_cache(maxsize=32)
def _load_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def load_curtain(pv: ProductVariable, t_start=None, t_stop=None):
    """Return (time[datetime64], height[m] or None, values) sliced to an
    optional time window (e.g. the currently selected hour). height is None
    and values is 1D (time,) for a time-only variable (pv.height_dim is
    None), e.g. mwr's lwp -- the caller decides how to display that.

    `pv.file_path` must already be resolved (not None) -- callers go through
    AppState.resolve_variable() first, which downloads a still-stubbed
    product on demand before ever reaching here."""
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
