"""catalog_id has to uniquely identify one variable, or catalog_variable()
lookups silently resolve to whichever entry happens to come first -- see
_catalog_id_for's own docstring for the real bug this caused (Munich runs
epsilon-radar through both mira-10 and mira-35; "radar:Zh" from one
collided with the other's).
"""
from prism import gridded_products as gp
from prism.cloudnet_client import RemoteFile


def test_catalog_id_stable_for_combined_products():
    # categorize/classification/etc. have no physical instrument of their
    # own -- the plain id must stay exactly "product:var" (persisted
    # settings.json entries from before instrument-suffixing existed rely
    # on this NOT changing for these products).
    assert gp._catalog_id_for("categorize", "Z", "categorize (combined)") == "categorize:Z"


def test_catalog_id_includes_instrument_for_physical_products():
    # Two instruments publishing the same product must never collide,
    # regardless of which other instruments happen to be present that day
    # (see _catalog_id_for's docstring) -- so this is ALWAYS suffixed, not
    # only when a collision is actually observed.
    id_a = gp._catalog_id_for("epsilon-radar", "epsilon", "mira-10")
    id_b = gp._catalog_id_for("epsilon-radar", "epsilon", "mira-35")
    assert id_a != id_b
    assert id_a == "epsilon-radar:epsilon#mira-10"


def _pv(catalog_id, label, instrument_id="x", var_name="v", product_id="p", **kw):
    defaults = dict(units="", file_path=None, height_dim="height")
    defaults.update(kw)
    return gp.ProductVariable(catalog_id=catalog_id, product_id=product_id, var_name=var_name,
                               label=label, instrument_id=instrument_id, **defaults)


def test_disambiguate_labels_only_touches_real_collisions():
    catalog = [
        _pv("lidar:beta#cl61d", "lidar: Attenuated backscatter coefficient", "cl61d", "beta"),
        _pv("lidar:beta_raw#cl61d", "lidar: Attenuated backscatter coefficient", "cl61d", "beta_raw"),
        _pv("radar:Zh#rpg", "radar: Radar reflectivity factor", "rpg", "Zh"),
    ]
    out = {pv.catalog_id: pv.label for pv in gp._disambiguate_labels(catalog)}
    # Colliding labels get the var_name appended so they're distinguishable
    # in the dropdown/plot titles/hover instead of showing as identical,
    # unpickable entries.
    assert out["lidar:beta#cl61d"] == "lidar: Attenuated backscatter coefficient [beta]"
    assert out["lidar:beta_raw#cl61d"] == "lidar: Attenuated backscatter coefficient [beta_raw]"
    # A label with no collision at all must be left exactly as-is.
    assert out["radar:Zh#rpg"] == "radar: Radar reflectivity factor"


def test_disambiguate_labels_scoped_per_instrument():
    # The SAME label on two DIFFERENT instruments is not a collision --
    # e.g. two radars both reporting "Radar reflectivity factor" is normal
    # and shouldn't get an ugly [Zh] suffix on both.
    catalog = [
        _pv("radar:Zh#rpg-fmcw-94", "radar: Radar reflectivity factor", "rpg-fmcw-94", "Zh"),
        _pv("radar:Zh#mira-35", "radar: Radar reflectivity factor", "mira-35", "Zh"),
    ]
    out = {pv.catalog_id: pv.label for pv in gp._disambiguate_labels(catalog)}
    assert out["radar:Zh#rpg-fmcw-94"] == "radar: Radar reflectivity factor"
    assert out["radar:Zh#mira-35"] == "radar: Radar reflectivity factor"


def test_stub_remote_uses_hardcoded_schema_without_network():
    remote = RemoteFile(uuid="u", filename="f.nc", size=1, checksum="c",
                         download_url="http://example.invalid/never-fetched",
                         instrument_id="rpg-fmcw-94", kind="product", product_id="radar")
    stubs = gp._stub_remote(remote)
    assert stubs, "PRODUCT_SCHEMA should have an entry for 'radar'"
    assert all(pv.file_path is None for pv in stubs)
    assert all(pv.remote is remote for pv in stubs)
    zh = next(pv for pv in stubs if pv.var_name == "Zh")
    assert zh.catalog_id == "radar:Zh#rpg-fmcw-94"
    assert zh.height_dim == "height"


def test_stub_remote_unknown_product_returns_empty():
    remote = RemoteFile(uuid="u", filename="f.nc", size=1, checksum="c",
                         download_url="http://example.invalid", instrument_id=None,
                         kind="product", product_id="some-future-product-type")
    assert gp._stub_remote(remote) == []


def test_to_uniform_height_keeps_true_heights_on_a_nonuniform_radar_grid():
    # RPG gate spacing grows with height (3 m -> 32 m across the chirps). An
    # hv.Image spaces rows by index, so the raw grid drew a real 4247 m cloud
    # top at ~7400 m. After resampling, the last valid gate must sit at its
    # true height.
    import numpy as np
    height = np.concatenate([np.arange(100, 400, 3.0), np.arange(400, 1200, 8.0), np.arange(1200, 11000, 32.0)])
    values = np.full((2, len(height)), np.nan)
    top = int(np.argmin(np.abs(height - 4247)))
    values[:, : top + 1] = 1.0

    grid, resampled = gp.to_uniform_height(height, values)

    assert np.allclose(np.diff(grid), np.diff(grid)[0])
    valid_top = grid[np.isfinite(resampled[0])].max()
    assert abs(valid_top - height[top]) <= 32  # within one coarse gate of the truth
    # a gate keeps its own value: lowest gate maps to the first grid rows
    assert np.isfinite(resampled[0, 0])


def test_to_uniform_height_leaves_a_uniform_grid_alone():
    import numpy as np
    height = np.arange(0.0, 1000.0, 30.0)
    values = np.random.default_rng(0).random((3, len(height)))
    grid, out = gp.to_uniform_height(height, values)
    assert grid is not None and out is values


def test_mwr_multi_and_disdrometer_are_offered():
    assert {"mwr-multi", "disdrometer"} <= gp.CANDIDATE_PRODUCTS
    mwr = RemoteFile(uuid="u", filename="f.nc", size=1, checksum="c", download_url="http://example.invalid",
                     instrument_id="hatpro", kind="product", product_id="mwr-multi")
    by_name = {pv.var_name: pv for pv in gp._stub_remote(mwr)}
    # profiles have a height axis, and temperature is shown in degrees C
    assert by_name["temperature"].height_dim == "height"
    assert by_name["temperature"].units == "°C"
    assert by_name["relative_humidity"].height_dim == "height"

    dis = RemoteFile(uuid="u", filename="f.nc", size=1, checksum="c", download_url="http://example.invalid",
                     instrument_id="parsivel", kind="product", product_id="disdrometer")
    stubs = gp._stub_remote(dis)
    assert stubs and all(pv.height_dim is None for pv in stubs)  # time series only


def test_load_curtain_converts_kelvin_temperature_to_celsius(tmp_path):
    # plot_meta's temperature colour range is in degrees C (ported from
    # CloudnetPy), the product files are in K.
    import numpy as np
    import xarray as xr
    times = np.array(["2024-01-10T11:00", "2024-01-10T11:10"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"temperature": (("time", "height"), np.full((2, 3), 273.15 - 20.0), {"units": "K"}),
         "potential_temperature": (("time", "height"), np.full((2, 3), 300.0), {"units": "K"})},
        coords={"time": times, "height": [10.0, 20.0, 30.0]})
    path = tmp_path / "mwr.nc"
    ds.to_netcdf(path)

    def pv(name, units):
        return gp.ProductVariable(catalog_id=f"mwr-multi:{name}#hatpro", product_id="mwr-multi", var_name=name,
                                   label=name, units=units, file_path=path, instrument_id="hatpro",
                                   height_dim="height")

    _, _, temp = gp.load_curtain(pv("temperature", "°C"))
    assert np.allclose(temp, -20.0)
    _, _, theta = gp.load_curtain(pv("potential_temperature", "K"))
    assert np.allclose(theta, 300.0)  # only the °C-scaled names are converted


def _model_file(tmp_path):
    import numpy as np
    import xarray as xr
    times = np.array(["2024-01-10T11:00", "2024-01-10T12:00"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"temperature": (("time", "level"), np.array([[280.0, 270.0], [290.0, 280.0]]), {"units": "K"}),
         "sfc_temp_2m": (("time",), np.array([270.0, 280.0]), {"units": "K"}),
         "height": (("time", "level"), np.array([[10.0, 1000.0], [12.0, 1010.0]]), {"units": "m"})},
        coords={"time": times, "level": [1, 2]})
    path = tmp_path / "model.nc"
    ds.to_netcdf(path)
    return path


def _model_pv(name, path, height_dim):
    return gp.ProductVariable(catalog_id=f"model:{name}#ecmwf", product_id="model", var_name=name, label=name,
                               units="", file_path=path, instrument_id="ecmwf", height_dim=height_dim)


def test_model_hourly_file_is_interpolated_across_the_window(tmp_path):
    # An hourly model file has ONE sample inside a one-hour window -- too few
    # for an image or a curve, so it is interpolated linearly in time.
    import numpy as np
    path = _model_file(tmp_path)
    t0, t1 = np.datetime64("2024-01-10T11:00:00"), np.datetime64("2024-01-10T11:30:00")

    times, height, temp = gp.load_curtain(_model_pv("temperature", path, "level"), t0, t1)

    assert len(times) == 31 and times[0] == t0 and times[-1] == t1  # per minute, exact endpoints
    assert np.allclose(height, [10.0, 1000.0])  # heights from the 2D "height" field, not level numbers
    assert np.allclose(temp[0], np.array([280.0, 270.0]) - 273.15)  # K -> degrees C
    assert np.allclose(temp[-1], np.array([285.0, 275.0]) - 273.15)  # halfway to the 12:00 sample

    _, no_height, surface = gp.load_curtain(_model_pv("sfc_temp_2m", path, None), t0, t1)
    assert no_height is None and np.isclose(surface[-1], 275.0 - 273.15)


def test_model_variables_are_offered_from_the_model_endpoint():
    assert "model" in gp.CANDIDATE_PRODUCTS and "level" in gp.HEIGHT_DIM_NAMES
    remote = RemoteFile(uuid="u", filename="f.nc", size=1, checksum="c", download_url="http://example.invalid",
                        instrument_id="ecmwf", kind="product", product_id="model")
    by_name = {pv.var_name: pv for pv in gp._stub_remote(remote)}
    assert by_name["temperature"].catalog_id == "model:temperature#ecmwf"
    assert by_name["temperature"].height_dim == "height"  # stubs use the generic name; a real scan finds "level"
    assert by_name["sfc_temp_2m"].height_dim is None
