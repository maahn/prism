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
