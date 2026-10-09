"""Persist user-adjustable UI state (selected variables, color/axis limits,
co/cross toggle) across sessions, with per-panel and global reset.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from prism import cloudnet_client as cc

DEFAULT_SETTINGS_PATH = Path(os.environ.get(
    "PRISM_SETTINGS",
    Path.home() / ".config" / "prism" / "settings.json",
))

# "auto" means: compute from the currently loaded data each time, rather
# than a fixed stored number.
DEFAULT_PANEL_STATE = {
    "variable": None,       # catalog_id, e.g. "radar:Zh"
    "color_limits": "auto",  # "auto" or [vmin, vmax]
    "axis_limits": "auto",   # "auto" or [xmin, xmax, ymin, ymax]
}

DEFAULTS: dict[str, Any] = {
    "moment_panels": [dict(DEFAULT_PANEL_STATE) for _ in range(3)],
    "spectra_channel": "co",  # "co" or "cross"
    "time_spectrogram_window_s": 600,
    "cache_dir": None,  # None -> cloudnet_client.DEFAULT_CACHE_DIR
    "path_pattern": None,  # e.g. "/data/{site_flat}/cloudnet/{product}/{year}/{file}"
    "site_dirs": {},    # site id -> existing data directory used as that site's cache
    "keep_existing": False,  # True: never replace a product file already on disk, even if the portal's differs
    "last_session": {"site": "hyytiala", "day": "2024-02-16", "hour": 13, "instrument": "rpg-fmcw-94"},
}


# Keys a --config file may set: where data lives, not user preferences.
CONFIG_KEYS = ("path_pattern", "cache_dir", "site_dirs", "keep_existing")


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a deployment config file (JSON with `cache_dir` and/or
    `site_dirs`, see README "Shared data directories"). Raises ValueError
    for anything unreadable or unknown, so a typo fails at startup rather
    than silently falling back to the default cache."""
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} must contain a JSON object")
    unknown = set(config) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) in {path}: {', '.join(sorted(unknown))} "
                         f"(allowed: {', '.join(CONFIG_KEYS)})")
    site_dirs = config.get("site_dirs", {})
    if not isinstance(site_dirs, dict):
        raise ValueError(f"'site_dirs' in {path} must map site ids to path patterns")
    if config.get("cache_dir") is not None and not isinstance(config["cache_dir"], str):
        raise ValueError(f"'cache_dir' in {path} must be a string")
    try:
        for pattern in [config.get("path_pattern"), *site_dirs.values()]:
            if pattern is not None:
                cc.check_path_pattern(pattern)
    except ValueError as exc:
        raise ValueError(f"in {path}: {exc}") from exc
    if not isinstance(config.get("keep_existing", False), bool):
        raise ValueError(f"'keep_existing' in {path} must be true or false")
    return config


class Settings:
    """path=None keeps everything in memory only (never reads or writes a
    file): server mode gives every user session its own throwaway Settings,
    since sessions must neither share nor overwrite one settings.json.

    `overrides` (from a --config file) win over both the defaults and the
    stored file, and are never written back to it."""

    def __init__(self, path: Path | None = DEFAULT_SETTINGS_PATH, overrides: dict | None = None):
        self.path = path
        self.overrides = dict(overrides or {})
        self._data = self._load()
        self._data.update(copy.deepcopy(self.overrides))

    @property
    def data_dirs_configured(self) -> bool:
        """True when the cache location was set explicitly (settings file or
        --config) rather than defaulting to ~/.cache/prism. Such a directory
        may be a shared archive, so PRISM neither measures nor cleans it."""
        return bool(self._data.get("cache_dir") or self._data.get("site_dirs") or self._data.get("path_pattern"))

    def _load(self) -> dict:
        if self.path is not None and self.path.exists():
            try:
                stored = json.loads(self.path.read_text())
                merged = copy.deepcopy(DEFAULTS)
                merged.update(stored)
                return merged
            except (json.JSONDecodeError, OSError):
                pass
        return copy.deepcopy(DEFAULTS)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stored = {k: v for k, v in self._data.items() if k not in self.overrides}
        self.path.write_text(json.dumps(stored, indent=2))

    def get(self, key: str) -> Any:
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def get_panel(self, index: int) -> dict:
        return self._data["moment_panels"][index]

    def set_panel(self, index: int, **updates) -> None:
        self._data["moment_panels"][index].update(updates)
        self.save()

    def reset_panel(self, index: int) -> None:
        self._data["moment_panels"][index] = dict(DEFAULT_PANEL_STATE)
        self.save()

    def reset_all(self) -> None:
        self._data = copy.deepcopy(DEFAULTS)
        self._data.update(copy.deepcopy(self.overrides))
        self.save()

    def set_last_session(self, site: str, day: str, hour: int, instrument: str) -> None:
        self._data["last_session"] = {"site": site, "day": day, "hour": hour, "instrument": instrument}
        self.save()
