"""Persist user-adjustable UI state (selected variables, color/axis limits,
co/cross toggle) across sessions, with per-panel and global reset.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

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
    "last_session": {"site": "hyytiala", "day": "2026-02-11", "hour": 8, "instrument": "rpg-fmcw-94"},
}


class Settings:
    def __init__(self, path: Path = DEFAULT_SETTINGS_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text())
                merged = copy.deepcopy(DEFAULTS)
                merged.update(stored)
                return merged
            except (json.JSONDecodeError, OSError):
                pass
        return copy.deepcopy(DEFAULTS)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

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
        self.save()

    def set_last_session(self, site: str, day: str, hour: int, instrument: str) -> None:
        self._data["last_session"] = {"site": site, "day": day, "hour": hour, "instrument": instrument}
        self.save()
