"""Query and download data from the Cloudnet data portal API.

https://docs.cloudnet.fmi.fi/api/data-portal.html
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import requests

API_BASE = "https://cloudnet.fmi.fi/api"
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "PRISM_CACHE", Path.home() / ".cache" / "prism"
))


@dataclass(frozen=True)
class RemoteFile:
    uuid: str
    filename: str
    size: int
    checksum: str
    download_url: str
    instrument_id: str | None
    kind: str  # "raw" or "product"
    product_id: str | None = None


def _get(path: str, **params) -> list[dict]:
    resp = requests.get(f"{API_BASE}/{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_sites() -> list[tuple[str, str]]:
    """List (id, human_readable_name) for regular Cloudnet observation sites,
    for populating a site-selector dropdown instead of free-text entry.

    Note: a site's "status" field (e.g. "inactive") reflects whether it is
    currently live-processing, not whether historical data exists -- Hyytiälä
    is flagged inactive yet has full archived data, so we don't filter on it.
    """
    data = _get("sites")
    sites = [(s["id"], s["humanReadableName"]) for s in data if "cloudnet" in s.get("type", [])]
    return sorted(sites, key=lambda s: s[1])


RAW_SPECTRA_INSTRUMENTS = ["rpg-fmcw-94", "mira-10", "mira-35"]


def _is_raw_spectra_file(instrument: str, filename: str) -> bool:
    """Each instrument's raw archive mixes spectra files with other raw data
    under the same `instrument` filter, so filename-matching is still needed
    per instrument:
    - rpg-fmcw-94: full Doppler spectra are ".LV0"; ".LV1" (also present at
      some sites) is a different, coarser moments-only format we don't parse.
    - mira-10: every ".znc.gz" file observed is a real per-hour spectra file.
    - mira-35: ".znc.gz" files exist too, but at every site checked so far
      they were all PPI wind-scan files ("...windppi...", not a vertical
      stare), which this app's height-profile UI can't use -- excluded here
      so the hour picker doesn't offer files with no usable vertical data.
    """
    upper = filename.upper()
    if instrument == "rpg-fmcw-94":
        return upper.endswith(".LV0")
    if instrument == "mira-10":
        return upper.endswith(".ZNC.GZ")
    if instrument == "mira-35":
        return upper.endswith(".ZNC.GZ") and "WINDPPI" not in upper
    return False


# Raw spectra filenames embed a YYYYMMDDHHMMSS (or YYMMDD_HHMMSS-style)
# timestamp, but neither its exact digit width nor whether an underscore
# separates the date from the time is consistent across sites/instruments:
#   260211_000000_P09_ZEN.LV0            (Hyytiälä/Bucharest RPG-FMCW-94)
#   20260916_000004.znc.gz               (Munich MIRA-10)
#   joyrad94_20240110000001_P01_ZEN.lv0  (Ny-Ålesund RPG, date+time RUN
#                                          TOGETHER with no separator)
#   mirac-a_20210120000001_P01_ZEN.lv0   (Jülich RPG, same)
# A previous version assumed the hour was always the first two characters
# of the SECOND underscore-separated token (filename.split("_")[1][0:2]),
# which happened to work for the first two formats above but silently broke
# for the other two: split("_")[1] there is the full run-together
# "20240110000001"/"20210120000001", so [0:2] read "20" -- the START OF THE
# YEAR, not the hour -- meaning every file that day was misfiled under
# "20 UTC" and the other 23-24 hours looked entirely missing. Matching the
# embedded timestamp with a regex instead of guessing at token positions
# handles both layouts, and isn't tied to a particular instrument having a
# name prefix or not.
_RUN_TOGETHER_TIMESTAMP = re.compile(r"(?<!\d)(\d{8})(\d{6})(?!\d)")  # YYYYMMDDHHMMSS, no separator
_SEPARATED_TIMESTAMP = re.compile(r"(?<!\d)(\d{6}|\d{8})_(\d{6})(?!\d)")  # (Y)YMMDD_HHMMSS


def hour_of_filename(filename: str) -> int:
    """Extract the UTC hour a raw spectra filename's embedded timestamp
    refers to. Raises ValueError if no recognized timestamp pattern is
    found, rather than silently returning a wrong hour."""
    for pattern in (_RUN_TOGETHER_TIMESTAMP, _SEPARATED_TIMESTAMP):
        m = pattern.search(filename)
        if m:
            hhmmss = m.group(2)
            hour = int(hhmmss[0:2])
            if 0 <= hour <= 23:
                return hour
    raise ValueError(f"could not find an embedded HHMMSS timestamp in filename: {filename!r}")


def list_available_instruments(site: str, day: date) -> list[str]:
    """Which of RAW_SPECTRA_INSTRUMENTS actually published usable raw
    spectra for this site/day -- so the instrument picker can offer only
    what's real instead of a fixed list that's mostly "(no data)" at most
    sites. Cheap: metadata-only listing calls, no file downloads."""
    return [i for i in RAW_SPECTRA_INSTRUMENTS if list_raw_spectra_files(site, day, i)]


def list_raw_spectra_files(site: str, day: date, instrument: str = "rpg-fmcw-94") -> list[RemoteFile]:
    """List raw Doppler-spectra files available for one day."""
    day_str = day.isoformat()
    data = _get(
        "raw-files",
        site=site,
        dateFrom=day_str,
        dateTo=day_str,
        instrument=instrument,
    )
    files = [
        RemoteFile(
            uuid=f["uuid"],
            filename=f["filename"],
            size=int(f["size"]),
            checksum=f["checksum"],
            download_url=f["downloadUrl"],
            instrument_id=f.get("instrument", {}).get("instrumentId"),
            kind="raw",
        )
        for f in data
        if _is_raw_spectra_file(instrument, f["filename"])
    ]
    return sorted(files, key=lambda f: f.filename)


def list_product_files(site: str, day: date) -> list[RemoteFile]:
    """List all processed Cloudnet product files available for one day."""
    day_str = day.isoformat()
    data = _get("files", site=site, dateFrom=day_str, dateTo=day_str)
    return [
        RemoteFile(
            uuid=f["uuid"],
            filename=f["filename"],
            size=int(f["size"]),
            checksum=f["checksum"],
            download_url=f["downloadUrl"],
            instrument_id=(f.get("instrument") or {}).get("instrumentId"),
            kind="product",
            product_id=f["product"]["id"],
        )
        for f in data
    ]


def list_model_files(site: str, day: date) -> list[RemoteFile]:
    """List the numerical-weather-model files (e.g. ECMWF) Cloudnet holds for
    one site and day. These are NOT returned by the regular "files"
    endpoint, hence the separate call; the model's id (e.g. "ecmwf") stands
    in for the instrument."""
    day_str = day.isoformat()
    data = _get("model-files", site=site, dateFrom=day_str, dateTo=day_str)
    return [
        RemoteFile(
            uuid=f["uuid"],
            filename=f["filename"],
            size=int(f["size"]),
            checksum=f["checksum"],
            download_url=f["downloadUrl"],
            instrument_id=f["model"]["id"],
            kind="product",
            product_id="model",
        )
        for f in data
    ]


def cache_path_for(remote: RemoteFile, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    subdir = "raw" if remote.kind == "raw" else "products"
    return cache_dir / subdir / remote.checksum[:2] / f"{remote.checksum}_{remote.filename}"


_path_locks: dict[str, list] = {}  # path -> [RLock, users]
_path_locks_guard = threading.Lock()


@contextmanager
def path_lock(path: Path):
    """Serialize work on one cache path across threads (i.e. across user
    sessions in server mode, which share one cache directory): two sessions
    asking for the same file must not both write its ".part" file or decode
    into the same zarr store. Re-entrant, so a caller can hold the lock
    across a download + decode whose inner steps lock the same path again.
    Entries are dropped once nobody holds or waits on them."""
    key = str(path)
    with _path_locks_guard:
        entry = _path_locks.setdefault(key, [threading.RLock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _path_locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                del _path_locks[key]


def ensure_downloaded(
    remote: RemoteFile,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_retries: int = 5,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Download a file into the local cache unless already present and valid.

    Resumes from a partial ".part" file via HTTP Range requests and retries
    on a dropped connection, instead of restarting from byte 0 -- important
    on a flaky connection where a multi-hundred-MB file may not complete in
    one shot.

    on_progress(filename, downloaded_bytes, total_bytes), if given, is called
    after every chunk -- e.g. to drive a status line -- and once more with
    downloaded_bytes == total_bytes when a cache hit skips the download.

    Thread-safe: concurrent callers for the same file queue on a lock, and
    the ones that waited find it already downloaded.
    """
    with path_lock(cache_path_for(remote, cache_dir)):
        return _download_locked(remote, cache_dir, max_retries, on_progress)


def _download_locked(remote, cache_dir, max_retries, on_progress) -> Path:
    dest = cache_path_for(remote, cache_dir)
    if dest.exists() and dest.stat().st_size == remote.size:
        if on_progress:
            on_progress(remote.filename, remote.size, remote.size)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_error: Exception | None = None
    for _attempt in range(max_retries):
        existing = tmp.stat().st_size if tmp.exists() else 0
        if existing >= remote.size:
            break
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(remote.download_url, stream=True, timeout=60, headers=headers) as resp:
                if existing and resp.status_code == 200:
                    # server ignored our Range request -- start over
                    existing = 0
                resp.raise_for_status()
                mode = "ab" if existing else "wb"
                with open(tmp, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        existing += len(chunk)
                        if on_progress:
                            on_progress(remote.filename, existing, remote.size)
            last_error = None
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise ConnectionError(
            f"Could not fully download {remote.filename} after {max_retries} attempts "
            f"({tmp.stat().st_size if tmp.exists() else 0}/{remote.size} bytes). "
            "The partial download was kept and will resume next time."
        ) from last_error
    if not tmp.exists() or tmp.stat().st_size != remote.size:
        raise ConnectionError(
            f"Downloaded size mismatch for {remote.filename}: "
            f"{tmp.stat().st_size if tmp.exists() else 0} != {remote.size}"
        )
    tmp.rename(dest)
    return dest


def cache_size_bytes(cache_dir: Path = DEFAULT_CACHE_DIR) -> int:
    """Total size of downloaded/decoded data on disk (raw LV0 + decoded zarr
    + product netCDFs), for a "cache: N GB" status display."""
    total = 0
    for sub in ("raw", "products", "spectra_zarr"):
        root = cache_dir / sub
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    return total


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def clear_data_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Delete downloaded/decoded data (raw LV0, decoded zarr, product netCDFs)
    but leave user settings (stored separately) untouched."""
    import shutil
    for sub in ("raw", "products", "spectra_zarr"):
        shutil.rmtree(cache_dir / sub, ignore_errors=True)


def verify_checksum(path: Path, expected: str) -> bool:
    """Raw files use md5 (32 hex chars), product files use sha256 (64 hex chars)."""
    algo = hashlib.md5() if len(expected) == 32 else hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            algo.update(chunk)
    return algo.hexdigest() == expected
