"""Query and download data from the Cloudnet data portal API.

https://docs.cloudnet.fmi.fi/api/data-portal.html
"""
from __future__ import annotations

import hashlib
import os
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


def cache_path_for(remote: RemoteFile, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    subdir = "raw" if remote.kind == "raw" else "products"
    return cache_dir / subdir / remote.checksum[:2] / f"{remote.checksum}_{remote.filename}"


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
    """
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
