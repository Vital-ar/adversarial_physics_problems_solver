"""Stage 1.2: download textbooks / solution manuals listed in ``configs/data/book_ingestion.yaml``.

Each book entry may define:

* ``url``  - direct download link to the file (e.g. ``https://archive.org/download/lm_20220102/lm.pdf``)
* ``path`` - local destination. If blank, defaults to ``<download_dir>/<book_id>.<format>``.

Only the Python standard library is used (``urllib``). Downloads are skipped when the local
file already exists (unless ``force=True``). Downloaded content is validated by magic bytes so
an HTML error page (e.g. archive.org "500 Internal Server Error") is never saved as a PDF.
"""
from __future__ import annotations

import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger

logger = get_logger("stage_1_2_book_downloader")

DEFAULT_DOWNLOAD_DIR = "data/books"
_USER_AGENT = "adversarial-physics-solver/0.1 (+book ingestion; research use)"

# Magic-byte prefixes used to validate downloaded files.
_MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF",),
    "docx": (b"PK\x03\x04",),
}


@dataclass(frozen=True)
class DownloadResult:
    book_id: str
    url: str
    path: str
    status: str  # "downloaded" | "exists" | "skipped_no_url" | "disabled" | "failed"
    bytes: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_book_path(entry: dict[str, Any], *, download_dir: str | Path = DEFAULT_DOWNLOAD_DIR) -> Path | None:
    """Return the absolute local path for a book entry (explicit ``path`` or default from ``book_id``).

    Returns None when neither ``path`` nor ``url`` is set (an unconfigured placeholder entry).
    """
    raw_path = str(entry.get("path") or "").strip()
    if raw_path:
        return resolve_project_path(raw_path)
    raw_url = str(entry.get("url") or "").strip()
    if not raw_url:
        return None
    book_id = str(entry.get("book_id") or "").strip() or Path(raw_url).stem
    fmt = str(entry.get("format") or Path(raw_url).suffix.lstrip(".") or "pdf").lower()
    return resolve_project_path(Path(download_dir) / f"{book_id}.{fmt}")


def _validate_magic(path: Path, fmt: str) -> None:
    prefixes = _MAGIC_BYTES.get(fmt)
    if not prefixes:
        return
    with path.open("rb") as file:
        head = file.read(8)
    if not any(head.startswith(p) for p in prefixes):
        raise ValueError(
            f"Downloaded file does not look like a .{fmt} (first bytes: {head!r}); "
            "the server probably returned an HTML error page."
        )


def download_file(url: str, destination: str | Path, *, file_format: str = "pdf", timeout: float = 120.0) -> int:
    """Stream ``url`` to ``destination`` atomically (via a ``.part`` file). Returns bytes written."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        _validate_magic(tmp, file_format.lower())
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest.stat().st_size


def download_books(
    sources: list[dict[str, Any]],
    *,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    book_ids: list[str] | None = None,
    force: bool = False,
    include_disabled: bool = False,
    timeout: float = 120.0,
) -> list[DownloadResult]:
    """Download every configured book that has a ``url`` and is not yet on disk."""
    wanted = set(book_ids or [])
    results: list[DownloadResult] = []
    for entry in sources:
        book_id = str(entry.get("book_id") or "")
        if wanted and book_id not in wanted:
            continue
        url = str(entry.get("url") or "").strip()
        dest = resolve_book_path(entry, download_dir=download_dir)
        if not entry.get("enabled", True) and not include_disabled and not wanted:
            results.append(DownloadResult(book_id, url, str(dest or ""), "disabled"))
            continue
        if not url or dest is None:
            results.append(DownloadResult(book_id, url, str(dest or ""), "skipped_no_url"))
            continue
        if dest.is_file() and not force:
            results.append(DownloadResult(book_id, url, str(dest), "exists", dest.stat().st_size))
            continue
        fmt = str(entry.get("format") or dest.suffix.lstrip(".") or "pdf")
        logger.info("Downloading %s from %s -> %s", book_id, url, dest)
        try:
            size = download_file(url, dest, file_format=fmt, timeout=timeout)
            results.append(DownloadResult(book_id, url, str(dest), "downloaded", size))
            logger.info("Downloaded %s (%.1f MB)", book_id, size / 1e6)
        except Exception as exc:  # noqa: BLE001 - report and continue with other books
            logger.error("Failed to download %s: %s", book_id, exc)
            results.append(DownloadResult(book_id, url, str(dest), "failed", error=str(exc)))
    return results
