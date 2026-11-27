"""Fetching the Zenodo archives and streaming their members out of them."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

import py7zr
from py7zr.io import Py7zIO, WriterFactory
from tqdm.auto import tqdm

from oilslick.sources import ZENODO_API, Archive

CHUNK = 8 * 1024 * 1024


def zenodo_file_info(archive: Archive) -> tuple[str, int, str]:
    """Return (download url, size in bytes, md5) for an archive."""
    with urllib.request.urlopen(ZENODO_API.format(record=archive.record)) as r:
        record = json.load(r)
    for f in record["files"]:
        if f["key"] == archive.filename:
            return f["links"]["self"], f["size"], f["checksum"].removeprefix("md5:")
    raise FileNotFoundError(f"{archive.filename} not in Zenodo record {archive.record}")


def ensure_downloaded(archive: Archive, directory: Path) -> Path:
    """Download ``archive`` into ``directory`` unless it is already there.

    Downloads resume from a ``.part`` file and are MD5-verified before being
    renamed to their final name, so an existing final file is always complete.
    """
    dest = directory / archive.filename
    if dest.exists():
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    url, size, md5 = zenodo_file_info(archive)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    free = shutil.disk_usage(directory).free
    if free < size - start:
        raise OSError(
            f"{archive.filename} needs {(size - start) / 1e9:.1f} GB in {directory}, "
            f"only {free / 1e9:.1f} GB free"
        )
    if start < size:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
        with urllib.request.urlopen(request) as r:
            if start and r.status != 206:  # server ignored Range: restart from scratch
                start = 0
            with (
                part.open("ab" if start else "wb") as f,
                tqdm(total=size, initial=start, unit="B", unit_scale=True, desc=dest.name) as bar,
            ):
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
    if _md5(part) != md5:
        part.unlink()
        raise RuntimeError(f"MD5 mismatch for {archive.filename}; deleted, please rerun")
    part.rename(dest)
    return dest


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


class _CallbackIO(Py7zIO):
    """In-memory buffer that hands its content to a callback once the member is complete."""

    def __init__(self, name: str, on_complete: Callable[[str, io.BytesIO], None]) -> None:
        self.name, self.on_complete = name, on_complete
        self.buffer = io.BytesIO()

    def write(self, s: bytes | bytearray) -> int:
        return self.buffer.write(s)

    def read(self, size: int | None = None) -> bytes:
        return self.buffer.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.buffer.seek(offset, whence)

    def flush(self) -> None:
        pass

    def size(self) -> int:
        return self.buffer.getbuffer().nbytes

    def close(self) -> None:
        self.buffer.seek(0)
        self.on_complete(self.name, self.buffer)
        self.buffer = io.BytesIO()  # free the member before the next one is decompressed


class _CallbackFactory(WriterFactory):
    def __init__(self, on_complete: Callable[[str, io.BytesIO], None]) -> None:
        self.on_complete = on_complete

    def create(self, filename: str) -> Py7zIO:
        return _CallbackIO(filename, self.on_complete)


def stream_members(
    path: Path, handle: Callable[[str, io.BytesIO], None], suffix: str = ".tif"
) -> None:
    """Decompress ``path`` once, calling ``handle(name, data)`` for every member
    ending in ``suffix``.

    py7zr decompresses independent 7z folders in parallel threads, so ``handle``
    may run concurrently and up to one member per thread is held in memory."""
    with py7zr.SevenZipFile(path) as z:
        names = [f.filename for f in z.list() if not f.is_directory]
        targets = [n for n in names if n.lower().endswith(suffix)]
        with tqdm(total=len(targets), desc=path.name, unit="file") as bar:

            def on_complete(name: str, data: io.BytesIO) -> None:
                handle(name, data)
                bar.update()

            z.extract(targets=targets, factory=_CallbackFactory(on_complete))
