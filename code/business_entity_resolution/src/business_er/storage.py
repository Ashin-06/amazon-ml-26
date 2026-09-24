"""Bounded SQLite working storage and explicit-file integrity helpers."""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import tempfile


def file_digest(path):
    """SHA-256 one explicit file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace a JSON file; reject NaN/Infinity and preserve UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            name = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and name.exists():
            name.unlink()


def configure_database(connection):
    """Bound SQLite's page cache to about 32 MiB; put sorting work on disk."""
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")


@contextmanager
def temporary_database(work_dir=None):
    """Yield a temporary SQLite connection; clean only its own explicit files."""
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="business-er-", suffix=".sqlite", dir=work_dir)
    os.close(descriptor)
    connection = sqlite3.connect(name, uri=True)
    try:
        configure_database(connection)
        yield connection
    finally:
        connection.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            path = Path(name + suffix)
            if path.exists():
                path.unlink()


def open_readonly(path):
    """Open an existing local database without creating or updating it."""
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    configure_database(connection)
    return connection


def code_manifest():
    """Hash only the runnable package's Python sources, never workspace secrets."""
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)).replace("\\", "/"): file_digest(path)
            for path in sorted(root.rglob("*.py"))}


def staging_directory(destination):
    """Create a new unpublished sibling directory; never overwrite an existing run.

    Failed stages are retained for diagnosis. Only a complete stage is renamed
    to the requested destination by its caller.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination exists; choose a new run: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
