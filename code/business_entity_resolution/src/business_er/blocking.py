"""Offline name/address retrieval. Word-token baseline; no external lookups."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import tempfile
import time

from .io import Record, iter_records
from .normalize import fold_latin_accents, normalize_text
from .storage import configure_database, file_digest, open_readonly

INDEX_VERSION = 1


def _partition(source, country):
    """Represent any normalized country as an ASCII FTS token, including blank."""
    return source.lower() + "c" + hashlib.sha256(normalize_text(country).encode("utf-8")).hexdigest()


def _quote(token):
    """Quote a literal FTS phrase; input text must not become a query operator."""
    return '"' + token.replace('"', '""') + '"'


def build_index(source1, source2, source3, destination):
    """Stream all sources into an atomic, immutable SQLite/FTS5 baseline index.

    Original text, IDs and arbitrary countries are preserved. Duplicate source
    IDs fail construction. Only targets enter FTS. The finished index includes
    complete source fingerprints; a partial database is never published.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Index exists; use a new path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".building-", dir=destination.parent)
    os.close(descriptor)
    connection = sqlite3.connect(name)
    counts = Counter({"S1": 0, "S2": 0, "S3": 0})
    countries = {}
    inputs = {}
    started = time.monotonic()
    try:
        configure_database(connection)
        connection.executescript("""
            CREATE TABLE records (
                rowid INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL, name TEXT NOT NULL, address TEXT NOT NULL, country TEXT NOT NULL);
            CREATE INDEX records_source ON records(source);
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE search USING fts5(name,address,partition,content='',
                tokenize="unicode61 remove_diacritics 0 categories 'L* N* M*'");
        """)
        for source, path in (("S1", source1), ("S2", source2), ("S3", source3)):
            path = Path(path).resolve(strict=True)
            before = file_digest(path)
            by_country = Counter()
            for record in iter_records(path, source):
                try:
                    cursor = connection.execute(
                        "INSERT INTO records(entity_id,source,name,address,country) VALUES (?,?,?,?,?)",
                        (record.entity_id, source, record.business_name, record.business_address, record.country),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate source ID: {record.entity_id}") from exc
                if source != "S1":
                    connection.execute("INSERT INTO search(rowid,name,address,partition) VALUES (?,?,?,?)", (
                        cursor.lastrowid, fold_latin_accents(record.business_name),
                        fold_latin_accents(record.business_address),
                        source.lower() + " " + _partition(source, record.country),
                    ))
                counts[source] += 1
                by_country[record.country] += 1
                if counts[source] % 50000 == 0:
                    connection.commit()
            connection.commit()
            if file_digest(path) != before:
                raise ValueError(f"Source changed during indexing: {path.name}")
            inputs[source] = {"path": str(path), "sha256": before, "bytes": path.stat().st_size}
            countries[source] = dict(by_country)
        if not counts["S1"] or not counts["S2"] + counts["S3"]:
            raise ValueError("Index requires nonempty references and at least one target source")
        connection.execute("INSERT INTO search(search) VALUES ('optimize')")
        connection.execute("INSERT INTO search(search) VALUES ('integrity-check')")
        index_code = {name: file_digest(Path(__file__).with_name(name))
                      for name in ("blocking.py", "normalize.py", "io.py")}
        manifest = {"index_version": INDEX_VERSION, "counts": dict(counts), "countries": countries,
                    "index_code_sha256": index_code,
                    "inputs": inputs, "elapsed_seconds": time.monotonic() - started,
                    "backend": "sqlite_fts5_word_name_address", "sqlite_version": sqlite3.sqlite_version}
        connection.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(manifest),))
        connection.commit()
        connection.close()
        connection = None
        if destination.exists():
            raise FileExistsError(f"Index destination was created by another process: {destination}")
        os.rename(name, destination)
        return manifest
    finally:
        if connection is not None:
            connection.close()
        for suffix in ("", "-journal"):
            temporary = Path(name + suffix)
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True)
class Candidate:
    """One unique target record with independent route ranks; zero means absent."""

    record: Record
    name_rank: int = 0
    address_rank: int = 0


class CandidateIndex:
    """Read-only index with bounded per-query results and explicit lifecycle."""

    def __init__(self, path):
        """Open an existing completed index and reject unsupported formats."""
        self.path = Path(path).resolve(strict=True)
        self.connection = open_readonly(self.path)
        try:
            row = self.connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if row is None:
                raise ValueError("Index has no completion manifest")
            self.manifest = json.loads(row[0])
            if self.manifest.get("index_version") != INDEX_VERSION:
                raise ValueError("Unsupported index version")
            current_code = {name: file_digest(Path(__file__).with_name(name))
                            for name in ("blocking.py", "normalize.py", "io.py")}
            if self.manifest.get("index_code_sha256") != current_code:
                raise ValueError("Indexing code changed; rebuild this index")
        except Exception:
            self.connection.close()
            raise

    def close(self):
        """Close the read-only database; callers should use a context manager."""
        self.connection.close()

    def __enter__(self):
        """Return the opened index."""
        return self

    def __exit__(self, *args):
        """Always release file handles, including after a failed experiment."""
        self.close()

    def iter_references(self):
        """Stream S1 records in original source-file order."""
        for row in self.connection.execute(
            "SELECT entity_id,name,address,country FROM records WHERE source='S1' ORDER BY rowid"
        ):
            yield Record(*row)

    def get_record(self, entity_id):
        """Fetch one exact original record or fail on a missing ID."""
        row = self.connection.execute(
            "SELECT entity_id,name,address,country FROM records WHERE entity_id=?", (entity_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Record is absent from index: {entity_id}")
        return Record(*row)

    def retrieve(self, record, k=10, max_query_tokens=12):
        """Union independent name/address top-K routes for each source.

        No labels are consulted. Literal Unicode tokens are quoted and query
        values are parameterized. Known countries include blank-country targets;
        a blank reference country searches the entire corresponding source.
        """
        if type(k) is not int or type(max_query_tokens) is not int or k <= 0 or max_query_tokens <= 0:
            raise ValueError("Positive integer retrieval limits are required")
        found = {}
        for source in ("S2", "S3"):
            if normalize_text(record.country):
                partition = "partition:(" + _quote(_partition(source, record.country)) + " OR " + _quote(_partition(source, "")) + ")"
            else:
                partition = "partition:" + _quote(source.lower())
            for field, text in (("name", record.business_name), ("address", record.business_address)):
                tokens = sorted(set(fold_latin_accents(text).split()), key=lambda value: (-len(value), value))[:max_query_tokens]
                if not tokens:
                    continue
                query = partition + " AND " + field + ":(" + " OR ".join(_quote(token) for token in tokens) + ")"
                rows = self.connection.execute(
                    "SELECT rowid FROM search WHERE search MATCH ? "
                    "ORDER BY bm25(search,1.0,1.0,0.0),rowid LIMIT ?", (query, k)
                ).fetchall()
                for rank, (rowid,) in enumerate(rows, 1):
                    found.setdefault(rowid, {})[field + "_rank"] = rank
        result = []
        for rowid, ranks in found.items():
            row = self.connection.execute(
                "SELECT entity_id,name,address,country FROM records WHERE rowid=?", (rowid,)
            ).fetchone()
            result.append(Candidate(Record(*row), **ranks))
        return sorted(result, key=lambda candidate: candidate.record.entity_id)
