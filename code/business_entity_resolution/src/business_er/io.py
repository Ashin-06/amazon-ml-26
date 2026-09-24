"""Strict streaming readers. Keep identifiers and missing fields as strings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import csv
import re

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
LIST_COLUMNS = ("matched_entity_ids", "candidate_entity_ids")
ID_PATTERN = re.compile(r"S([123])-[0-9]+\Z")


@dataclass(frozen=True)
class Record:
    """One original source row, without destructive cleaning or numeric IDs."""

    entity_id: str
    business_name: str
    business_address: str
    country: str


def validate_id(value: str, sources: tuple[str, ...]) -> None:
    """Raise ValueError unless value is an exact ID in an allowed source."""
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid entity ID: {value!r}")
    if value[:2] not in sources:
        raise ValueError(f"Unexpected source for {value!r}; expected {sources}")


def _iter_rows(path: str | Path, header: tuple[str, ...]) -> Iterator[list[str]]:
    """Read exact-width UTF-8 TSV rows, raising a line-specific schema error."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        actual = next(reader, None)
        if actual != list(header):
            raise ValueError(f"{path}: expected header {header}, got {actual!r}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: line {reader.line_num}: expected {len(header)} fields, "
                    f"got {len(row)}"
                )
            yield row


def iter_records(
    path: str | Path, expected_source: str | None = None
) -> Iterator[Record]:
    """Stream records; validate schema/prefix, preserve raw text and all countries.

    Memory is bounded by a row. File-wide ID uniqueness belongs to the audit stage.
    expected_source, when provided, is 'S1', 'S2' or 'S3'.
    """
    if expected_source is not None and expected_source not in ("S1", "S2", "S3"):
        raise ValueError("expected_source must be S1, S2 or S3")
    allowed = (expected_source,) if expected_source else ("S1", "S2", "S3")
    for row in _iter_rows(path, SOURCE_COLUMNS):
        validate_id(row[0], allowed)
        yield Record(*row)


def parse_id_list(value: str) -> tuple[str, ...]:
    """Parse a possibly empty target list; reject duplicates and malformed IDs.

    This checks syntax, not membership in a particular train/test source pool.
    """
    if not isinstance(value, str):
        raise ValueError("ID-list field must be a string")
    if value == "":
        return ()
    values = tuple(value.split(","))
    for item in values:
        validate_id(item, ("S2", "S3"))
    if len(values) != len(set(values)):
        raise ValueError("Duplicate target ID inside a list")
    return values


def iter_id_lists(
    path: str | Path, column: str = "matched_entity_ids"
) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Stream exact two-column ID-list rows; file-wide uniqueness is not retained."""
    if column not in LIST_COLUMNS:
        raise ValueError(f"Unsupported ID-list column: {column}")
    for sid, raw in _iter_rows(path, ("source1_entity_id", column)):
        validate_id(sid, ("S1",))
        yield sid, parse_id_list(raw)
