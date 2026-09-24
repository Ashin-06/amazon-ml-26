"""Atomic TSV writing and strict aligned checks for foundational/pilot workflows."""

from itertools import zip_longest
from pathlib import Path
import os
import tempfile

from .io import LIST_COLUMNS, parse_id_list, validate_id


def write_id_lists(path, rows, column="matched_entity_ids") -> int:
    """Write validated rows atomically; preserve order and empty list fields.

    Exhausts rows with bounded row memory. File-wide S1 uniqueness must be checked
    separately. A failed write leaves an existing destination unchanged.
    """
    if column not in LIST_COLUMNS:
        raise ValueError(f"Unsupported ID-list column: {column}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(f"source1_entity_id\t{column}\n")
            for sid, targets in rows:
                validate_id(sid, ("S1",))
                targets = tuple(targets)
                # Validate individual elements before joining so embedded commas
                # cannot silently turn one malformed ID into multiple IDs.
                for target in targets:
                    validate_id(target, ("S2", "S3"))
                raw = ",".join(targets)
                parse_id_list(raw)
                stream.write(f"{sid}\t{raw}\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return count


def validate_aligned_outputs(
    reference_rows, matching_rows, candidate_rows, valid_targets=None
) -> dict:
    """Check coverage, uniqueness, syntax and candidate containment in lockstep.

    Target existence is checked only when valid_targets is provided. Duplicate
    reference detection uses O(number of S1 IDs) memory. A target set adds further
    memory; this is not the later disk-backed full-scale validation implementation.
    """
    seen = set()
    sentinel = object()
    empty = match_links = candidate_links = 0
    for record, matching, candidate in zip_longest(
        reference_rows, matching_rows, candidate_rows, fillvalue=sentinel
    ):
        if any(value is sentinel for value in (record, matching, candidate)):
            raise ValueError("Reference, matching and candidate row counts differ")
        sid = record.entity_id
        validate_id(sid, ("S1",))
        if sid in seen:
            raise ValueError(f"Duplicate reference row: {sid}")
        seen.add(sid)
        if matching[0] != sid or candidate[0] != sid:
            raise ValueError(f"Missing or misaligned output row for {sid}")
        mids, cids = tuple(matching[1]), tuple(candidate[1])
        for targets in (mids, cids):
            for target in targets:
                validate_id(target, ("S2", "S3"))
            parse_id_list(",".join(targets))
        if not set(mids).issubset(cids):
            raise ValueError(f"Matches outside exported candidates for {sid}")
        if valid_targets is not None:
            for target in cids:
                if target not in valid_targets:
                    raise ValueError(f"Unknown target ID {target!r} for {sid}")
        empty += not mids
        match_links += len(mids)
        candidate_links += len(cids)
    if not seen:
        raise ValueError("Cannot validate an empty reference universe")
    return {
        "references": len(seen), "empty_predictions": empty,
        "matching_links": match_links, "candidate_links": candidate_links,
        "target_ids_checked": valid_targets is not None,
        "scope": "aligned structural and subset checks; target existence is optional",
    }
