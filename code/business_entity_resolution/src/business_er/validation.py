"""Order-independent, disk-backed scoring and complete output validation."""

from collections import Counter
from itertools import tee
from pathlib import Path
import sqlite3

from .io import iter_id_lists, iter_records, parse_id_list
from .metrics import score_aligned_rows
from .storage import temporary_database


def _insert_lists(connection, table, path, column):
    """Load one unique S1-keyed ID-list table; table names are internal constants."""
    if table not in ("truth", "predictions"):
        raise ValueError("Unknown internal table")
    connection.execute(f"CREATE TABLE {table}(sid TEXT PRIMARY KEY, targets TEXT NOT NULL)")
    try:
        connection.executemany(
            f"INSERT INTO {table} VALUES (?,?)",
            ((sid, ",".join(ids)) for sid, ids in iter_id_lists(path, column)),
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Duplicate S1 row in {Path(path).name}") from exc


def score_files(truth_path, prediction_path, work_dir=None):
    """Score complete TSVs in any order; reject duplicate, missing or extra S1 IDs."""
    with temporary_database(work_dir) as connection:
        _insert_lists(connection, "truth", truth_path, "matched_entity_ids")
        _insert_lists(connection, "predictions", prediction_path, "matched_entity_ids")
        for left, right in (("truth", "predictions"), ("predictions", "truth")):
            missing = connection.execute(
                f"SELECT sid FROM {left} EXCEPT SELECT sid FROM {right} LIMIT 1"
            ).fetchone()
            if missing:
                raise ValueError(f"Unequal reference coverage: {missing[0]} missing from {right}")
        rows = connection.execute(
            "SELECT t.sid,t.targets,p.targets FROM truth t JOIN predictions p USING(sid) ORDER BY t.sid"
        )
        truth, prediction = tee(rows)
        result = score_aligned_rows(
            ((sid, parse_id_list(targets)) for sid, targets, _ in truth),
            ((sid, parse_id_list(targets)) for sid, _, targets in prediction),
        )
        result["reference_uniqueness_checked"] = True
        result["row_order_independent"] = True
        return result


def validate_files(source1, source2, source3, matching, candidates, work_dir=None):
    """Check both outputs against all sources using bounded cache and disk indexes.

    All three sources and both outputs are mandatory. Output ordering is arbitrary.
    Raises on duplicates, invalid IDs, missing/extra rows or matches not scored as
    candidates. A successful return always includes target-existence checks.
    """
    counts = Counter()
    countries = Counter()
    with temporary_database(work_dir) as connection:
        connection.execute("CREATE TABLE entities(id TEXT PRIMARY KEY, source TEXT NOT NULL)")
        connection.execute("CREATE TABLE outputs(sid TEXT PRIMARY KEY, matches TEXT, candidates TEXT)")
        for source, path in (("S1", source1), ("S2", source2), ("S3", source3)):
            try:
                for record in iter_records(path, source):
                    connection.execute("INSERT INTO entities VALUES (?,?)", (record.entity_id, source))
                    if source == "S1":
                        connection.execute("INSERT INTO outputs(sid) VALUES (?)", (record.entity_id,))
                        countries[record.country] += 1
                    counts[source] += 1
                    if counts[source] % 50000 == 0:
                        connection.commit()
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate source ID in {Path(path).name}") from exc
        if not counts["S1"]:
            raise ValueError("Reference source must not be empty")
        for column, path, header in (("matches", matching, "matched_entity_ids"),
                                      ("candidates", candidates, "candidate_entity_ids")):
            for sid, targets in iter_id_lists(path, header):
                cursor = connection.execute(
                    f"UPDATE outputs SET {column}=? WHERE sid=? AND {column} IS NULL",
                    (",".join(targets), sid),
                )
                if cursor.rowcount != 1:
                    raise ValueError(f"Duplicate or unknown reference {sid} in {Path(path).name}")
                # Chunk IN lists below conservative SQLite bound-parameter limits.
                for offset in range(0, len(targets), 400):
                    batch = targets[offset:offset + 400]
                    placeholders = ",".join("?" for _ in batch)
                    known = {row[0] for row in connection.execute(
                        f"SELECT id FROM entities WHERE source IN ('S2','S3') AND id IN ({placeholders})", batch
                    )}
                    missing = set(batch) - known
                    if missing:
                        raise ValueError(f"Unknown target ID {min(missing)!r} in {Path(path).name}")
                counts[column + "_rows"] += 1
                counts[column + "_links"] += len(targets)
                if column == "matches" and not targets:
                    counts["empty_predictions"] += 1
                if counts[column + "_rows"] % 50000 == 0:
                    connection.commit()
            connection.commit()
            missing = connection.execute(f"SELECT sid FROM outputs WHERE {column} IS NULL LIMIT 1").fetchone()
            if missing:
                raise ValueError(f"Missing reference {missing[0]} in {Path(path).name}")
        for sid, raw_matches, raw_candidates in connection.execute("SELECT * FROM outputs"):
            if not set(parse_id_list(raw_matches)).issubset(parse_id_list(raw_candidates)):
                raise ValueError(f"Matches outside candidate set for {sid}")
    return {"references": counts["S1"], "target_records": counts["S2"] + counts["S3"],
            "matching_links": counts["matches_links"], "candidate_links": counts["candidates_links"],
            "empty_predictions": counts["empty_predictions"], "references_by_country": dict(countries),
            "target_ids_checked": True, "reference_uniqueness_checked": True,
            "candidate_containment_checked": True, "row_order_independent": True}
