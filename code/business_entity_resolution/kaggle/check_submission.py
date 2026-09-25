"""Byte-level + rule-level check of a leaderboard file before uploading it (stricter than needed on purpose).
Memory-light: format checks stream line by line; ID/ownership/subset checks run in DuckDB with a hard cap.

Usage: python check_submission.py MATCHING_TSV TEST_DIR [--candidate CANDIDATE_TSV] [--mem 1.5GB]
TEST_DIR holds test_source1/2/3.tsv. Exit code 0 only if every check passes.
"""
import argparse
import sys
from pathlib import Path

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("matching"); ap.add_argument("test_dir"); ap.add_argument("--candidate"); ap.add_argument("--mem", default="1.5GB")
a = ap.parse_args()
problems = []


def fail(msg):
    """Record and print one failed check."""
    problems.append(msg)
    print("  FAIL:", msg)


def fmt(path, header):
    """Stream the file: encoding, BOM, CR, quotes, header, 2 fields, whitespace, placeholders, dup IDs in a list."""
    print(f"\n== {path}")
    with open(path, "rb") as f:
        head = f.read(3)
        if head == b"\xef\xbb\xbf":
            fail("starts with a UTF-8 BOM (invisible bytes before the header)")
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            fail("does not end with a newline")
    n = empty = 0
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            if b"\r" in raw or b'"' in raw or b"'" in raw:
                fail(f"line {i}: CR or quote character: {raw[:80]!r}")
                break
            try:
                line = raw.decode("utf-8").rstrip("\n")
            except UnicodeDecodeError as e:
                fail(f"line {i}: not UTF-8 ({e})")
                break
            if i == 1:
                if line != header:
                    fail(f"header is {line!r}, expected {header!r}")
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                fail(f"line {i}: {len(parts)} tab-separated fields (need exactly 2): {line[:80]!r}")
                break
            sid, lst = parts
            if sid != sid.strip() or " " in lst or not sid.startswith("S1-"):
                fail(f"line {i}: stray whitespace or bad S1 id: {line[:80]!r}")
                break
            if lst.lower() in ("nan", "none", "null", "[]", "na"):
                fail(f"line {i}: placeholder text instead of an empty list: {lst!r}")
                break
            ids = lst.split(",") if lst else []
            if "" in ids or len(ids) != len(set(ids)):
                fail(f"line {i}: empty or duplicate ID inside the list: {lst[:80]!r}")
                break
            n += 1
            empty += not ids
    print(f"  {n:,} rows ({empty:,} empty lists) read OK")


fmt(a.matching, "source1_entity_id\tmatched_entity_ids")
if a.candidate:
    fmt(a.candidate, "source1_entity_id\tcandidate_entity_ids")

con = duckdb.connect()
con.execute(f"SET memory_limit='{a.mem}'; SET threads=2; SET preserve_insertion_order=false")
rd = lambda p: f"read_csv('{Path(p).as_posix()}', delim='\t', header=true, quote='', escape='', all_varchar=true)"  # noqa: E731
td = Path(a.test_dir)
con.execute(f"CREATE TABLE s1 AS SELECT entity_id FROM {rd(td / 'test_source1.tsv')}")
con.execute(f"CREATE TABLE tg AS SELECT entity_id FROM {rd(td / 'test_source2.tsv')} UNION ALL SELECT entity_id FROM {rd(td / 'test_source3.tsv')}")
con.execute(f"CREATE TABLE m AS SELECT source1_entity_id AS s, coalesce(matched_entity_ids, '') AS l FROM {rd(a.matching)}")
q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
checks = {
    "duplicate S1 rows": "SELECT count(*) - count(DISTINCT s) FROM m",
    "test S1 entities missing a row": "SELECT count(*) FROM s1 ANTI JOIN m ON m.s = s1.entity_id",
    "rows that are not test S1 entities": "SELECT count(*) FROM m ANTI JOIN s1 ON m.s = s1.entity_id",
}
for name, sql in checks.items():
    v = q(sql)
    print(f"  {name}: {v:,}")
    if v:
        fail(f"{v:,} {name}")
con.execute("CREATE TABLE ml AS SELECT s, unnest(string_split(l, ',')) AS t FROM m WHERE l <> ''")
for name, sql in {"matched IDs not in test S2/S3": "SELECT count(*) FROM ml ANTI JOIN tg ON tg.entity_id = ml.t",
                  "targets matched to 2+ S1s": "SELECT count(*) - count(DISTINCT t) FROM ml"}.items():
    v = q(sql)
    print(f"  {name}: {v:,}")
    if v:
        fail(f"{v:,} {name}")
if a.candidate:
    con.execute(f"CREATE TABLE cl AS SELECT source1_entity_id AS s, unnest(string_split(candidate_entity_ids, ',')) AS t "
                f"FROM {rd(a.candidate)} WHERE coalesce(candidate_entity_ids, '') <> ''")
    v = q("SELECT count(*) FROM ml ANTI JOIN cl USING (s, t)")
    print(f"  matched pairs missing from candidate_pairs: {v:,}")
    if v:
        fail(f"{v:,} matched pairs are not in candidate_pairs (must be a subset)")
print(f"  links {q('SELECT count(*) FROM ml'):,}")
print("\nALL CHECKS PASSED" if not problems else f"\n{len(problems)} PROBLEM(S) FOUND")
sys.exit(1 if problems else 0)
