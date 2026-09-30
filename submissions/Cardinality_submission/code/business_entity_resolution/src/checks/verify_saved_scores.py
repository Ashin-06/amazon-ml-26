"""Fail if a compact-package neural score cache does not match its regenerated pair band."""
import sys
from pathlib import Path

import duckdb


def q(path):
    """Quote an absolute path for DuckDB's read_parquet SQL call."""
    return "'" + str(Path(path).resolve()).replace("\\", "/").replace("'", "''") + "'"


def main(expected_band, saved_scores):
    """Require the score cache to have exactly one finite score for each band pair."""
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='1GB'; SET preserve_insertion_order=false")
    con.execute("SET temp_directory='work/saved_score_check_tmp'")
    a, b = q(expected_band), q(saved_scores)
    con.execute(f"CREATE VIEW expected AS SELECT sid,tid FROM read_parquet({a})")
    con.execute(f"CREATE VIEW saved AS SELECT sid,tid,xe FROM read_parquet({b})")
    n1 = con.execute("SELECT count(*) FROM expected").fetchone()[0]
    n2, unique, bad = con.execute("""SELECT count(*), count(DISTINCT (sid,tid)),
        count(*) FILTER (WHERE sid IS NULL OR tid IS NULL OR xe IS NULL OR NOT isfinite(xe))
        FROM saved""").fetchone()
    extra = con.execute("SELECT count(*) FROM (SELECT sid,tid FROM saved EXCEPT SELECT sid,tid FROM expected)").fetchone()[0]
    missing = con.execute("SELECT count(*) FROM (SELECT sid,tid FROM expected EXCEPT SELECT sid,tid FROM saved)").fetchone()[0]
    con.close()
    if n1 != n2 or unique != n2 or bad or extra or missing:
        raise ValueError(f'Saved score pair mismatch: expected={n1}, saved={n2}, unique={unique}, invalid={bad}, extra={extra}, missing={missing}')
    print(f'Saved scores match regenerated band: {n1:,} pairs', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
