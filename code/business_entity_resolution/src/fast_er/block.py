"""Stage 2: candidate generation (blocking) with DuckDB set joins.

Every record gets hashed keys, namespaced by country and field:
  name    : core-name tokens (len>=2), adjacent-token concatenations incl. short
            tokens ("n"+"infra" -> "ninfra", "paulita"+"wilson" -> "paulitawilson",
            catches @handles / domains), first-3 concat; whole core name (cap `cap_whole`).
  name x address word : each core-name token (len>=2) / whole-name concat joined with
            each non-numeric address word (any position) ("pioneerinfotech|pune"), so
            generic names still block within their city. Counted as name evidence.
  address : tokens (len>=3 or containing a digit) and adjacent token bigrams
            ("1550 beverly") that survive component reordering.
Keys held by more than `cap` targets are dropped (too common to discriminate).
Pair score = sum of idf over shared keys (total, name-only, address-only).
Name-rank ties: more address evidence first, then an empty address (no conflict) over a conflicting one.
Kept per S1: top-`k` by total score  UNION  top-`k_field` by name  UNION
top-`k_field` by address, so address-only (renamed DBA) and name-only
(missing address) matches both survive. Output: work/<split>/cands.parquet.
"""
import os
import time
from pathlib import Path

import duckdb

from .model import configure, lit, normalized_sql

KEYS_SQL = """
CREATE OR REPLACE TABLE keys AS
WITH t AS (
  SELECT rid, src, country, addr = '' AS te,
         list_filter(string_split(core, ' '), x -> x <> '') AS nt,
         list_filter(string_split(addr, ' '), x -> x <> '') AS atok
  FROM rec)
SELECT rid, src, te, 0::TINYINT AS f, hash(country || '|n|' || k) AS h FROM (
  SELECT rid, src, te, country, unnest(list_distinct(
      list_filter(nt, x -> length(x) >= 2)
         || list_transform(range(1, len(nt)), i -> nt[i] || nt[i + 1])
         || CASE WHEN len(nt) >= 3 THEN [nt[1] || nt[2] || nt[3]] ELSE [] END)) AS k
  FROM t)
UNION ALL
-- whole core name (f=2: name evidence with its own, higher document-frequency cap)
SELECT rid, src, te, 2::TINYINT, hash(country || '|w|' || array_to_string(nt, '')) FROM t WHERE len(nt) >= 1
UNION ALL
-- composite name x address-word keys: "pioneerinfotech|pune" is rare even when both parts are common
SELECT rid, src, te, 0::TINYINT, hash(country || '|x|' || k) FROM (
  SELECT rid, src, te, country, unnest(list_distinct(flatten(list_transform(
      list_distinct(list_filter(nt, x -> length(x) >= 2) || [array_to_string(nt, '')]),
      n -> list_transform(list_distinct(list_filter(atok, x -> length(x) >= 3 AND NOT regexp_matches(x, '^[0-9]+$'))),
                          a -> n || '|' || a))))) AS k
  FROM t)
UNION ALL
SELECT rid, src, te, 1::TINYINT, hash(country || '|a|' || k) FROM (
  SELECT rid, src, te, country, unnest(list_distinct(
      list_filter(atok, x -> length(x) >= 3 OR regexp_matches(x, '[0-9]'))
      || list_transform(range(1, len(atok)), i -> atok[i] || ' ' || atok[i + 1]))) AS k
  FROM t)
"""


def block(split, work="work", cap=300, k=25, k_field=10, chunks=8, threads=None, mem="20GB", cap_whole=1000):
    """Build fresh natural candidates; partial chunks are never treated as completed work."""
    for name, value in (("cap", cap), ("cap_whole", cap_whole), ("k", k), ("k_field", k_field), ("chunks", chunks)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    out = Path(work) / split / "cands.parquet"
    if out.exists():
        raise FileExistsError("Use the stage driver to verify completed blocking outputs")
    t0 = time.time()
    con = duckdb.connect()
    configure(con, mem, threads, Path(os.environ.get("FASTER_TMP") or work) / "duck_tmp")
    con.execute(f"CREATE TABLE rec AS SELECT rid, src, country, core, addr FROM read_parquet({normalized_sql(Path(work) / split)})")
    con.execute(KEYS_SQL)
    con.execute("DROP TABLE rec")
    n_t = con.execute("SELECT count(DISTINCT rid) FROM keys WHERE src > 1").fetchone()[0]
    # Target-side postings with document frequency; common keys dropped.
    con.execute(f"""
        CREATE TABLE post AS
        WITH d AS (SELECT h, count(*) AS df FROM keys WHERE src > 1 GROUP BY h, f
                   HAVING count(*) <= CASE WHEN f = 2 THEN {cap_whole} ELSE {cap} END)
        SELECT k.h, k.rid AS tid, k.f, k.te, ln(1 + {n_t} / d.df)::FLOAT AS w
        FROM keys k JOIN d USING (h) WHERE k.src > 1""")
    con.execute("CREATE TABLE q AS SELECT rid AS sid, h FROM keys WHERE src = 1")
    con.execute("DROP TABLE keys")
    print(f"[block {split}] keys ready {time.time() - t0:.0f}s  targets={n_t:,}", flush=True)
    tmp = Path(work) / split / "cands_parts"
    tmp.mkdir(parents=True, exist_ok=False)
    for c in range(chunks):
        part = tmp / f"{c}.parquet"
        con.execute(f"""
            COPY (
              WITH s AS (
                SELECT q.sid, p.tid, sum(p.w)::FLOAT AS s,
                       coalesce(sum(p.w) FILTER (WHERE p.f <> 1), 0)::FLOAT AS sn,
                       coalesce(sum(p.w) FILTER (WHERE p.f = 1), 0)::FLOAT AS sa,
                       count(*)::INTEGER AS nk, bool_or(p.te) AS te
                FROM q JOIN post p USING (h)
                WHERE q.sid % {chunks} = {c}
                GROUP BY q.sid, p.tid),
              r AS (
                SELECT *, row_number() OVER (PARTITION BY sid ORDER BY s DESC, tid) AS r_s,
                          row_number() OVER (PARTITION BY sid ORDER BY sn DESC, sa DESC, te DESC, tid) AS r_n,
                          row_number() OVER (PARTITION BY sid ORDER BY sa DESC, tid) AS r_a
                FROM s)
              SELECT sid, tid, s, sn, sa, nk, r_s::INTEGER AS r_s,
                     (CASE WHEN sn > 0 THEN r_n ELSE 9999 END)::INTEGER AS r_n,
                     (CASE WHEN sa > 0 THEN r_a ELSE 9999 END)::INTEGER AS r_a
              FROM r WHERE r_s <= {k} OR (r_n <= {k_field} AND sn > 0) OR (r_a <= {k_field} AND sa > 0)
            ) TO {lit(part)} (FORMAT parquet)""")
        print(f"[block {split}] chunk {c + 1}/{chunks} {time.time() - t0:.0f}s", flush=True)
    con.execute(f"COPY (SELECT * FROM read_parquet({'[' + ','.join(lit(tmp / f'{i}.parquet') for i in range(chunks)) + ']'})) TO {lit(out)} (FORMAT parquet)")
    n = con.execute(f"SELECT count(*), count(DISTINCT sid) FROM read_parquet({lit(out)})").fetchone()
    print(f"[block {split}] done {n[0]:,} pairs for {n[1]:,} S1  {time.time() - t0:.0f}s", flush=True)
    con.close()
    return out


def blocking_recall(split="train", work="work", gt_path="work/pq/train_ground_truth.parquet"):
    """Train-only: link recall and oracle macro F0.5 (empty prediction for singletons = 1.0)."""
    con = duckdb.connect()
    w = Path(work) / split
    from .validation import load_truth
    con.execute(f"CREATE VIEW rec AS SELECT * FROM read_parquet({normalized_sql(w)})")
    load_truth(con, gt_path)
    res = {}
    for label, cond in (("all", "TRUE"), ("top_s10", "c.r_s <= 10"), ("top_s25", "c.r_s <= 25")):
        r = con.execute(f"""
            WITH hit AS (SELECT g.sid, count(c.tid) AS hits, count(*) AS n FROM gt g
                         LEFT JOIN (SELECT * FROM read_parquet({lit(w / 'cands.parquet')}) c WHERE {cond}) c USING (sid, tid) GROUP BY g.sid),
                 s1 AS (SELECT rid AS sid FROM read_parquet({lit(w / 'norm_s1.parquet')}))
            SELECT sum(hits)::DOUBLE / sum(n),
                   (SELECT avg(coalesce(5.0 * h.hits / (5 * h.hits + (h.n - h.hits)), 1.0))
                    FROM s1 LEFT JOIN hit h USING (sid))
            FROM hit""").fetchone()
        res[label] = {"link_recall": r[0], "oracle_f05": r[1]}
    res["pairs_per_s1"] = con.execute(f"SELECT count(*) / count(DISTINCT sid) FROM read_parquet({lit(w / 'cands.parquet')})").fetchone()[0]
    con.close()
    return res
