"""Stage 2: candidate generation (blocking) with DuckDB set joins.

Every record gets hashed keys, namespaced by country and field:
  name    : core-name tokens, adjacent-token concatenations ("paulita"+"wilson"
            -> "paulitawilson", catches @handles / domains), first-3 concat.
  address : tokens (len>=3 or containing a digit) and adjacent token bigrams
            ("1550 beverly") that survive component reordering.
Keys held by more than `cap` targets are dropped (too common to discriminate).
Pair score = sum of idf over shared keys (total, name-only, address-only).
Kept per S1: top-`k` by total score  UNION  top-`k_field` by name  UNION
top-`k_field` by address, so address-only (renamed DBA) and name-only
(missing address) matches both survive. Output: work/<split>/cands.parquet.
"""
import time
from pathlib import Path

import duckdb

KEYS_SQL = """
CREATE OR REPLACE TABLE keys AS
WITH t AS (
  SELECT rid, src, country,
         list_filter(string_split(core, ' '), x -> length(x) >= 2) AS nt,
         list_filter(string_split(addr, ' '), x -> x <> '') AS atok
  FROM rec)
SELECT rid, src, 0::TINYINT AS f, hash(country || '|n|' || k) AS h FROM (
  SELECT rid, src, country, unnest(list_distinct(
      nt || list_transform(range(1, len(nt)), i -> nt[i] || nt[i + 1])
         || CASE WHEN len(nt) >= 3 THEN [nt[1] || nt[2] || nt[3]] ELSE [] END)) AS k
  FROM t)
UNION ALL
SELECT rid, src, 1::TINYINT, hash(country || '|a|' || k) FROM (
  SELECT rid, src, country, unnest(list_distinct(
      list_filter(atok, x -> length(x) >= 3 OR regexp_matches(x, '[0-9]'))
      || list_transform(range(1, len(atok)), i -> atok[i] || ' ' || atok[i + 1]))) AS k
  FROM t)
"""


def block(split, work="work", cap=300, k=25, k_field=10, chunks=8, threads=None, mem="20GB"):
    out = Path(work) / split / "cands.parquet"
    if out.exists():
        return out
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{Path(work) / 'duck_tmp'}'")
    if threads:
        con.execute(f"SET threads={threads}")
    con.execute(f"CREATE TABLE rec AS SELECT rid, src, country, core, addr FROM read_parquet('{Path(work) / split}/norm_s*.parquet')")
    con.execute(KEYS_SQL)
    con.execute("DROP TABLE rec")
    n_t = con.execute("SELECT count(DISTINCT rid) FROM keys WHERE src > 1").fetchone()[0]
    # Target-side postings with document frequency; common keys dropped.
    con.execute(f"""
        CREATE TABLE post AS
        WITH d AS (SELECT h, count(*) AS df FROM keys WHERE src > 1 GROUP BY h HAVING count(*) <= {cap})
        SELECT k.h, k.rid AS tid, k.f, ln(1 + {n_t} / d.df)::FLOAT AS w
        FROM keys k JOIN d USING (h) WHERE k.src > 1""")
    con.execute("CREATE TABLE q AS SELECT rid AS sid, h FROM keys WHERE src = 1")
    con.execute("DROP TABLE keys")
    print(f"[block {split}] keys ready {time.time() - t0:.0f}s  targets={n_t:,}", flush=True)
    tmp = Path(work) / split / "cands_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    for c in range(chunks):
        part = tmp / f"{c}.parquet"
        if part.exists():
            continue
        con.execute(f"""
            COPY (
              WITH s AS (
                SELECT q.sid, p.tid, sum(p.w)::FLOAT AS s,
                       coalesce(sum(p.w) FILTER (WHERE p.f = 0), 0)::FLOAT AS sn,
                       coalesce(sum(p.w) FILTER (WHERE p.f = 1), 0)::FLOAT AS sa,
                       count(*)::SMALLINT AS nk
                FROM q JOIN post p USING (h)
                WHERE q.sid % {chunks} = {c}
                GROUP BY q.sid, p.tid),
              r AS (
                SELECT *, row_number() OVER (PARTITION BY sid ORDER BY s DESC, tid) AS r_s,
                          row_number() OVER (PARTITION BY sid ORDER BY sn DESC, tid) AS r_n,
                          row_number() OVER (PARTITION BY sid ORDER BY sa DESC, tid) AS r_a
                FROM s)
              SELECT sid, tid, s, sn, sa, nk, r_s::SMALLINT AS r_s,
                     (CASE WHEN sn > 0 THEN r_n ELSE 9999 END)::SMALLINT AS r_n,
                     (CASE WHEN sa > 0 THEN r_a ELSE 9999 END)::SMALLINT AS r_a
              FROM r WHERE r_s <= {k} OR (r_n <= {k_field} AND sn > 0) OR (r_a <= {k_field} AND sa > 0)
            ) TO '{part}' (FORMAT parquet)""")
        print(f"[block {split}] chunk {c + 1}/{chunks} {time.time() - t0:.0f}s", flush=True)
    con.execute(f"COPY (SELECT * FROM read_parquet('{tmp}/*.parquet')) TO '{out}' (FORMAT parquet)")
    n = con.execute(f"SELECT count(*), count(DISTINCT sid) FROM '{out}'").fetchone()
    print(f"[block {split}] done {n[0]:,} pairs for {n[1]:,} S1  {time.time() - t0:.0f}s", flush=True)
    return out


def blocking_recall(split="train", work="work", gt_path="work/pq/train_ground_truth.parquet"):
    """Train-only: link recall and oracle macro F0.5 (empty prediction for singletons = 1.0)."""
    con = duckdb.connect()
    w = Path(work) / split
    con.execute(f"CREATE VIEW ids AS SELECT rid, entity_id FROM read_parquet('{w}/norm_s*.parquet')")
    con.execute(f"""CREATE TABLE gt AS
        SELECT a.rid AS sid, b.rid AS tid FROM (
          SELECT source1_entity_id AS s, unnest(string_split(matched_entity_ids, ',')) AS t
          FROM '{gt_path}' WHERE coalesce(matched_entity_ids, '') <> '') g
        JOIN ids a ON a.entity_id = g.s JOIN ids b ON b.entity_id = g.t""")
    res = {}
    for label, cond in (("all", "TRUE"), ("top_s10", "c.r_s <= 10"), ("top_s25", "c.r_s <= 25")):
        r = con.execute(f"""
            WITH hit AS (SELECT g.sid, count(c.tid) AS hits, count(*) AS n FROM gt g
                         LEFT JOIN (SELECT * FROM '{w}/cands.parquet' c WHERE {cond}) c USING (sid, tid) GROUP BY g.sid),
                 s1 AS (SELECT rid AS sid FROM read_parquet('{w}/norm_s1.parquet'))
            SELECT sum(hits)::DOUBLE / sum(n),
                   (SELECT avg(coalesce(5.0 * h.hits / (5 * h.hits + (h.n - h.hits)), 1.0))
                    FROM s1 LEFT JOIN hit h USING (sid))
            FROM hit""").fetchone()
        res[label] = {"link_recall": r[0], "oracle_f05": r[1]}
    res["pairs_per_s1"] = con.execute(f"SELECT count(*) / count(DISTINCT sid) FROM '{w}/cands.parquet'").fetchone()[0]
    return res
