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

from .db import configure, lit, normalized_sql

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


# Extra natural routes (v21), label-free and identical for every split/country; added to the key-based union:
#  A. same non-empty core name, empty target address: common names lose their name keys to `cap`, and name-only
#     targets have no address keys (tuning pilot: 372 missed true links recovered for 0.85 pairs/reference);
#  B. same exact non-empty address sharing a core word: address-frequency-capped same-address variants that the top-k
#     cut drops in dense same-address clusters (4.8% of such French pairs were missing, ~0% US/India).
EXTRA_SQL = """
CREATE TABLE extra AS
WITH s AS (SELECT rid AS sid, country, core, addr, string_split(core, ' ') AS ct FROM rec WHERE src = 1 AND core <> ''),
     t AS (SELECT rid AS tid, country, core, addr, string_split(core, ' ') AS ct FROM rec WHERE src > 1 AND core <> ''),
     cc AS (SELECT country, core, count(*) AS n, count(*) FILTER (WHERE addr = '') AS ne FROM t GROUP BY 1, 2),
     ac AS (SELECT country, addr, count(*) AS n FROM t WHERE addr <> '' GROUP BY 1, 2),
     a AS (SELECT s.sid, t.tid FROM s JOIN cc USING (country, core) JOIN t USING (country, core)
           WHERE t.addr = '' AND cc.n <= {route_core_max} AND cc.ne <= {route_empty_max}),
     b AS (SELECT s.sid, t.tid FROM s JOIN ac ON ac.country = s.country AND ac.addr = s.addr
           JOIN t ON t.country = s.country AND t.addr = s.addr
           WHERE s.addr <> '' AND ac.n <= {route_addr_max} AND len(list_intersect(s.ct, t.ct)) > 0),
     -- C (v24): name keys counted WITHIN the name-only (empty-address) target pool, so a name that is too common
     -- in the full pool (dropped by `cap`) can still retrieve its name-only variants; posting lists capped at
     -- {route_pool_cap}; top {route_pool_k} per reference by inverse-frequency score (tuning pilot: candidate
     -- oracle 0.99446 -> 0.99566 over v21's candidates, 8.1 pairs/reference).
     e AS (SELECT tid, country, list_filter(string_split(core, ' '), x -> x <> '') tk FROM t WHERE addr = ''),
     sq AS (SELECT sid, country, list_filter(string_split(core, ' '), x -> x <> '') tk FROM s),
     ek AS (SELECT DISTINCT tid, country, unnest(list_distinct(list_transform(list_filter(tk, x -> length(x) >= 2), x -> 'w:' || x)
              || list_transform(range(1, len(tk)), i -> 'w:' || tk[i] || tk[i + 1]) || ['n:' || array_to_string(tk, '')])) AS k FROM e),
     qk AS (SELECT DISTINCT sid, country, unnest(list_distinct(list_transform(list_filter(tk, x -> length(x) >= 2), x -> 'w:' || x)
              || list_transform(range(1, len(tk)), i -> 'w:' || tk[i] || tk[i + 1]) || ['n:' || array_to_string(tk, '')])) AS k FROM sq),
     ef AS (SELECT country, k, count(*) AS df FROM ek GROUP BY 1, 2 HAVING count(*) <= {route_pool_cap}),
     en AS (SELECT count(*) AS n FROM e),
     cs AS (SELECT q.sid, x.tid, sum(ln(1 + (SELECT n FROM en) / f.df)) AS sc FROM qk q JOIN ef f USING (country, k)
            JOIN ek x USING (country, k) GROUP BY 1, 2),
     c AS (SELECT sid, tid FROM cs QUALIFY row_number() OVER (PARTITION BY sid ORDER BY sc DESC, tid) <= {route_pool_k})
SELECT DISTINCT sid, tid FROM (SELECT * FROM a UNION ALL SELECT * FROM b UNION ALL SELECT * FROM c)
"""


def block(split, work="work", cap=300, k=25, k_field=10, chunks=8, threads=None, mem="20GB", cap_whole=1000,
          route_core_max=1000, route_empty_max=64, route_addr_max=50, route_pool_cap=128, route_pool_k=16):
    """Build fresh natural candidates; partial chunks are never treated as completed work."""
    for name, value in (("cap", cap), ("cap_whole", cap_whole), ("k", k), ("k_field", k_field), ("chunks", chunks),
                        ("route_core_max", route_core_max), ("route_empty_max", route_empty_max), ("route_addr_max", route_addr_max),
                        ("route_pool_cap", route_pool_cap), ("route_pool_k", route_pool_k)):
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
    con.execute(EXTRA_SQL.format(route_core_max=route_core_max, route_empty_max=route_empty_max, route_addr_max=route_addr_max,
                                 route_pool_cap=route_pool_cap, route_pool_k=route_pool_k))
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
                          row_number() OVER (PARTITION BY sid ORDER BY sa DESC, tid) AS r_a,
                          rank() OVER (PARTITION BY sid ORDER BY s DESC) AS f_s,
                          rank() OVER (PARTITION BY sid ORDER BY sn DESC, sa DESC, te DESC) AS f_n,
                          rank() OVER (PARTITION BY sid ORDER BY sa DESC) AS f_a
                FROM s)
              , x AS (SELECT sid, tid, s, sn, sa, nk, f_s::INTEGER AS r_s,
                     (CASE WHEN sn > 0 THEN f_n ELSE 9999 END)::INTEGER AS r_n,
                     (CASE WHEN sa > 0 THEN f_a ELSE 9999 END)::INTEGER AS r_a,
                     r_s::INTEGER AS route_r_s, r_n::INTEGER AS route_r_n, r_a::INTEGER AS route_r_a,
                     r_s <= {k} OR (r_n <= {k_field} AND sn > 0) OR (r_a <= {k_field} AND sa > 0) AS keep
                FROM r)
              SELECT * EXCLUDE (keep) FROM x WHERE keep
              UNION ALL  -- extra routes: pairs outside the key-based union keep their key scores/ranks (or none)
              SELECT e.sid, e.tid, coalesce(x.s, 0)::FLOAT, coalesce(x.sn, 0)::FLOAT, coalesce(x.sa, 0)::FLOAT,
                     coalesce(x.nk, 0)::INTEGER, coalesce(x.r_s, 9999)::INTEGER, coalesce(x.r_n, 9999)::INTEGER,
                     coalesce(x.r_a, 9999)::INTEGER, coalesce(x.route_r_s, 9999)::INTEGER,
                     coalesce(x.route_r_n, 9999)::INTEGER, coalesce(x.route_r_a, 9999)::INTEGER
              FROM extra e LEFT JOIN x USING (sid, tid)
              WHERE e.sid % {chunks} = {c} AND NOT coalesce(x.keep, false)
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
    for label, cond in (("all", "TRUE"), ("top_s10", "c.route_r_s <= 10"), ("top_s25", "c.route_r_s <= 25")):
        r = con.execute(f"""
            WITH hit AS (SELECT g.sid, count(c.tid) AS hits, count(*) AS n FROM gt g
                         LEFT JOIN (SELECT * FROM read_parquet({lit(w / 'cands.parquet')}) c WHERE {cond}) c USING (sid, tid) GROUP BY g.sid),
                 s1 AS (SELECT rid AS sid FROM read_parquet({lit(w / 'norm_s1.parquet')}))
            SELECT sum(hits)::DOUBLE / sum(n),
                   (SELECT avg(coalesce(5.0 * h.hits / (5 * h.hits + (h.n - h.hits)), 1.0))
                    FROM s1 LEFT JOIN hit h USING (sid))
            FROM hit""").fetchone()
        res[label] = {"link_recall": r[0], "oracle_f05": r[1]}
    res["pairs_per_s1"] = con.execute(f"SELECT count(*) / (SELECT count(*) FROM rec WHERE src=1) FROM read_parquet({lit(w / 'cands.parquet')})").fetchone()[0]
    con.close()
    return res
