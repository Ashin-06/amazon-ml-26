"""Read-only tuning diagnostics for candidate cost, recall and oracle macro F0.5.

Policies use only blocking evidence before any model scoring. This module never
changes an existing candidate file, selects a matcher threshold or reads final labels.
"""
import argparse
import json
import math
import time
from pathlib import Path

import duckdb

from .integrity import code_digest, digest, verify
from .model import configure, hold_rule, lit, normalized_sql


def profile(con, labeled=True):
    """Summarize c against every refs row, including zero-candidate references.

    Tables: refs(sid,country), targets(country,n_targets), c(sid,tid),
    and, when labeled, gt(sid,tid). All candidate/truth keys must be unique.
    Only aggregate country counts are required for target-pool sizing.
    """
    n, unique = con.execute("SELECT count(*), count(DISTINCT sid) FROM refs").fetchone()
    if not n or n != unique:
        raise ValueError("Reference keys must be nonempty and unique")
    for table in (("c", "gt") if labeled else ("c",)):
        bad = con.execute(f"""SELECT count(*) FROM {table} p LEFT JOIN refs r USING(sid)
            WHERE r.sid IS NULL OR p.sid IS NULL OR p.tid IS NULL""").fetchone()[0]
        duplicate = con.execute(f"SELECT count(*)-count(DISTINCT(sid,tid)) FROM {table}").fetchone()[0]
        if bad or duplicate:
            raise ValueError("Invalid or duplicate " + table + " pairs")
    if con.execute("SELECT count(*) FROM targets WHERE n_targets < 0 OR country IS NULL").fetchone()[0]:
        raise ValueError("Invalid target-pool sizes")
    if con.execute("SELECT count(*)-count(DISTINCT country) FROM targets").fetchone()[0]:
        raise ValueError("Duplicate target country totals")
    con.execute("CREATE OR REPLACE TEMP TABLE cost AS SELECT sid, count(*) nc FROM c GROUP BY sid")
    if labeled:
        con.execute("""CREATE OR REPLACE TEMP TABLE quality AS
            SELECT g.sid, count(*) nt, count(c.tid) hits FROM gt g
            LEFT JOIN c USING(sid,tid) GROUP BY g.sid""")
        label_cols = ", coalesce(q.nt,0) nt, coalesce(q.hits,0) hits"
        label_join = "LEFT JOIN quality q USING(sid)"
    else:
        label_cols, label_join = "", ""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE per_reference AS
        SELECT r.*, coalesce(c.nc,0) nc, coalesce(t.n_targets,0) pool {label_cols}
        FROM refs r LEFT JOIN cost c USING(sid) LEFT JOIN targets t USING(country) {label_join}""")
    result = {}
    countries = [r[0] for r in con.execute("SELECT DISTINCT country FROM refs ORDER BY country").fetchall()]
    for country in [None] + countries:
        where = "TRUE" if country is None else "country=" + lit(country)
        fields = "count(*),sum(nc),avg(nc),quantile_cont(nc,.5),quantile_cont(nc,.95),quantile_cont(nc,.99),max(nc),count(*) FILTER(WHERE nc=0),sum(pool)"
        values = con.execute(f"SELECT {fields} FROM per_reference WHERE {where}").fetchone()
        keys = ("references", "pairs", "mean_candidates", "p50_candidates", "p95_candidates",
                "p99_candidates", "max_candidates", "zero_candidate_references", "same_country_exhaustive_pairs")
        row = dict(zip(keys, values))
        row["reduction_ratio"] = 1 - row["pairs"] / row["same_country_exhaustive_pairs"] if row["same_country_exhaustive_pairs"] else None
        if labeled:
            nt, hits, oracle, std, singles = con.execute(f"""SELECT sum(nt),sum(hits),
                avg(CASE WHEN nt=0 THEN 1.0 ELSE 5.0*hits/(4*hits+nt) END),
                stddev_samp(CASE WHEN nt=0 THEN 1.0 ELSE 5.0*hits/(4*hits+nt) END),
                count(*) FILTER(WHERE nt=0) FROM per_reference WHERE {where}""").fetchone()
            row.update(true_links=nt, retrieved_true_links=hits, singleton_references=singles,
                       link_recall=hits/nt if nt else None, oracle_macro_f05=oracle,
                       oracle_standard_error=std/math.sqrt(row["references"]) if std is not None else None,
                       candidate_precision=hits/row["pairs"] if row["pairs"] else None)
        if country is None:
            result["all"] = row
        else:
            result.setdefault("by_country", {})[country] = row
    return result


def policy_condition(k, k_field, prefix=""):
    """Validated label-free union of total/name/address blocking ranks."""
    if any(type(v) is not int or v < 0 for v in (k,k_field)) or not (k or k_field):
        raise ValueError("Candidate ranks need nonnegative integers and a nonempty route")
    if prefix not in ("", "route_"):
        raise ValueError("Invalid routing-column prefix")
    return f"{prefix}r_s <= {k} OR ({prefix}r_n <= {k_field} AND sn > 0) OR ({prefix}r_a <= {k_field} AND sa > 0)"


def benchmark(work, gt_path, output, sample=5000, mem="512MB", threads=1):
    """Profile natural saved candidates on fixed tuning references of a verified historical run.

    Historical source may differ: this is explicitly a retrieval diagnostic, never
    permission to reuse its files for changed-source training. Final references are excluded.
    """
    if type(sample) is not int or sample < 1:
        raise ValueError("sample must be a positive integer")
    output, work = Path(output), Path(work)
    if output.exists():
        raise FileExistsError("Use a new diagnostic report path")
    t0 = time.monotonic()
    stage = verify(work/"train", "stage1_meta.json")
    if stage["inputs"].get("split_version") != "md5-hold26-v1":
        raise ValueError("Unsupported historical split rule")
    expected = stage["inputs"]["files"].get(Path(gt_path).name, {})
    if expected.get("sha256") != digest(gt_path):
        raise ValueError("Labels do not match the completed historical stage")
    output.parent.mkdir(parents=True,exist_ok=True)
    with duckdb.connect() as con:
        configure(con,mem,threads,output.parent/"efficiency_spill")
        con.execute(f"""CREATE TABLE refs AS SELECT rid sid, entity_id, country
            FROM read_parquet({lit(work/'train/norm_s1.parquet')})
            WHERE {hold_rule('entity_id',stage['inputs']['settings']['hold_mod'])}
            AND md5_number_lower(entity_id || 'tune26') % 2 = 0
            ORDER BY md5(entity_id || 'candidate-pilot26') LIMIT {sample}""")
        con.execute(f"""CREATE TABLE targets AS SELECT country,count(*) n_targets
            FROM read_parquet({normalized_sql(work/'train')}) WHERE src>1 GROUP BY country""")
        con.execute(f"""CREATE TABLE pool AS SELECT c.* FROM read_parquet({lit(work/'train/cands.parquet')}) c
            SEMI JOIN refs r ON c.sid=r.sid""")
        con.execute(f"""CREATE TABLE named_truth AS SELECT r.sid,
            unnest(string_split(g.matched_entity_ids,',')) entity_id FROM read_parquet({lit(gt_path)}) g
            JOIN refs r ON r.entity_id=g.source1_entity_id WHERE coalesce(g.matched_entity_ids,'')<>''""")
        con.execute(f"""CREATE TABLE gt AS SELECT g.sid,r.rid tid FROM named_truth g
            JOIN read_parquet({normalized_sql(work/'train')}) r USING(entity_id) WHERE r.src>1""")
        if con.execute("SELECT (SELECT count(*) FROM named_truth)!=(SELECT count(*) FROM gt)").fetchone()[0]:
            raise ValueError("Truth endpoints disappeared during diagnostic lookup")
        policies = {}
        prefix = "route_" if "route_r_s" in {r[0] for r in con.execute('DESCRIBE pool').fetchall()} else ""
        for label, condition in [("current", "TRUE")] + [
                (f"total{k}_field{f}",policy_condition(k,f,prefix))
                for k,f in ((10,0),(25,0),(10,5),(20,10),(25,10),(30,15),(40,20))]:
            con.execute(f"CREATE OR REPLACE TEMP VIEW c AS SELECT * FROM pool WHERE {condition}")
            policies[label] = profile(con)
        for key, item in policies.items():
            item["dominated"] = any(other["all"]["mean_candidates"] <= item["all"]["mean_candidates"]
                and other["all"]["oracle_macro_f05"] >= item["all"]["oracle_macro_f05"]
                and (other["all"]["mean_candidates"] < item["all"]["mean_candidates"]
                     or other["all"]["oracle_macro_f05"] > item["all"]["oracle_macro_f05"])
                for name,other in policies.items() if name != key)
        report = {"scope":"historical full-target-pool retrieval; tuning references only; no matcher score",
                  "stage_manifest_sha256":digest(work/'train/stage1_meta.json'),
                  "diagnostic_code":code_digest(),"stage_code":stage['inputs']['code'],
                  "sample_requested":sample,"policies":policies,"seconds":time.monotonic()-t0}
    with output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2)
    return report


def main():
    """Run a bounded read-only tuning frontier against a completed, fingerprinted stage."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work"); parser.add_argument("truth"); parser.add_argument("output")
    parser.add_argument("--sample",type=int,default=5000)
    parser.add_argument("--mem",default="512MB");parser.add_argument("--threads",type=int,default=1)
    a=parser.parse_args()
    r=benchmark(a.work,a.truth,a.output,a.sample,a.mem,a.threads)
    print(json.dumps({k:v["all"] for k,v in r["policies"].items()},indent=2))


if __name__ == "__main__":
    main()
