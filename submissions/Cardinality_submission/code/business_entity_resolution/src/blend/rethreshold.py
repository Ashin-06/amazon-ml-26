# Rewrite a matching_results.tsv from saved test scores at another threshold, with the pipeline writer's rule:
# p >= thr, then each target to its highest-p S1 (ties -> lowest sid); every S1 gets one row; IDs ordered by rid.
# rid = row position across test_source1, 2, 3 (the numbering the scores use), taken from parquet file row numbers.
# Usage: python work/rethreshold.py SCORES.parquet THRESHOLD OUT.tsv [PQ_DIR (default work/pq)] [FRANCE_THRESHOLD]
# FRANCE_THRESHOLD (optional): separate cutoff for French references (> 1 empties them: a leaderboard probe of France).
import sys
import duckdb

scores, thr, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
pq = sys.argv[4] if len(sys.argv) > 4 else "work/pq"
fr = float(sys.argv[5]) if len(sys.argv) > 5 else thr
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='1500MB'; SET preserve_insertion_order=false; SET temp_directory='work/duck_tmp_rt'")
off, parts = 0, []
for i in (1, 2, 3):
    f = f"{pq}/test_source{i}.parquet"
    parts.append(f"SELECT file_row_number + {off} AS rid, entity_id, {i} AS src, country FROM read_parquet('{f}', file_row_number=true)")
    off += con.execute(f"SELECT count(*) FROM '{f}'").fetchone()[0]
con.execute("CREATE TABLE ids AS " + " UNION ALL ".join(parts))
con.execute(f"""CREATE TABLE pr AS SELECT sid, tid FROM (
    SELECT sid, tid, row_number() OVER (PARTITION BY tid ORDER BY p DESC, sid) r FROM '{scores}' s JOIN ids ON ids.rid = s.sid
    WHERE p >= CASE WHEN ids.country = 'France' THEN {fr} ELSE {thr} END) WHERE r = 1""")
con.execute("""CREATE TABLE agg AS SELECT pr.sid, string_agg(t.entity_id, ',' ORDER BY t.rid) m
               FROM pr JOIN ids t ON t.rid = pr.tid GROUP BY pr.sid""")
cur = con.execute("""SELECT s.entity_id, coalesce(a.m, '') FROM ids s LEFT JOIN agg a ON a.sid = s.rid
                      WHERE s.src = 1 ORDER BY s.rid""")
n = 0
with open(out, "w", encoding="utf-8", newline="\n") as f:  # streamed in batches (bounded memory)
    f.write("source1_entity_id\tmatched_entity_ids\n")
    while rows := cur.fetchmany(200_000):
        f.writelines(f"{a}\t{b}\n" for a, b in rows)
        n += len(rows)
print(f"{out}: {n:,} rows, {con.execute('SELECT count(*) FROM pr').fetchone()[0]:,} links at thr {thr} (France {fr})")
