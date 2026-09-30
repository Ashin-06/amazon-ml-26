# Hard-pair training set for the cross-encoder: v19's out-of-fold stage-1 scores (p1, copied from the training run)
# pick the uncertain band 0.02 < p1 < 0.995 among NON-holdout references (holdout = md5(entity_id||'hold26') % 22 = 0,
# the lexicon/model rule), plus a small share of confident pairs. The run featurized candidates in (sid, tid) order
# chunk by chunk over ascending sid ranges, so p1 aligns with cands.parquet sorted by (sid, tid).
# Usage: python work/xenc_hard_export.py P1.npy OUT_DIR [N_HARD=3000000] [N_EASY=400000]
import sys, duckdb, numpy as np, pandas as pd
p1_path, out = sys.argv[1], sys.argv[2]
n_hard = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000
n_easy = int(sys.argv[4]) if len(sys.argv) > 4 else 400_000
N, PQ = "work/v19/run/train", "work/v14/pq"
con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='1500MB'; SET temp_directory='work/duck_tmp_rt'")
p1 = np.load(p1_path, mmap_mode="r")
hold = set(con.execute(f"SELECT rid FROM read_parquet('{N}/norm_s1.parquet') WHERE md5_number_lower(entity_id || 'hold26') % 22 = 0").df().rid)
n_sid = con.execute(f"SELECT max(sid) + 1 FROM read_parquet('{N}/cands.parquet')").fetchone()[0]
rng, off, hard, easy, n_band = np.random.default_rng(26), 0, [], [], 0
for lo in range(0, n_sid, 200_000):
    k = con.execute(f"SELECT sid, tid FROM read_parquet('{N}/cands.parquet') WHERE sid >= {lo} AND sid < {lo + 200_000} ORDER BY sid, tid").fetchnumpy()
    p = np.asarray(p1[off:off + len(k["sid"])]); off += len(k["sid"])
    d = pd.DataFrame({"sid": k["sid"], "tid": k["tid"], "p": p})
    d = d[~d.sid.isin(hold)]
    b = (d.p > 0.02) & (d.p < 0.995)
    n_band += int(b.sum())
    hard.append(d[b]); easy.append(d[~b].sample(frac=0.01, random_state=lo))
if off != len(p1):
    raise SystemExit(f"p1 has {len(p1):,} rows but candidates {off:,}: alignment broken")
hard, easy = pd.concat(hard, ignore_index=True), pd.concat(easy, ignore_index=True)
print(f"non-holdout band pairs {n_band:,}; sampling {min(n_hard, len(hard)):,} hard + {min(n_easy, len(easy)):,} easy", flush=True)
d = pd.concat([hard.sample(min(n_hard, len(hard)), random_state=1), easy.sample(min(n_easy, len(easy)), random_state=1)], ignore_index=True)
con.register("d", d[["sid", "tid"]])
g = con.execute(f"""WITH x AS (SELECT a.rid sid, b.rid tid FROM (SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
      FROM 'work/pq/train_ground_truth.parquet' WHERE coalesce(matched_entity_ids,'') <> '') gt
      JOIN read_parquet('{N}/norm_s1.parquet') a ON a.entity_id = gt.s
      JOIN read_parquet(['{N}/norm_s2.parquet','{N}/norm_s3.parquet']) b ON b.entity_id = gt.t)
    SELECT d.sid, d.tid FROM d SEMI JOIN x ON x.sid = d.sid AND x.tid = d.tid""").df()
d = d.merge(g.assign(y=1), on=["sid", "tid"], how="left").fillna({"y": 0})
d["y"] = d.y.astype(int)
ids = pd.DataFrame({"rid": np.unique(np.r_[d.sid, d.tid])}); con.register("ids", ids)
txt = con.execute(f"""SELECT n.rid, coalesce(r.business_name, '') || ' | ' || coalesce(r.business_address, '') AS txt
    FROM read_parquet('{N}/norm_s*.parquet') n JOIN ids USING (rid)
    JOIN read_parquet(['{PQ}/train_source1.parquet','{PQ}/train_source2.parquet','{PQ}/train_source3.parquet']) r
    ON r.entity_id = n.entity_id""").df().set_index("rid").txt
d = d.assign(a=txt.loc[d.sid].values, b=txt.loc[d.tid].values).sample(frac=1, random_state=2).reset_index(drop=True)
assert not d.sid.isin(hold).any()
d.to_parquet(f"{out}/hard_pairs.parquet", index=False)
print(f"wrote {len(d):,} pairs, pos {d.y.mean():.3f}, band share {((d.p > 0.02) & (d.p < 0.995)).mean():.3f}")
