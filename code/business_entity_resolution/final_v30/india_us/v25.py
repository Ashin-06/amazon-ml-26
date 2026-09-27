"""v25: v24 correction model re-selected on tune (LightGBM 600 trees, depth 5, 15 leaves; cutoff 0.74, picked by
grouped 3-fold tune OOF 0.989149 vs 0.989127), fitted on all tune band pairs, scored once on the untouched final
half (0.988918 vs 0.988792 for the 320-tree v24stack model). Same frozen features (stacker.structural_features).
usage: python v25.py fit OUTDIR | score OUTDIR | export OUTDIR FRANCE_SOURCE.tsv OUT.tsv
"""
import os, sys, json, gc, hashlib, importlib.util, time, joblib, numpy as np, pandas as pd, duckdb, lightgbm as lgb
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
P = Path('artifacts/push995-20260927-0915'); NORM = Path('work/v21L/run/test'); NEW = Path('work/v24/kaggle/out')
CUT = float(os.environ.get('V25_CUT', 0.74))
CFG = dict(n_estimators=600, num_leaves=15, max_depth=5, min_child_samples=100, learning_rate=.04, reg_lambda=15, colsample_bytree=.9)
spec = importlib.util.spec_from_file_location('st', P / 'stacker.py'); st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
q = lambda v: "'" + str(v).replace(chr(92), '/') + "'"
mode, out = sys.argv[1], Path(sys.argv[2])
if mode == 'fit':
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(P / 'stack_v4/pairs_labels_masked.parquet'); f = pd.read_parquet(P / 'stack_v4/band_features.parquet')
    ix = np.flatnonzero(d.xe.notna().to_numpy()); y = d.y.to_numpy()[ix]; tune = y >= 0
    mdl = lgb.LGBMClassifier(**CFG, n_jobs=6, verbosity=-1, random_state=26, deterministic=True, force_col_wise=True).fit(f.to_numpy()[tune], y[tune])
    joblib.dump(mdl, out / 'v25_lgb600.joblib')
    json.dump({'features': list(f.columns), 'config': CFG, 'cutoff': CUT, 'train_pairs': int(tune.sum())}, open(out / 'v25_model.json', 'w'), indent=1)
    print('fitted on', int(tune.sum()), 'tune band pairs')
elif mode == 'score':
    mdl = joblib.load(out / 'v25_lgb600.joblib'); feats = json.load(open(out / 'v25_model.json'))['features']
    src = P / 'test_v1/test_band_inputs.parquet'; w = None; n = 0; t0 = time.time()
    for k, b in enumerate(pq.ParquetFile(src).iter_batches(batch_size=125000, use_threads=False)):
        d = b.to_pandas(); tmp = out / f'feat_tmp_{k:03d}'; tmp.mkdir(exist_ok=True)
        f = st.structural_features(d, tmp, norm=NORM)
        if list(f.columns) != feats: raise ValueError('feature order')
        p = mdl.predict_proba(f.to_numpy())[:, 1]
        if not np.isfinite(p).all(): raise ValueError('non-finite')
        z = pa.Table.from_pandas(d[['sid', 'tid']].assign(p=p), preserve_index=False)
        w = w or pq.ParquetWriter(out / 'corrected_band_v25.parquet', z.schema); w.write_table(z); n += len(d)
        print(f'scored {n:,} in {time.time()-t0:.0f}s', flush=True); del d, f, z, p; gc.collect()
    w.close()
elif mode == 'export':
    france, dest = Path(sys.argv[3]), Path(sys.argv[4])
    c = duckdb.connect(); c.execute("SET threads=6; SET memory_limit='5GB'")
    c.execute(f"CREATE TABLE ids AS SELECT rid, entity_id, src, country FROM read_parquet([{q(NORM/'norm_s1.parquet')},{q(NORM/'norm_s2.parquet')},{q(NORM/'norm_s3.parquet')}])")
    c.execute(f"""CREATE TABLE possible AS SELECT s.sid, s.tid, s.p FROM read_parquet({q(NEW/'scores.parquet')}) s JOIN ids r ON r.rid=s.sid
      WHERE r.country IN ('india','us') AND s.p>=.9995 UNION ALL SELECT sid, tid, p FROM read_parquet({q(os.environ.get('V25_BAND', out/'corrected_band_v25.parquet'))}) WHERE p>={CUT}""")
    c.execute("CREATE TABLE winners AS SELECT * FROM possible QUALIFY row_number() OVER (PARTITION BY tid ORDER BY p DESC, sid)=1")
    rd = f"read_csv({q(france)}, delim='\t', header=true, all_varchar=true, quote='', escape='')"
    c.execute(f"""CREATE TABLE fr AS SELECT u.s, u.t FROM (SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t FROM {rd} WHERE coalesce(matched_entity_ids,'')<>'') u
      JOIN ids r ON r.entity_id=u.s AND r.country='france'""")
    c.execute("""CREATE TABLE links AS SELECT a.entity_id s, b.entity_id t FROM winners w JOIN ids a ON a.rid=w.sid JOIN ids b ON b.rid=w.tid UNION ALL SELECT s, t FROM fr""")
    if c.execute("SELECT count(*)-count(DISTINCT t) FROM links").fetchone()[0]: raise ValueError('target owned twice')
    lk = {}
    for s, t in c.execute("SELECT s, t FROM links ORDER BY s, t").fetchall(): lk.setdefault(s, []).append(t)
    refs = [r[0] for r in c.execute(f"SELECT entity_id FROM read_csv('6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='')").fetchall()]
    with open(dest, 'w', encoding='utf-8', newline='\n') as fo:
        fo.write('source1_entity_id\tmatched_entity_ids\n')
        for s in refs: fo.write(s + '\t' + ','.join(lk.get(s, [])) + '\n')
    print('wrote', dest, len(refs), 'references', sum(map(len, lk.values())), 'links')
