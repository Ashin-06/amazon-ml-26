"""Rescue route (India/US): references and targets sharing the first core-name word and the leading house
number that blocking did not produce. Rescue score = name edit ratio + address token-set ratio (both /100).
Blocking adds route pairs with score >= PRE to the candidate set; the matcher accepts score >= T for targets
nobody owns (best reference per target). T chosen on the tune half; final half scored once.
usage: python rescue.py eval | python rescue.py test BASE.tsv OUT_PAIRS.parquet"""
import sys, numpy as np, pandas as pd, duckdb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
PRE = 1.5
import os
KEY = os.environ.get('RESCUE_KEY', 't.fw=s.fw')
EXCL = os.environ.get('RESCUE_EXCL', 'FALSE')
def route(con, refs_sql, norm):
    con.execute(f"CREATE OR REPLACE TABLE r AS SELECT rid, src, country, name, addr, string_split(core,' ')[1] fw, string_split(core,' ')[2] sw, left(string_split(core,' ')[1],3) f3, regexp_extract(addr,'[0-9]+') num FROM read_parquet(['{norm}/norm_s1.parquet','{norm}/norm_s2.parquet','{norm}/norm_s3.parquet'])")
    con.execute(f"CREATE OR REPLACE TABLE s AS SELECT r.* FROM r WHERE src=1 AND num<>'' AND fw<>'' AND rid IN ({refs_sql})")
def scored(con, extra_where, chunks):
    parts = []
    for k in range(chunks):
        d = con.execute(f"""SELECT DISTINCT s.rid sid, t.rid tid, s.name sn, t.name tn, s.addr sa, t.addr ta FROM s JOIN r t ON t.src>1 AND t.country=s.country AND t.num=s.num AND ({KEY})
            WHERE hash(s.rid)%{chunks}={k} AND NOT ({EXCL}) {extra_where}""").df()
        d['score'] = (cpdist(d.sn.tolist(), d.tn.tolist(), scorer=fuzz.ratio, workers=-1) + cpdist(d.sa.tolist(), d.ta.tolist(), scorer=fuzz.token_set_ratio, workers=-1)) / 100
        parts.append(d.loc[d.score >= PRE, ['sid', 'tid', 'score']]); print('chunk', k, len(d), flush=True)
    return pd.concat(parts, ignore_index=True)
def f05(tp, fp, fn):
    P = np.where(tp+fp > 0, tp/np.maximum(tp+fp, 1), 1.0); R = np.where(tp+fn > 0, tp/np.maximum(tp+fn, 1), 1.0)
    f = np.where(tp > 0, 1.25*P*R/np.maximum(0.25*P+R, 1e-12), 0.0); return np.where((tp+fp == 0) & (tp+fn == 0), 1.0, f).mean()
def pick(x, t):  # unowned targets, best reference per target
    a = x[(x.score >= t) & ~x.owned].sort_values(['tid', 'score', 'sid'], ascending=[True, False, True])
    return a.drop_duplicates('tid')
if sys.argv[1] == 'eval':
    S = 'artifacts/push995-20260927-0915'; M = 'work/v24/kaggle/work/model'; N = 'work/v21L/run/train'
    con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='7GB'")
    route(con, f"SELECT DISTINCT sid FROM read_parquet('{M}/hold_preds.parquet')", N)
    con.execute(f"CREATE TABLE cand AS SELECT sid, tid FROM read_parquet('{M}/hold_preds.parquet')")
    x = scored(con, "AND NOT EXISTS (SELECT 1 FROM cand c WHERE c.sid=s.rid AND c.tid=t.rid)", 4)
    gt = pd.read_parquet('work/comp/hold_truth.parquet'); x = x.merge(gt.assign(y=1), on=['sid', 'tid'], how='left').fillna({'y': 0})
    own = set(pd.read_parquet(f'{S}/final_v1/locked_predictions.parquet').query('lgb_small_tree>=0.77').tid); x['owned'] = x.tid.isin(own)
    halves = {'tune': pd.read_parquet(f'{S}/stack_v4/tune_references.parquet').set_index('sid'), 'final': pd.read_parquet(f'{S}/final_v1/error_by_reference.parquet').set_index('sid')}
    best = None
    for t in np.round(np.arange(1.60, 2.001, 0.02), 2):
        row = [t]
        for h, er in halves.items():
            a = pick(x[x.sid.isin(er.index)], t); g = a.groupby('sid').y.agg(['sum', 'count'])
            tp, fp, fn = er.tp.copy(), er.fp.copy(), er.fn.copy(); tp.loc[g.index] += g['sum']; fp.loc[g.index] += g['count']-g['sum']; fn.loc[g.index] -= g['sum']
            row += [len(a), round(a.y.mean(), 3) if len(a) else None, f05(tp.values, fp.values, fn.values) - f05(er.tp.values, er.fp.values, er.fn.values)]
        print(row, flush=True)
        if best is None or row[3] > best[3]: best = row
    print('TUNE-SELECTED', best)
else:
    base, outp = sys.argv[2], sys.argv[3]; T = float(sys.argv[4]); N = 'work/v21L/run/test'
    con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='7GB'")
    route(con, f"SELECT rid FROM read_parquet('{N}/norm_s1.parquet') WHERE country IN ({os.environ.get('RESCUE_COUNTRIES', chr(39)+'india'+chr(39)+','+chr(39)+'us'+chr(39))})", N)
    x = scored(con, "", 24)
    con.register('xr', x[['sid', 'tid']])  # drop pairs blocking already produced (streamed anti-join, no 179M table in RAM)
    known = con.execute("SELECT xr.sid, xr.tid FROM xr SEMI JOIN read_parquet('work/v24/kaggle/out/scores.parquet') c ON c.sid=xr.sid AND c.tid=xr.tid").df()
    x = x.merge(known.assign(k=1), on=['sid', 'tid'], how='left'); x = x[x.k.isna()].drop(columns='k')
    ids = con.execute(f"SELECT rid, entity_id FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])").df().set_index('rid').entity_id
    owned = set(t for row in pd.read_csv(base, sep='\t', dtype=str, keep_default_na=False, quoting=3).matched_entity_ids for t in row.split(',') if t)
    x['owned'] = x.tid.map(ids).isin(owned); x['se'] = x.sid.map(ids); x['te'] = x.tid.map(ids)
    x.to_parquet(outp, index=False); a = pick(x, T); print('route candidates (score>=PRE)', len(x), 'accepted', len(a))
