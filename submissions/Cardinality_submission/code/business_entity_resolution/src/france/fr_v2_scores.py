"""France re-blend with the v2 French cross-encoder inputs: same blend coefficients (work/v24x4f/blend.json), same
band; verifies the old French blended scores are reproduced exactly from the v1 inputs, then writes French pairs
(all French candidate pairs; band scores replaced) to work/v24x4f2/scores_blend_fr.parquet."""
import json, os, numpy as np, pandas as pd, duckdb
from scipy.special import expit, logit
bl = json.load(open('work/v24x4f/blend.json'))
def blend(p, xe):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return expit(bl['intercept'] + bl['coef_logit_p'] * lp + bl['coef_xe'] * xe + bl['coef_interaction'] * lp * xe)
XE2 = os.environ.get('FR_XE2', 'work/kaggle_out/xenc_inf12/v24_test_fr2_band_xe.parquet'); OUT = os.environ.get('FR_SCORES_OUT', 'work/v24x4f2/scores_blend_fr.parquet')
band = pd.read_parquet('work/kaggle_xenc_inf11/v24_test_fr2_band.parquet', columns=['sid', 'tid', 'p'])
old = band.merge(pd.read_parquet('work/kaggle_out/xenc_inf11/v24_test_fr_band_xe.parquet'), on=['sid', 'tid'], validate='one_to_one')
new = band.merge(pd.read_parquet(XE2), on=['sid', 'tid'], validate='one_to_one')
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='5GB'")
con.execute("CREATE TABLE fr AS SELECT rid FROM read_parquet('work/v21L/run/test/norm_s1.parquet') WHERE country='france'")
sb = con.execute("SELECT b.sid, b.tid, b.p FROM read_parquet('work/v24x4f/scores_blend.parquet') b SEMI JOIN fr ON fr.rid=b.sid").df()
chk = old.merge(sb.rename(columns={'p': 'pb'}), on=['sid', 'tid'])
err = np.abs(blend(chk.p.to_numpy(), chk.xe.to_numpy()) - chk.pb.to_numpy()).max()
print('reproduce old French blend: max abs diff', err, 'pairs', len(chk))
if err > 1e-5: raise SystemExit('blend not reproduced')
new['pn'] = blend(new.p.to_numpy(), new.xe.to_numpy())
out = sb.merge(new[['sid', 'tid', 'pn']], on=['sid', 'tid'], how='left'); out['p'] = out.pn.fillna(out.p)
os.makedirs(os.path.dirname(OUT), exist_ok=True); out[['sid', 'tid', 'p']].to_parquet(OUT, index=False)
cmp = chk.merge(new[['sid', 'tid', 'pn']], on=['sid', 'tid'])
print('French band pairs', len(cmp), '| >=0.925 old', int((cmp.pb >= 0.925).sum()), 'new', int((cmp.pn >= 0.925).sum()),
      '| xe old mean', round(float(old.xe.mean()), 3), 'new', round(float(new.xe.mean()), 3))
