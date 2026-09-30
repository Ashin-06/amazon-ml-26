"""Three-fold S1-grouped calibration experiment on existing tuning references.

Uses frozen v19/xenc4 predictions. Final labels never enter calibration or
metrics. Target ownership uses the held-reference pool, not the full training
graph; results are tuning diagnostics, never advertised as final/public scores.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import psutil
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def scores_for_threshold(sid, tid, probability, label, ref_ids, true_counts, threshold):
    """Held-pool ownership and complete tuning-reference F0.5 vector."""
    order = np.lexsort((sid, -probability, tid))
    first = np.r_[True, tid[order][1:] != tid[order][:-1]]
    win = order[first]
    return evaluate_winners(sid[win], probability[win], label[win], ref_ids, true_counts, threshold)


def evaluate_winners(sid, probability, label, ref_ids, true_counts, threshold):
    """Compute metrics after one global held-pool winner per target is fixed."""
    ix = np.searchsorted(ref_ids, sid)
    in_tune = ix < len(ref_ids)
    in_tune &= ref_ids[np.minimum(ix, len(ref_ids)-1)] == sid
    chosen = in_tune & (probability >= threshold)
    if np.any(label[chosen] < 0):
        raise ValueError("Missing tuning label")
    pp = np.bincount(ix[chosen], minlength=len(ref_ids))
    tp = np.bincount(ix[chosen], weights=label[chosen], minlength=len(ref_ids))
    den = true_counts + 4 * pp
    out = np.ones(len(ref_ids), dtype=float)
    np.divide(5 * tp, den, out=out, where=den != 0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--xe", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    a = parser.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    proc = psutil.Process()
    if os.name == "nt":
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start = time.monotonic()
    peak = proc.memory_info().rss
    inputs = [a.model_dir / "hold_error_rows.parquet", a.model_dir / "hold_preds.parquet", a.xe]
    stamps = {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
    rr = []
    for b in pq.ParquetFile(inputs[0]).iter_batches(batch_size=8192, columns=["sid","half","country","true_links"], use_threads=False):
        rr.extend(b.filter(pc.equal(b.column("half"), "tune")).to_pylist())
    ref = pd.DataFrame(rr).sort_values("sid"); del rr
    ref_ids = ref.sid.to_numpy(); true_counts = ref.true_links.to_numpy()
    if len(ref_ids) != len(np.unique(ref_ids)):
        raise ValueError("Duplicate tuning reference")
    tune_set = pa.array(ref_ids)
    frames = []
    for b in pq.ParquetFile(inputs[1]).iter_batches(batch_size=65536, columns=["sid","tid","p","y"], use_threads=False):
        # All potential scoring-band pairs plus unchanged high-confidence pairs.
        b = b.filter(pc.greater(b.column("p"), .002))
        # Discard final labels before creating a pandas frame; keep only their
        # unlabeled scores to provide the same held-pool competitor context.
        b = b.set_column(b.schema.get_field_index("y"), "y", pc.if_else(pc.is_in(b.column("sid"), value_set=tune_set), pc.cast(b.column("y"),pa.int8()), pa.scalar(-1,pa.int8())))
        frames.append(b.to_pandas())
    d = pd.concat(frames, ignore_index=True); del frames
    if d.duplicated(["sid","tid"]).any():
        raise ValueError("Duplicate saved candidate key")
    xe = pd.read_parquet(a.xe)
    if xe.duplicated(["sid","tid"]).any() or not np.isfinite(xe.xe).all():
        raise ValueError("Invalid cross-encoder keys/scores")
    expected = d[(d.p > .002) & (d.p < .9995)][["sid","tid"]]
    coverage = expected.merge(xe[["sid","tid"]],on=["sid","tid"],how="outer",indicator=True,validate="one_to_one")
    if not coverage._merge.eq("both").all():
        raise ValueError("Cross-encoder keys do not match the exact band")
    del expected, coverage
    d = d.merge(xe,on=["sid","tid"],how="left",validate="one_to_one"); del xe
    if not np.isfinite(d.p).all() or not d.p.between(0,1).all():
        raise ValueError("Invalid base probabilities")
    folds = {int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little') % 3 for s in ref_ids}
    is_tune = d.y.to_numpy() >= 0
    band = d.xe.notna().to_numpy()
    row_fold = d.sid.map(folds).fillna(-1).to_numpy(dtype=np.int8)
    lp = np.log(np.clip(d.p.to_numpy(),1e-6,1-1e-6)/(1-np.clip(d.p.to_numpy(),1e-6,1-1e-6)))
    xx = d.xe.fillna(0).to_numpy()
    sid,tid,labels = d.sid.to_numpy(),d.tid.to_numpy(),d.y.to_numpy()
    grid = sorted(set(np.round(np.arange(.40,.991,.01),4)) | {.7125,.925})
    outputs={}
    with threadpool_limits(limits=1):
        for name,X in [('base',None),('linear_logits',np.c_[lp,xx]),('interaction_logits',np.c_[lp,xx,lp*xx])]:
            probability=d.p.to_numpy().copy(); coefficients=[]
            if X is not None:
                other=band & ~is_tune
                other_p=np.zeros(int(other.sum()))
                for fold in range(3):
                    fit=band & is_tune & (row_fold!=fold)
                    check=band & is_tune & (row_fold==fold)
                    lr=LogisticRegression(max_iter=2000,tol=1e-6).fit(X[fit],labels[fit])
                    probability[check]=lr.predict_proba(X[check])[:,1]
                    other_p+=lr.predict_proba(X[other])[:,1]/3
                    coefficients.append({'fold':fold,'fit_pairs':int(fit.sum()),'validation_pairs':int(check.sum()),'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])})
                probability[other]=other_p
            order=np.lexsort((sid,-probability,tid))
            win=order[np.r_[True,tid[order][1:]!=tid[order][:-1]]]
            ww=(sid[win],probability[win],labels[win],ref_ids,true_counts)
            curve={str(t):float(evaluate_winners(*ww,t).mean()) for t in grid}
            threshold=max(grid,key=lambda t:(curve[str(t)],t))
            values=evaluate_winners(*ww,threshold)
            outputs[name]={'selected_threshold':threshold,'tune_macro_f05':float(values.mean()),'by_country':{c:float(values[ref.country.to_numpy()==c].mean()) for c in sorted(ref.country.unique())},'at_export_thresholds':{str(t):curve[str(t)] for t in [.9,.925]},'threshold_curve':curve,'fold_coefficients':coefficients}
            peak=max(peak,proc.memory_info().rss)
    for path,stat in stamps.items():
        p=Path(path)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=stat:
            raise ValueError("An input changed during evaluation")
    info={'protocol':'Tune-only, three-fold S1-grouped out-of-fold calibration, held-pool ownership; non-tune competitors use mean fold predictions; diagnostic not production validation','caveat':'Cutoff selected on OOF tuning predictions; no untouched score. Final labels unused. Public/test and full competitor behavior are not established.','references':len(ref_ids),'retained_candidate_rows':len(d),'band_pairs':int(band.sum()),'tune_band_pairs':int((band & is_tune).sum()),'fold_reference_counts':dict(Counter(folds.values())) if False else {str(f):sum(v==f for v in folds.values()) for f in range(3)},'seconds':time.monotonic()-start,'sampled_peak_rss_MiB':peak/2**20,'input_stats':stamps,'xe_sha256':hashlib.sha256(a.xe.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'models':outputs}
    (a.out_dir/'audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({**{k:v for k,v in info.items() if k not in ('models','input_stats')},'models':{k:{kk:vv for kk,vv in v.items() if kk not in ('threshold_curve','fold_coefficients')} for k,v in outputs.items()}},indent=2))


if __name__=='__main__':
    main()
