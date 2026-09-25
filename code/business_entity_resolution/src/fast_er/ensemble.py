"""Verified equal-weight seed ensembles with complete-graph evaluation and immutable outputs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.integrity import code_digest, digest, seal, verify, verify_stage
from fast_er.model import (_connect, _load_truth, _score_all, check_scores, evaluate_threshold,
                           lit, load_models, scratch, write_outputs, model_path)
from fast_er.validation import validate_pairs


def members(work, model_dirs):
    """Load at least two distinct, verified models sharing hold membership and training inputs."""
    if len(model_dirs) < 2 or len(set(model_dirs)) != len(model_dirs):
        raise ValueError("Ensemble needs at least two distinct model directories")
    result, hold = [], None
    for name in model_dirs:
        directory = model_path(work, name)
        first, second, _ = load_models(directory)
        meta = verify(directory, "model_manifest.json")
        split = pd.read_parquet(directory / "split_ids.parquet")
        current = split.loc[split.g == "hold", ["entity_id", "half"]].sort_values("entity_id").reset_index(drop=True)
        if not current.entity_id.is_unique or set(current.half) != {"tune", "final"}:
            raise ValueError("Invalid ensemble hold membership")
        if hold is not None and not current.equals(hold):
            raise ValueError("Ensemble members have different tune/final references")
        if result and meta["inputs"]["train_stage"] != result[0][3]["inputs"]["train_stage"]:
            raise ValueError("Ensemble members have different training inputs")
        hold = current
        result.append((name, first, second, meta))
    return result, hold


def align_scores(base, scores):
    """Align complete unique key sets, rejecting duplicates even if group totals would cancel."""
    check_scores(scores.p.to_numpy())
    if scores[["sid", "tid"]].isna().any().any() or scores.duplicated(["sid", "tid"]).any():
        raise ValueError("Duplicate or null ensemble pair keys")
    scores = scores.sort_values(["sid", "tid"]).reset_index(drop=True)
    if base is not None and not scores[["sid", "tid"]].equals(base):
        raise ValueError("Ensemble members scored different pair sets")
    return scores


def evaluate(work, gt_path, model_dirs, name="ensemble", mem="4GB", threads=1, chunks=24):
    """Re-score every candidate for each selected member; tune once and evaluate final once."""
    target = model_path(work, name)
    if target.exists():
        raise FileExistsError("Ensemble destination exists")
    selected, hold = members(work, model_dirs)
    if digest(gt_path) != next(iter(selected[0][3]["inputs"]["labels"].values()))["sha256"]:
        raise ValueError("Ensemble labels differ from training labels")
    con = _connect(work, "train", mem, threads)
    try:
        _load_truth(con, gt_path)
        con.register("hold_ids", hold)
        refs = con.execute("SELECT r.rid sid,h.half FROM rec r JOIN hold_ids h USING(entity_id)").df()
        labels = con.execute("SELECT sid,tid FROM gt WHERE sid IN (SELECT r.rid FROM rec r JOIN hold_ids h USING(entity_id))").df()
        truth = labels.groupby("sid").tid.apply(set).to_dict()
        base, total = None, None
        for _, first, second, _ in selected:
            keys, _, probabilities = _score_all(con, first, second, threads, chunks, scratch(work))
            scores = align_scores(base, keys.assign(p=probabilities))
            if base is None:
                base = scores[["sid", "tid"]]
                total = np.zeros(len(scores), np.float64)
            total += scores.p.to_numpy() / len(selected)
        result = evaluate_threshold(base.sid.to_numpy(), base.tid.to_numpy(), total, truth,
            refs.sid[refs.half == "tune"].tolist(), refs.sid[refs.half == "final"].tolist())
        target.mkdir(parents=True, exist_ok=False)
        (target / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        seal(target, "ensemble_manifest.json", {"code": code_digest(), "train_stage": verify_stage(work,"train"),
            "members": {n: digest(model_path(work,n)/"model_manifest.json") for n in model_dirs}}, ["metrics.json"])
        return result
    finally:
        con.close()


def apply(work, split, out_dir, score_dirs, name="ensemble", mem="4GB", threads=1):
    """Average verified member predictions on identical candidates using the selected threshold."""
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError("Ensemble output exists")
    selected = verify(model_path(work,name), "ensemble_manifest.json")
    if selected["inputs"]["code"] != code_digest():
        raise ValueError("Ensemble source changed")
    model_dirs = list(selected["inputs"]["members"])
    members(work, model_dirs)
    if {n:digest(model_path(work,n)/"model_manifest.json") for n in model_dirs} != selected["inputs"]["members"]:
        raise ValueError("Ensemble member changed")
    if len(score_dirs) != len(model_dirs):
        raise ValueError("One prediction directory required per selected member")
    expected = sorted(selected["inputs"]["members"].values())
    metas = [verify(p,"prediction_manifest.json") for p in score_dirs]
    stage = verify_stage(work,split)
    if sorted(m["inputs"].get("model", "") for m in metas) != expected:
        raise ValueError("Prediction members do not match selected ensemble")
    if any(m["inputs"]["stage"] != stage or m["inputs"]["code"] != code_digest() for m in metas):
        raise ValueError("Prediction stage/source mismatch")
    threshold = json.loads((model_path(work,name)/"metrics.json").read_text(encoding="utf-8"))["threshold"]
    check_scores([], threshold)
    con = _connect(work,split,mem,threads)
    try:
        for i, directory in enumerate(score_dirs):
            con.execute(f"CREATE OR REPLACE TABLE sc AS SELECT sid,tid,p FROM read_parquet({lit(Path(directory)/'scores.parquet')})")
            validate_pairs(con,"sc")
            if con.execute("SELECT count(*) FROM sc WHERE p IS NULL OR NOT isfinite(p) OR p<0 OR p>1").fetchone()[0]:
                raise ValueError("Invalid ensemble member probabilities")
            if con.execute("SELECT count(*) FROM ((SELECT sid,tid FROM c EXCEPT SELECT sid,tid FROM sc) UNION ALL (SELECT sid,tid FROM sc EXCEPT SELECT sid,tid FROM c))").fetchone()[0]:
                raise ValueError("Member differs from the exact candidate set")
            if i == 0:
                con.execute(f"CREATE TABLE averaged AS SELECT sid,tid,p/{len(score_dirs)} p FROM sc")
            else:
                con.execute(f"UPDATE averaged a SET p=a.p+s.p/{len(score_dirs)} FROM sc s WHERE a.sid=s.sid AND a.tid=s.tid")
        out.mkdir(parents=True,exist_ok=False)
        con.execute(f"COPY averaged TO {lit(out/'scores.parquet')} (FORMAT parquet)")
        links, refs = write_outputs(con,out/"scores.parquet",threshold,out)
        seal(out,"prediction_manifest.json",{"code":code_digest(),"stage":stage,"ensemble_name":name,
             "ensemble":digest(model_path(work,name)/"ensemble_manifest.json"),"threshold":threshold,
             "references":refs,"matches":links},["scores.parquet","matching_results.tsv","candidate_pairs.tsv"])
    finally:
        con.close()


def main():
    """Parse explicit ensemble members or prediction directories; threshold comes from evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("eval","apply")); parser.add_argument("work")
    parser.add_argument("input", help="ground-truth parquet for eval; train/test split for apply")
    parser.add_argument("items", nargs="+", help="model subdirectories for eval; prediction directories for apply")
    parser.add_argument("--name",default="ensemble"); parser.add_argument("--out")
    parser.add_argument("--mem",default="4GB"); parser.add_argument("--threads",type=int,default=1)
    a=parser.parse_args()
    if a.mode == "eval":
        print(json.dumps(evaluate(a.work,a.input,a.items,a.name,a.mem,a.threads),indent=2))
    else:
        if not a.out:
            parser.error("apply needs --out")
        apply(a.work,a.input,a.out,a.items,a.name,a.mem,a.threads)


if __name__ == "__main__":
    main()
