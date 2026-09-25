"""Regressions for validation leakage, malformed data and the actual fast pipeline."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fast_er import model
from fast_er.integrity import verify, verify_stage
from fast_er.prep import _chunk, clean_name
from fast_er.run_stage1 import run
from fast_er.to_parquet import convert
from fast_er.validation import load_truth


def records():
    con = duckdb.connect(config={"threads": 1})
    con.execute("""CREATE TABLE rec AS SELECT *, 'clinic' AS name, 'clinic' AS core, '' AS addr FROM (VALUES
      (0,1,'S1-1','us'), (1,1,'S1-2','us'), (2,1,'S1-3','fr'),
      (3,2,'S2-1','us'), (4,3,'S3-1','fr')) t(rid,src,entity_id,country)""")
    return con


@pytest.mark.parametrize("labels", [
    [('S1-1',''),('S1-1',''),('S1-3','')],
    [('S1-1',''),('S1-9',''),('S1-3','')],
    [('S1-1','S2-9'),('S1-2',''),('S1-3','')],
    [('S1-1','S2-1,S2-1'),('S1-2',''),('S1-3','')],
    [('S1-1','S2-1'),('S1-2','S2-1'),('S1-3','')],
    [('S1-1','S3-1'),('S1-2',''),('S1-3','')],
    [('S1-1','S2-1,'),('S1-2',''),('S1-3','')],
])
def test_bad_labels_fail_before_training(tmp_path, labels):
    path = tmp_path / 'gt.parquet'
    pd.DataFrame(labels, columns=['source1_entity_id','matched_entity_ids']).to_parquet(path)
    with records() as con, pytest.raises(ValueError):
        load_truth(con, path)


def test_empty_labels_and_france_are_preserved(tmp_path):
    path = tmp_path / 'gt.parquet'
    pd.DataFrame([('S1-1','S2-1'),('S1-2',''),('S1-3','S3-1')],
                 columns=['source1_entity_id','matched_entity_ids']).to_parquet(path)
    with records() as con:
        assert load_truth(con, path)['label_rows'] == 3


@pytest.mark.parametrize('sid,tid', [(0,99),(99,3),(3,0),(0,4)])
def test_writer_rejects_bad_endpoints(tmp_path, sid, tid):
    path = tmp_path / 'scores.parquet'
    pd.DataFrame({'sid':[sid], 'tid':[tid], 'p':[.9]}).to_parquet(path)
    with records() as con, pytest.raises(ValueError):
        model.write_outputs(con, path, .5, tmp_path)
    assert not (tmp_path / 'matching_results.tsv').exists()


def test_missing_scored_candidate_rejected(tmp_path):
    path = tmp_path / 'scores.parquet'
    pd.DataFrame({'sid':[0], 'tid':[3], 'p':[.9]}).to_parquet(path)
    with records() as con:
        con.execute('CREATE TABLE c AS SELECT 0 sid, 3 tid UNION ALL SELECT 1,3')
        with pytest.raises(ValueError, match='exactly'):
            model.write_outputs(con, path, .5, tmp_path)


def test_final_labels_used_only_once_after_threshold_choice(monkeypatch):
    original = model.macro_f05
    calls = []
    def observed(truth, pred, sids):
        calls.append(tuple(sids))
        return original(truth, pred, sids)
    monkeypatch.setattr(model, 'macro_f05', observed)
    # A competitor with stronger production probability must win target ownership.
    result = model.evaluate_threshold([0,1,2], [10,11,10], [.7,.8,.95],
                                     {0:{10},1:{11}}, [0], [1], grid=[.5,.9])
    assert calls == [(0,), (0,), (1,)]
    assert result['threshold'] == .9  # tied tune scores choose the conservative threshold
    assert 'grid_final' not in result
    changed = model.evaluate_threshold([0,1,2], [10,11,10], [.7,.8,.95],
                                      {0:{10},1:set()}, [0], [1], grid=[.5,.9])
    assert changed['threshold'] == result['threshold']


def test_context_zero_probability_and_tied_ranks():
    with records() as con:
        model._context(con, [0,1], [3,3], [0.,0.])
        context = model._ctx_matrix(con, pd.DataFrame({'sid':[1,0], 'tid':[3,3]}))
        assert np.isfinite(context).all()
        assert (context[:,model.CTX.index('p1_rel')] == 0).all()
        assert (context[:,model.CTX.index('p1_trank')] == 1).all()
        with pytest.raises(ValueError):
            model._ctx_matrix(con, pd.DataFrame({'sid':[2], 'tid':[4]}))


def test_original_alias_and_numeric_name_survive():
    values = _chunk((['S1-1'], ['First Brand DBA Second Brand'], [''], ['France']))
    assert 'firstbrand' in values[-1][0] and 'secondbrand' in values[-1][0]
    assert clean_name('12345678') == '12345678'


def write_toy_data(root, n=120):
    for split in ('train','test'):
        directory = root / split
        directory.mkdir(parents=True)
        for src in (1,2,3):
            with (directory / f'{split}_source{src}.tsv').open('w',encoding='utf-8',newline='') as stream:
                writer = csv.writer(stream,delimiter='\t',lineterminator='\n')
                writer.writerow(['entity_id','business_name','business_address','country'])
                for i in range(n):
                    key = i if split == 'train' else i+1000
                    country = 'France' if split == 'test' and i%4 == 0 else 'US'
                    name = f'Clinic echo {i}' if src != 3 else f'Shop alpha {i}'
                    addr = f'{100+i} Lake Street' if src != 3 else f'{500+i} Lake Street'
                    if src == 2 and i%7 == 0:
                        addr = ''
                    writer.writerow([f'S{src}-{key}',name,addr,country])
        if split == 'train':
            with (directory / 'train_ground_truth.tsv').open('w',encoding='utf-8',newline='') as stream:
                writer = csv.writer(stream,delimiter='\t',lineterminator='\n')
                writer.writerow(['source1_entity_id','matched_entity_ids'])
                writer.writerows((f'S1-{i}', '' if i%10 == 0 else f'S2-{i}') for i in range(n))


def args(pq, work, split):
    return argparse.Namespace(split=split, pq_dir=pq, work=work,
        gt=pq/'train_ground_truth.parquet' if split=='train' else None,
        mem='512MB', threads=1, workers=1, chunks=7, cap=500, cap_whole=1000,
        k=8, k_field=5, hold_mod=4)


@pytest.fixture(scope='module')
def completed_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("fast_pipeline")
    data, pq, work = root/'data', root/'pq', root/'work'
    write_toy_data(data)
    convert(data, pq)
    run(args(pq, work, 'train'))
    run(args(pq, work, 'test'))
    model.train(work, pq/'train_ground_truth.parquet', n_train=70, hold_mod=4, backend='lgb',
                mem='512MB', threads=1, rounds=3, chunks=7, leaves=7)
    model.predict(work, 'test', root/'out', mem='512MB', threads=1, chunks=7)
    return root


def test_fast_raw_to_model_to_validated_outputs(completed_pipeline):
    root = completed_pipeline
    meta = verify(root/'out', 'prediction_manifest.json')
    assert meta['inputs']['references'] == 120
    assert meta['inputs']['pairs'] > 120
    gt = pd.read_parquet(root/'pq/train_ground_truth.parquet')
    assert gt.matched_entity_ids.eq('').sum() == 12
    from fast_er.prep import skeleton
    prepared = pd.read_parquet(root/'work/train/norm_s2.parquet')
    assert prepared.skel.tolist() == prepared.core.map(skeleton).tolist()
    from business_er.validation import validate_files
    result = validate_files(*(root/'data/test'/f'test_source{i}.tsv' for i in (1,2,3)),
        root/'out/matching_results.tsv', root/'out/candidate_pairs.tsv', root/'checks')
    assert result['references'] == 120
    split = pd.read_parquet(root/'work/model/split_ids.parquet')
    assert split.entity_id.is_unique
    assert {'tune','final'} <= set(split.half)
    metrics = json.loads((root/'work/model/metrics.json').read_text())
    assert 'grid_final' not in metrics['stage2']
    assert metrics['references_fit']['stage2'] <= 70
    assert metrics['params']['stage2_all'] is False
    assert set(metrics['params']['devices'].values()) == {'cpu'}
    error_rows = pd.read_parquet(root/'work/model/hold_error_rows.parquet')
    decisions = pd.read_parquet(root/'work/model/hold_decisions.parquet')
    assert len(error_rows) == len(decisions) == metrics['holdout_s1']
    for half in ('tune','final'):
        summary = metrics['error_diagnostics'][half]['all']
        assert summary['macro_f05'] == pytest.approx(metrics['stage2'][half+'_macro_f05'])
        assert summary['retrieval_loss']+summary['later_miss_loss']+summary['false_positive_loss'] == pytest.approx(1-summary['macro_f05'])
    model.predict(root/'work','test',root/'repeat',mem='512MB',threads=1,chunks=13)
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        assert (root/'out'/name).read_bytes() == (root/'repeat'/name).read_bytes()


def test_cached_training_scores_equal_full_refeaturization(completed_pipeline):
    """train() reuses cached stage-1 matrices; production scoring from scratch must give the same scores."""
    root = completed_pipeline
    m1, m2, _ = model.load_models(root/'work/model')
    con = model._connect(root/'work', 'train', '512MB', 1)
    try:
        keys, p1, p2 = model._score_all(con, m1, m2, 1, 7, root/'tmp_rescore')
    finally:
        con.close()
    fresh = keys.assign(p1=p1, p2=p2)
    hold = pd.read_parquet(root/'work/model/hold_preds.parquet').merge(fresh, on=['sid', 'tid'])
    assert len(hold) == len(pd.read_parquet(root/'work/model/hold_preds.parquet')) > 0
    assert np.allclose(hold.p_stage1, hold.p1, atol=1e-6) and np.allclose(hold.p_stage2, hold.p2, atol=1e-6)
    threshold = json.loads((root/'work/model/metrics.json').read_text())['threshold']
    expected = model.decide(keys.sid, keys.tid, p2, threshold)
    saved = pd.read_parquet(root/'work/model/hold_decisions.parquet')
    assert all(set(r.matched_tids) == expected.get(r.sid,set()) for r in saved.itertuples())
    assert not list((root/'work').glob('fe_train_*'))  # scratch removed


def test_cascade_exports_first_model_inputs_even_when_later_rejected(completed_pipeline, monkeypatch):
    """A higher early-rejection threshold must not shrink the official exported candidates."""
    root = completed_pipeline
    full = pd.read_parquet(root/'out/scores.parquet')
    monkeypatch.setattr(model, "CAND_P", float(full.p1.median()))
    model.predict(root/'work', 'test', root/'filtered', mem='512MB', threads=1, chunks=7)
    scores = pd.read_parquet(root/'filtered/scores.parquet')
    assert len(scores) == len(full)
    assert (scores.p[scores.p1 < model.CAND_P] == 0).all()
    assert (root/'filtered/candidate_pairs.tsv').read_bytes() == (root/'out/candidate_pairs.tsv').read_bytes()
    read = lambda n: pd.read_csv(root/'filtered'/n, sep='\t', dtype=str, keep_default_na=False)
    cand, match = read('candidate_pairs.tsv'), read('matching_results.tsv')
    pairs = {(s, t) for s, ids in zip(cand.source1_entity_id, cand.candidate_entity_ids) for t in ids.split(',') if ids}
    assert len(pairs) == len(scores)
    for s, ids in zip(match.source1_entity_id, match.matched_entity_ids):
        assert all((s, t) in pairs for t in ids.split(',') if ids)


def test_completed_cache_reuse_requires_exact_inputs(completed_pipeline):
    root = completed_pipeline
    settings = args(root/'pq', root/'work','train')
    assert run(settings)['complete']
    settings.chunks += 1
    with pytest.raises(ValueError, match='fingerprint'):
        run(settings)


def test_tampered_model_and_output_cannot_be_reused(completed_pipeline, tmp_path):
    shutil.copytree(completed_pipeline/'work/model', tmp_path/'model')
    path = tmp_path/'model/model_2.txt'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='fingerprint'):
        model.load_models(tmp_path/'model')
    shutil.copytree(completed_pipeline/'out', tmp_path/'out')
    path = tmp_path/'out/matching_results.tsv'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='fingerprint'):
        verify(tmp_path/'out', 'prediction_manifest.json')


def test_existing_predictions_are_never_overwritten(completed_pipeline):
    with pytest.raises(FileExistsError):
        model.predict(completed_pipeline/'work','test',completed_pipeline/'out')

def test_gpu_probe_rejects_silent_cpu_fallback(monkeypatch):
    import xgboost
    class CpuFallback:
        def save_config(self):
            return '{"learner":{"generic_param":{"device":"cpu"}}}'
    monkeypatch.setattr(model, "_GPU", None)
    monkeypatch.setattr(xgboost, "train", lambda *a, **k: CpuFallback())
    assert model.gpu_available() is False


def test_reference_weights_and_cpu_xgboost_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_GPU", False)
    weights = model.pair_weights(np.array([1,1,1,2]), "ref")
    assert weights[:3].sum() == pytest.approx(weights[3])
    with pytest.raises(ValueError):
        model.pair_weights(np.array([1]), "typo")
    X = np.random.default_rng(26).random((256,4), dtype=np.float32)
    y = (X[:,0] > .5).astype(int)
    booster = model._fit("xgb", model._params(1,26,7,.1), X, y, None, ["a","b","c","d"], 2)
    model._save(booster, tmp_path/"booster")
    restored = model._load(tmp_path/"booster")
    np.testing.assert_allclose(model._pred(booster,X,1),model._pred(restored,X,1))
    assert model._names(restored) == ["a","b","c","d"]


def test_streamed_xgboost_matches_in_memory(monkeypatch):
    monkeypatch.setattr(model, "_GPU", False)
    rng = np.random.default_rng(3)
    X = rng.random((3000, 4), dtype=np.float32)
    y = (X[:, 0] + .2 * X[:, 1] > .6).astype(np.int8)
    w = model.pair_weights(rng.integers(0, 500, 3000), "ref")
    P = model._params(1, 26, 7, .1)
    whole = model._fit("xgb", P, X, y, w, list("abcd"), 5)
    blocks = [(lambda a=a: X[a:a + 700]) for a in range(0, 3000, 700)]
    streamed = model._fit_parts("xgb", P, blocks, y, w, list("abcd"), 5)
    np.testing.assert_allclose(model._pred(whole, X, 1), model._pred(streamed, X, 1), atol=1e-6)


def test_verified_archive_rejects_drafts_and_wrong_originals(completed_pipeline, tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "kaggle/package_submission.py"
    spec = importlib.util.spec_from_file_location("fast_packager", path)
    pkg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pkg)
    root = completed_pipeline
    doc = tmp_path/"method.md"
    doc.write_text("<!-- DRAFT: pending -->",encoding="utf-8")
    with pytest.raises(ValueError, match="draft"):
        pkg.check_document(doc)
    # Synthetic fixture identities only; never used for the real team's methodology.
    doc.write_text("**Team Name:** QA Fixture\n**Team Members:** Synthetic Test\n**Submission Date:** 2026-09-25\n",encoding="utf-8")
    expected = json.loads((root/"pq/conversion_manifest.json").read_text())["inputs"]["files"]
    manifest = tmp_path/"inputs.json"
    manifest.write_text(json.dumps(expected),encoding="utf-8")
    archive = tmp_path/"qa.zip"
    result = pkg.build_package(root/"out",root/"work",doc,manifest,archive)
    assert result["files"] > 10 and archive.is_file()
    wrong_scope = tmp_path/'wrong_scope'
    shutil.copytree(root/'out', wrong_scope)
    meta_path = wrong_scope/'prediction_manifest.json'
    invalid = json.loads(meta_path.read_text())
    invalid['inputs']['candidate_scope'] = 'stage2_survivors'
    meta_path.write_text(json.dumps(invalid),encoding='utf-8')
    with pytest.raises(ValueError,match='first scoring model'):
        pkg.build_package(wrong_scope,root/'work',doc,manifest,tmp_path/'wrong_scope.zip')
    expected["test_source1.tsv"]["sha256"] = "0"*64
    manifest.write_text(json.dumps(expected),encoding="utf-8")
    with pytest.raises(ValueError, match="audited"):
        pkg.build_package(root/"out",root/"work",doc,manifest,tmp_path/"bad.zip")


def test_offline_kernel_has_exact_version_preflight(tmp_path):
    import ast
    import subprocess
    pkg = Path(__file__).resolve().parents[1]
    out = tmp_path/"kernel"
    result = subprocess.run([sys.executable,str(pkg/"kaggle/build_kernel.py"),str(out),
        "qa-owner","qa-kernel","block_train,block_test,model","--no-internet"],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    body = (out/"script.py").read_text(encoding="utf-8")
    ast.parse(body)
    assert '"--no-index"' in body and "check=False" not in body
    assert json.loads((out/"kernel-metadata.json").read_text())["enable_internet"] is False
    # Emitted Python can be interpreted up to setup without executing imports, pip or jobs.
    tree = ast.parse(body)
    values = {node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
              if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
              and node.targets[0].id in ("FILES","REQUIREMENTS","OFFLINE")}
    assert values["OFFLINE"] is True
    assert "xgboost==3.0.0" in values["REQUIREMENTS"]
    assert values["FILES"]["__init__.py"] == (pkg/"src/fast_er/__init__.py").read_text(encoding="utf-8")

def test_french_alias_and_non_alias_prefix():
    assert clean_name("New Shop née Old Shop") == "Old Shop"
    assert clean_name("New Shop anciennement Old Shop") == "Old Shop"
    assert clean_name("Nee Services") == "Nee Services"


def test_partial_and_changed_input_cache_rejected(completed_pipeline, tmp_path):
    work = tmp_path/"partial"
    (work/"train").mkdir(parents=True)
    with pytest.raises(ValueError, match="manifest"):
        run(args(completed_pipeline/"pq",work,"train"))
    copied = tmp_path/"pq"
    shutil.copytree(completed_pipeline/"pq",copied)
    path = copied/"train_source1.parquet"
    df = pd.read_parquet(path)
    df.loc[0,"business_name"] = "Changed input"
    df.to_parquet(path,index=False)
    with pytest.raises(ValueError, match="fingerprint"):
        run(args(copied,completed_pipeline/"work","train"))


def test_empty_target_source_is_supported(tmp_path):
    from fast_er.prep import prepare
    data, pq = tmp_path/"data", tmp_path/"pq"
    write_toy_data(data, n=4)
    path = data/"test/test_source3.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n",encoding="utf-8")
    convert(data,pq)
    result = prepare("test",pq,tmp_path/"work",workers=1)
    assert len(pd.read_parquet(result/"norm_s3.parquet")) == 0

def test_ensemble_duplicate_compensation_is_rejected():
    from fast_er.ensemble import align_scores
    # Earlier UNION counts could accept these compensating duplicates.
    bad = pd.DataFrame({"sid":[0,0],"tid":[3,3],"p":[.8,.9]})
    with pytest.raises(ValueError, match="Duplicate"):
        align_scores(None,bad)


def test_verified_seed_ensemble_end_to_end(completed_pipeline):
    from fast_er.ensemble import evaluate, apply
    root=completed_pipeline
    model.train(root/"work",root/"pq/train_ground_truth.parquet",n_train=70,hold_mod=4,
        mem="512MB",threads=1,rounds=3,chunks=7,leaves=7,seed=27,model_dir="model_s27")
    model.predict(root/"work","test",root/"out_s27",mem="512MB",threads=1,chunks=7,model_dir="model_s27")
    a=pd.read_parquet(root/"work/model/split_ids.parquet")
    b=pd.read_parquet(root/"work/model_s27/split_ids.parquet")
    pd.testing.assert_frame_equal(a[a.g=="hold"].sort_values("entity_id").reset_index(drop=True),
                                  b[b.g=="hold"].sort_values("entity_id").reset_index(drop=True))
    result=evaluate(root/"work",root/"pq/train_ground_truth.parquet",["model","model_s27"],
                    mem="512MB",threads=1,chunks=7)
    assert "grid_final" not in result
    apply(root/"work","test",root/"ensemble_out",[root/"out",root/"out_s27"],mem="512MB",threads=1)
    from business_er.validation import validate_files
    checked=validate_files(*(root/"data/test"/f"test_source{i}.tsv" for i in (1,2,3)),
        root/"ensemble_out/matching_results.tsv",root/"ensemble_out/candidate_pairs.tsv",root/"ensemble_checks")
    assert checked["references"]==120
    verify(root/"ensemble_out","prediction_manifest.json")
    import importlib.util
    spec=importlib.util.spec_from_file_location("ensemble_packager",Path(__file__).resolve().parents[1]/"kaggle/package_submission.py")
    pkg=importlib.util.module_from_spec(spec);spec.loader.exec_module(pkg)
    doc=root/"ensemble_method.md"
    doc.write_text("**Team Name:** QA Fixture\n**Team Members:** Synthetic Test\n**Submission Date:** 2026-09-25\n",encoding="utf-8")
    original=root/"ensemble_originals.json"
    original.write_text(json.dumps(json.loads((root/"pq/conversion_manifest.json").read_text())["inputs"]["files"]),encoding="utf-8")
    pkg.build_package(root/"ensemble_out",root/"work",doc,original,root/"ensemble_qa.zip")
    with pytest.raises(ValueError,match="members"):
        apply(root/"work","test",root/"wrong_members",[root/"out",root/"out"],mem="512MB",threads=1)


def test_model_directory_escape_rejected(tmp_path):
    for name in ("../other","/absolute","x/y",""):
        with pytest.raises(ValueError):
            model.model_path(tmp_path,name)

def test_unmanifested_normalized_file_cannot_change_prediction(completed_pipeline):
    root=completed_pipeline
    extra=root/"work/test/norm_s_extra.parquet"
    extra.write_bytes((root/"work/test/norm_s2.parquet").read_bytes())
    try:
        model.predict(root/"work","test",root/"extra_file_check",mem="512MB",threads=1,chunks=7)
        assert (root/"extra_file_check/matching_results.tsv").read_bytes()==(root/"out/matching_results.tsv").read_bytes()
    finally:
        extra.unlink()


def test_stage1_fingerprint_covers_exactly_the_code_stage1_runs():
    """Stage-1 outputs are bound to the modules candidate generation executes; none of them may import the
    matcher (model.py), so matcher changes cannot silently alter candidates without a rebuild."""
    import ast
    from fast_er import integrity
    src = Path(integrity.__file__).parent
    for name in integrity.STAGE1_CODE:
        tree = ast.parse((src / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                   [a.name for a in node.names] if isinstance(node, ast.Import) else []
            local = [m.split(".")[-1] for m in mods if m.startswith("fast_er") or (isinstance(node, ast.ImportFrom) and node.level)]
            assert all(m + ".py" in integrity.STAGE1_CODE or m in ("", "fast_er") for m in local), (name, mods)
    assert integrity.code_digest(integrity.STAGE1_CODE) != integrity.code_digest()
    assert "model.py" not in integrity.STAGE1_CODE and "run_stage1.py" in integrity.STAGE1_CODE



def test_stage1_negative_sampling_keeps_positives_and_hard_candidates(completed_pipeline):
    root = completed_pipeline
    m = model.train(root/'work', root/'pq/train_ground_truth.parquet', n_train=70, hold_mod=4, backend='lgb',
                    mem='512MB', threads=1, rounds=3, chunks=7, leaves=7, model_dir='model_s1sample',
                    stage1_hard_rank=1, stage1_neg_keep=0.5)
    full = json.loads((root/'work/model/metrics.json').read_text())
    assert m["params"]["stage1_neg_keep"] == 0.5 and m["params"]["stage1_hard_rank"] == 1
    assert 0 < m["stage1_training_rows"]["A"] and m["holdout_s1"] == full["holdout_s1"]
    with pytest.raises(ValueError):
        model.train(root/'work', root/'pq/train_ground_truth.parquet', n_train=70, hold_mod=4, backend='lgb',
                    mem='512MB', threads=1, rounds=3, chunks=7, leaves=7, model_dir='model_bad', stage1_neg_keep=0)
