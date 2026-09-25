"""Candidate-contract, resource-control and all-reference efficiency regressions."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fast_er import model
from fast_er.efficiency import profile, policy_condition


def test_efficiency_includes_empty_references_and_singletons():
    with duckdb.connect(config={"threads":1}) as con:
        con.execute("CREATE TABLE refs AS SELECT * FROM (VALUES (0,'us'),(1,'us'),(2,'france')) t(sid,country)")
        con.execute("CREATE TABLE targets AS SELECT * FROM (VALUES ('us',10),('france',4)) t(country,n_targets)")
        con.execute("CREATE TABLE c AS SELECT * FROM (VALUES (0,10),(0,11)) t(sid,tid)")
        con.execute("CREATE TABLE gt AS SELECT * FROM (VALUES (0,10),(0,12),(1,13)) t(sid,tid)")
        r=profile(con)
        assert r['all']['references']==3
        assert r['all']['mean_candidates']==pytest.approx(2/3)
        assert r['all']['zero_candidate_references']==2
        assert r['all']['link_recall']==pytest.approx(1/3)
        assert r['all']['oracle_macro_f05']==pytest.approx((5/6+0+1)/3)
        assert r['all']['reduction_ratio']==pytest.approx(1-2/24)
        assert r['by_country']['france']['oracle_macro_f05']==1
        assert 'oracle_macro_f05' not in profile(con,labeled=False)['all']
        con.execute('INSERT INTO c VALUES (0,10)')
        with pytest.raises(ValueError,match='duplicate'):
            profile(con)


def test_policy_uses_blocking_ranks_only():
    assert 'p1' not in policy_condition(20,10)
    for values in [(0,0),(-1,4),(True,4),(2.5,1)]:
        with pytest.raises(ValueError):
            policy_condition(*values)


def test_invalid_stage1_scores_cannot_hide_behind_early_rejection(monkeypatch):
    keys=pd.DataFrame({'sid':[0],'tid':[10]})
    monkeypatch.setattr(model,'_pairs',lambda *a:keys)
    monkeypatch.setattr(model,'features',lambda *a:np.ones((1,len(model.FEATS))))
    monkeypatch.setattr(model,'_pred',lambda *a:np.array([np.nan]))
    with pytest.raises(ValueError):
        model._predict_stage1(None,[object()],'TRUE',1,1)


@pytest.mark.parametrize('probabilities',[[.001,.9,.002],[.001,.002,.003]])
def test_early_rejection_preserves_first_scoring_model_candidates(tmp_path,monkeypatch,probabilities):
    keys=pd.DataFrame({'sid':[0,1,2],'tid':[10,11,12]})
    def stage1(con,models,where,threads,chunks,tmp,tag):
        np.save(tmp/'x0.npy',np.ones((3,len(model.FEATS)),np.float32))
        return keys,np.array(probabilities)
    monkeypatch.setattr(model,'_predict_stage1',stage1)
    monkeypatch.setattr(model,'_context',lambda *a,**k:None)
    monkeypatch.setattr(model,'_ctx_matrix',lambda con,kk:np.zeros((len(kk),len(model.CTX)),np.float32))
    monkeypatch.setattr(model,'_pred',lambda m,X,threads:np.full(len(X),.95))
    actual,p1,p2=model._score_all(None,[],None,1,1,tmp_path)
    pd.testing.assert_frame_equal(actual,keys)
    np.testing.assert_array_equal(p1,probabilities)
    np.testing.assert_allclose(p2,np.where(p1>=model.CAND_P,.95,0))
    assert not list(tmp_path.glob('fe_*'))


def test_cpu_fallback_is_not_silently_moved_back_to_gpu(monkeypatch,tmp_path):
    import xgboost as xgb
    rng=np.random.default_rng(42)
    X=rng.random((100,4),dtype=np.float32); y=(X[:,0]>.5).astype(int)
    monkeypatch.setattr(model,'_GPU',True)
    monkeypatch.setattr(model,'_cupy',lambda:None)
    calls=[]
    def make(gpu):
        calls.append(gpu)
        if gpu:
            raise xgb.core.XGBoostError('GPU out of memory during matrix construction')
        return xgb.QuantileDMatrix(X,label=y,feature_names=list('abcd'),nthread=1)
    b=model._xgb_train(model._params(1,26,7,.1),make,len(y),2)
    assert calls==[True,False]
    assert model.model_device(b)=='cpu' and b.attr('cpu_retry')=='true'
    model._save(b,tmp_path/'cpu')
    loaded=model._load(tmp_path/'cpu')
    assert model.model_device(loaded)=='cpu'
    def forbidden_gpu():
        raise AssertionError('CPU model must not allocate GPU arrays')
    monkeypatch.setattr(model,'_cupy',forbidden_gpu)
    assert np.isfinite(model._pred(loaded,X,1)).all()


def test_gpu_retry_does_not_swallow_unrelated_errors(monkeypatch):
    monkeypatch.setattr(model,'_GPU',True)
    def bad(gpu):
        raise ValueError('Feature schema mismatch')
    with pytest.raises(ValueError,match='schema'):
        model._xgb_train(model._params(1,26),bad,10,1)


def test_tied_blocking_features_do_not_encode_target_id_order(tmp_path):
    from fast_er.block import block
    folder=tmp_path/'toy'; folder.mkdir()
    rows=[(0,1,'S1-1'),(1,2,'S2-1'),(2,2,'S2-2'),(3,3,'S3-1')]
    df=pd.DataFrame(rows,columns=['rid','src','entity_id']).assign(country='us',core='blue clinic',addr='9 lake rd')
    for src in (1,2,3):
        df[df.src==src].to_parquet(folder/f'norm_s{src}.parquet',index=False)
    path=block('toy',tmp_path,cap=10,k=3,k_field=3,chunks=1,threads=1,mem='512MB')
    pairs=pd.read_parquet(path)
    assert len(pairs)==3
    assert pairs[['r_s','r_n','r_a']].eq(1).all().all()
    assert set(pairs.route_r_s)=={1,2,3}
    assert not set(['route_r_s','route_r_n','route_r_a']) & set(model.FEATS)


def test_prediction_gpu_transfer_is_bounded_and_allocation_retry_is_exact(monkeypatch):
    class Booster:
        def __init__(self):
            self.device='cuda:0'
        def save_config(self):
            return '{"learner":{"generic_param":{"device":"'+self.device+'"}}}'
        def set_param(self,params):
            self.device=params['device']
        def inplace_predict(self,X):
            return X[:,0]
    class Pool:
        def free_all_blocks(self):
            pass
    class Arrays:
        def __init__(self):
            self.sizes=[]
        def asarray(self,X):
            self.sizes.append(len(X))
            if len(self.sizes)==2:
                raise MemoryError('test GPU allocation failure')
            return X
        def asnumpy(self,X):
            return X
        def get_default_memory_pool(self):
            return Pool()
    arrays=Arrays(); booster=Booster()
    monkeypatch.setattr(model,'_cupy',lambda:arrays)
    X=np.linspace(0,1,150000,dtype=np.float32).reshape(-1,1)
    np.testing.assert_array_equal(model._pred(booster,X,1),X[:,0])
    assert arrays.sizes==[65536,65536] and booster.device=='cpu'
    assert not model.is_memory_failure(ValueError('out of memory in invalid schema'))
