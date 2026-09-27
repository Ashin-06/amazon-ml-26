"""Failure-oriented checks for xenc7 score alignment and validation gates."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('xenc7eval',Path(__file__).with_name('evaluate_xenc7.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_exact_join_is_order_independent_and_rejects_missing_extra_duplicates_nan():
    keys=pd.DataFrame({'sid':[1,2],'tid':[7,8]})
    xe=pd.DataFrame({'sid':[2,1],'tid':[8,7],'xe':[.2,.7]})
    assert np.array_equal(m.exact_logits(keys,xe),[.7,.2])
    bad=[xe.iloc[:1],pd.concat([xe,xe.iloc[:1]]),xe.assign(tid=[8,9]),xe.assign(xe=[np.nan,.7])]
    for frame in bad:
        with pytest.raises(ValueError):m.exact_logits(keys,frame)


def test_final_labels_must_be_masked_and_folds_group_references():
    d=pd.DataFrame({'sid':[1,1,2,3],'y':[1,0,1,-1]})
    refs=pd.DataFrame({'sid':[1,2]})
    folds=m.fold_assignments(d,refs)
    assert folds[0]==folds[1] and folds[-1]==-1
    with pytest.raises(ValueError):m.fold_assignments(d.assign(y=[1,0,1,0]),refs)
    with pytest.raises(ValueError):m.fold_assignments(d.assign(y=[1,0,-1,-1]),refs)


def test_gate_rejects_noise_and_country_regression():
    r={'delta':.0002,'paired_95_interval':[.00001,.00039],
       'by_country':{'india':{'delta':.0001},'us':{'delta':.00025}}}
    assert m.gate(r)
    assert not m.gate({**r,'delta':.000009})
    assert not m.gate({**r,'paired_95_interval':[-.00001,.00041]})
    assert not m.gate({**r,'by_country':{'india':{'delta':-.0001},'us':{'delta':.0004}}})


def test_all_references_and_multiple_targets_per_reference_are_preserved():
    base=m.base_module()
    # S1=1 has two true targets; S1=2 loses a competing wrong target; S1=3
    # is correctly empty; S1=4 has one wholly unretrieved true target.
    d=pd.DataFrame({'sid':[1,1,2],'tid':[7,8,7],'y':[1,1,0]})
    refs=pd.DataFrame({'sid':[1,2,3,4],'true_links':[2,0,0,1]})
    assert np.array_equal(base.winner_values(d,np.array([.99,.95,.9]),refs,.77),[1,1,1,0])


def test_incomplete_cloud_run_is_not_consumed(tmp_path):
    (tmp_path/'xenc7_audit.json').write_text('{"complete": false, "held_train_overlap": 0}')
    with pytest.raises(ValueError,match='incomplete'):m.verified_cloud(tmp_path)
