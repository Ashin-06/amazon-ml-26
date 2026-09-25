import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fast_er.diagnostics import error_rows, summarize_errors, reweight_known_countries
from fast_er.integrity import require_first_model_candidates


def test_error_accounting_separates_retrieval_matcher_and_singleton_false_merges():
    refs = ['retrieval','later','false','singleton','empty']
    truth = {'retrieval':{1,2},'later':{3,4},'false':{5},'singleton':set(),'empty':set()}
    pred = {'retrieval':{1},'later':{3},'false':{5,6},'singleton':{7},'competitor':{8}}
    hits = {'retrieval':1,'later':2,'false':1,'singleton':0,'empty':0}
    countries = dict.fromkeys(refs,'India')
    rows = error_rows(truth,pred,refs,hits,countries)
    for row in rows:
        assert row['retrieval_loss']+row['later_miss_loss']+row['false_positive_loss'] == pytest.approx(1-row['f05'])
    assert rows[0]['retrieval_loss'] > 0 and rows[0]['later_miss_loss'] == 0
    assert rows[1]['retrieval_loss'] == 0 and rows[1]['later_miss_loss'] > 0
    assert rows[2]['false_positive_loss'] == pytest.approx(4/9)
    assert rows[3]['false_positive_loss'] == 1 and rows[4]['f05'] == 1
    report = summarize_errors(rows)
    assert report['all']['references'] == 5
    assert report['all']['singleton_accuracy'] == .5
    weighted = reweight_known_countries(report,{'India':80,'France':20})
    assert weighted['covered_reference_fraction'] == .8
    assert weighted['unlabeled_country_counts'] == {'France':20}
    assert weighted['known_country_reweighted_f05'] == report['all']['macro_f05']


def test_error_accounting_rejects_missing_or_impossible_evidence():
    with pytest.raises(ValueError,match='Missing'):
        error_rows({}, {}, [1],{1:0},{1:'US'})
    with pytest.raises(ValueError,match='unique'):
        error_rows({1:set()}, {}, [1,1],{1:0},{1:'US'})
    with pytest.raises(ValueError,match='Inconsistent'):
        error_rows({1:{2}}, {1:{2}}, [1],{1:0},{1:'US'})
    with pytest.raises(ValueError,match='positive total'):
        reweight_known_countries({'by_country':{}},{'France':0})


def test_candidate_contract_rejects_learned_survivor_export():
    valid = {'inputs':{'candidate_scope':'input_to_first_scoring_model','pairs':95,'blocked_pairs':95}}
    require_first_model_candidates(valid)
    for bad in ({'pairs':4,'blocked_pairs':95}, {'candidate_scope':'stage2_survivors'}, {'pairs':True}):
        with pytest.raises(ValueError,match='first scoring model'):
            require_first_model_candidates({'inputs':dict(valid['inputs'],**bad)})
