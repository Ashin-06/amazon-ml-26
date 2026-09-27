"""Check augmentation invariants before spending GPU resources."""
import importlib.util,random
from pathlib import Path

spec=importlib.util.spec_from_file_location('xenc7',Path(__file__).parent/'kaggle_xenc7/script.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)

def test_address_is_never_rewritten():
    for seed in range(100):
        z=x.perturb('Example LLC | 20 Group Road, Inc Township',random.Random(seed))
        assert z.partition(' | ')[2]=='20 Group Road, Inc Township'

def test_negative_collapse_is_reverted():
    class R:
        def random(self):return 0.
        def choice(self,options):return options[0]
    a,b='Example SARL | 1 Road','Example LLC | 1 Road'
    assert x.augment_pair(a,b,0,R())==(a,b)
    aa,bb=x.augment_pair(a,b,1,R())
    assert aa.casefold()==bb.casefold()==a.casefold()

def test_seeded_augmentation_reproduces_and_keeps_empty_address():
    a='Example LLC | ';b='Example Inc | '
    assert x.augment_pair(a,b,1,random.Random(4))==x.augment_pair(a,b,1,random.Random(4))
    for seed in range(30):
        aa,bb=x.augment_pair(a,b,1,random.Random(seed))
        assert aa.endswith(' | ') and bb.endswith(' | ')
