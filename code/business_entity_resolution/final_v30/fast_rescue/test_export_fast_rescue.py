import importlib.util,io
from pathlib import Path
import pytest
p=Path(__file__).with_name('export_fast_rescue.py');s=importlib.util.spec_from_file_location('export',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_france_empty_and_multiple_targets_preserved():
 out=io.StringIO();n,links=m.write_rows([['a','x'],['b','y'],['c','']],[['a','old'],['b','french,new'],['c','']],out,{'b'},{'a':['z']})
 assert (n,links)==(3,4)
 assert out.getvalue().endswith('a\tx,z\nb\tfrench,new\nc\t\n')
def test_reference_mismatch_and_duplicate_new_link_rejected():
 with pytest.raises(ValueError):m.write_rows([['a','x']],[['b','x']],io.StringIO(),set(),{})
 with pytest.raises(ValueError):m.write_rows([['a','x']],[['a','x']],io.StringIO(),set(),{'a':['x']})
 with pytest.raises(ValueError):m.write_rows([['a','x']],[],io.StringIO(),set(),{})
