"""Regression checks for invalid labels, partial artifacts and empty sources."""

from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from business_er.blocking import CandidateIndex, build_index
from business_er.config import BaselineConfig
from business_er.io import iter_id_lists
from business_er.submission import write_id_lists
from business_er.train import train_baseline
from test_baseline import source_file, synthetic_dataset


@pytest.mark.parametrize("defect", ["missing_reference", "unknown_reference", "unknown_target", "shared_target"])
def test_invalid_truth_fails_before_publishing_model(tmp_path, defect):
    paths, truth_path = synthetic_dataset(tmp_path / "train", "train", 50)
    index = tmp_path / "index.sqlite"
    build_index(*paths, index)
    truth = list(iter_id_lists(truth_path))
    if defect == "missing_reference":
        truth.pop()
    elif defect == "unknown_reference":
        truth.append(("S1-999999", ()))
    elif defect == "unknown_target":
        truth[0] = (truth[0][0], ("S2-999999",))
    else:
        truth[0] = (truth[0][0], ("S2-2",))
    write_id_lists(truth_path, truth)
    model = tmp_path / "model"
    with pytest.raises(ValueError):
        train_baseline(index, truth_path, model, BaselineConfig())
    assert not model.exists()


def test_duplicate_source_never_publishes_partial_index(tmp_path):
    one = source_file(tmp_path / "s1.tsv", [("S1-1", "A", "B", "US"), ("S1-1", "A", "B", "US")])
    two = source_file(tmp_path / "s2.tsv", [("S2-1", "A", "B", "US")])
    three = source_file(tmp_path / "s3.tsv", [])
    index = tmp_path / "index.sqlite"
    with pytest.raises(ValueError, match="Duplicate"):
        build_index(one, two, three, index)
    assert not index.exists()
    assert not list(tmp_path.glob("index.sqlite.building-*"))


def test_empty_target_source_is_counted_and_other_source_works(tmp_path):
    one = source_file(tmp_path / "s1.tsv", [("S1-1", "École", "12 Paris", "France")])
    two = source_file(tmp_path / "s2.tsv", [])
    three = source_file(tmp_path / "s3.tsv", [("S3-1", "Ecole", "12 Paris", "France")])
    index = tmp_path / "index.sqlite"
    result = build_index(one, two, three, index)
    assert result["counts"] == {"S1": 1, "S2": 0, "S3": 1}
    with CandidateIndex(index) as opened:
        found = opened.retrieve(next(opened.iter_references()))
        assert [item.record.entity_id for item in found] == ["S3-1"]
