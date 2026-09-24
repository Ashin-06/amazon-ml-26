"""Verify score semantics and data contracts that could invalidate a submission."""

from pathlib import Path
import subprocess
import sys
import unicodedata

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from business_er.io import Record, iter_id_lists, iter_records, parse_id_list
from business_er.metrics import candidate_metrics, entity_f05, macro_f05, score_aligned_rows
from business_er.normalize import fold_latin_accents, normalize_text, numeric_tokens, token_jaccard
from business_er.splits import assign_split
from business_er.submission import validate_aligned_outputs, write_id_lists


@pytest.mark.parametrize("truth,prediction,expected", [
    ([], [], 1), ([], ["a"], 0), (["a"], [], 0),
    (["a", "b"], ["a", "b"], 1),
    (["a", "b"], ["a", "b", "c"], 5 / 7),
    (["a", "b"], ["a"], 5 / 6), (["a"], ["b"], 0),
])
def test_entity_metric(truth, prediction, expected):
    assert entity_f05(truth, prediction) == pytest.approx(expected)


def test_macro_weights_entities_and_singletons_equally():
    truth = {"S1-1": ["a", "b"], "S1-2": []}
    prediction = {"S1-1": ["a"], "S1-2": []}
    assert macro_f05(truth, prediction) == pytest.approx(11 / 12)
    result = score_aligned_rows(truth.items(), prediction.items())
    assert result["macro_f05"] == pytest.approx(11 / 12)
    assert result["singleton_accuracy"] == 1
    assert result["false_negative_links"] == 1


def test_metric_requires_complete_universe():
    with pytest.raises(ValueError):
        macro_f05({"S1-1": []}, {})
    with pytest.raises(ValueError):
        macro_f05({}, {})
    with pytest.raises(ValueError):
        score_aligned_rows([("S1-1", [])], [])
    with pytest.raises(ValueError):
        score_aligned_rows([("S1-1", [])], [("S1-2", [])])


def test_candidate_ceiling_keeps_unretrieved_positives_and_singletons():
    result = candidate_metrics(
        {"S1-1": ["a", "b"], "S1-2": []},
        {"S1-1": ["a", "wrong"], "S1-2": ["wrong"]},
    )
    assert result["link_recall"] == .5
    assert result["oracle_macro_f05"] == pytest.approx(11 / 12)
    assert result["mean_candidates"] == 1.5
    assert candidate_metrics({"S1-1": []}, {"S1-1": []})["link_recall"] is None


@pytest.mark.parametrize("bad", ["S1-1", "S2-1,S2-1", "S2-1,", " S2-1", "S2-1, S3-2", "nan", "[]", "S2-१"])
def test_invalid_id_lists(bad):
    with pytest.raises(ValueError):
        parse_id_list(bad)


def test_reader_preserves_empty_text_commas_and_open_countries(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text(
        'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
        'S1-01\tÉcole\t12 Rue, Paris\tFrance\n'
        'S1-02\tNA\t\tFutureCountry\n', encoding="utf-8"
    )
    records = list(iter_records(path, "S1"))
    assert records[0].entity_id == "S1-01"
    assert records[0].business_address == "12 Rue, Paris"
    assert records[1].business_name == "NA"
    assert records[1].business_address == ""
    assert records[1].country == "FutureCountry"
    with pytest.raises(ValueError):
        list(iter_records(path, "S2"))


@pytest.mark.parametrize("content", [
    "entity_id,business_name,business_address,country\n",
    "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tName\tAddress\n",
    "entity_id\tbusiness_name\tbusiness_address\tcountry\n\n",
])
def test_reader_rejects_wrong_delimiter_width_and_blank_rows(tmp_path, content):
    path = tmp_path / "bad.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_records(path))


def test_unicode_views_preserve_indic_marks():
    for text in ["हाई फाइनेंस", "সিটি ইনভেস্টমেন্টস প্রাইভেট লিমিটেড"]:
        assert fold_latin_accents(text) == normalize_text(text)
        assert any(unicodedata.category(c).startswith("M") for c in normalize_text(text))
    assert normalize_text("  ÉCOLE & Fils, SARL  ") == "école fils sarl"
    assert fold_latin_accents("ÉCOLE à Paris") == "ecole a paris"
    assert normalize_text("१२ Road") == "१२ road"
    assert numeric_tokens("No 28/2D, Unit B202") == {"28", "2d", "b202"}
    assert token_jaccard("", "") == 0
    assert token_jaccard("A B", "B C") == pytest.approx(1 / 3)


def test_split_stable_across_processes_and_balanced():
    ids = [f"S1-{i}" for i in range(10000)]
    splits = [assign_split(sid) for sid in ids]
    assert .77 < splits.count("train") / len(ids) < .83
    assert .08 < splits.count("tune") / len(ids) < .12
    command = "from business_er.splits import assign_split; print(assign_split('S1-123'))"
    result = subprocess.run([sys.executable, "-c", command], cwd=SRC, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == assign_split("S1-123")
    with pytest.raises(ValueError):
        assign_split("S1-1", train_fraction=.9, tune_fraction=.2)
    with pytest.raises(ValueError):
        assign_split("S1-1", train_fraction=float("nan"))


def test_writer_roundtrip_multiple_matches_and_empty(tmp_path):
    rows = [("S1-1", ("S2-1", "S2-2", "S3-1")), ("S1-2", ())]
    path = tmp_path / "matching.tsv"
    assert write_id_lists(path, rows) == 2
    assert list(iter_id_lists(path)) == rows
    assert path.read_bytes().endswith(b"S1-2\t\n")


@pytest.mark.parametrize("invalid", [("S2-1", "S2-1"), ("S2-1,S3-1",), ("S1-1",)])
def test_failed_write_keeps_existing_file(tmp_path, invalid):
    path = tmp_path / "matching.tsv"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        write_id_lists(path, [("S1-1", invalid)])
    assert path.read_text() == "existing"
    assert list(tmp_path.glob("*.tmp")) == []


def test_output_checks_strict_containment_and_target_existence():
    references = [Record("S1-1", "A", "B", "France")]
    matching = [("S1-1", ("S2-1",))]
    candidates = [("S1-1", ("S2-1", "S3-1"))]
    result = validate_aligned_outputs(references, matching, candidates, {"S2-1", "S3-1"})
    assert result["target_ids_checked"]
    assert not validate_aligned_outputs(references, matching, candidates)["target_ids_checked"]
    with pytest.raises(ValueError):
        validate_aligned_outputs(references, matching, [("S1-1", ())])
    with pytest.raises(ValueError):
        validate_aligned_outputs(references, matching, candidates, {"S2-1"})
    with pytest.raises(ValueError):
        validate_aligned_outputs(references, matching, [])
    with pytest.raises(ValueError):
        validate_aligned_outputs(references * 2, matching * 2, candidates * 2)
