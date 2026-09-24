"""Adversarial contracts and an offline training-to-archive integration check."""

from dataclasses import replace
from pathlib import Path
import csv
import json
import math
import subprocess
import sys
import zipfile

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from business_er.blocking import Candidate, CandidateIndex, build_index
from business_er.config import BaselineConfig, load_config
from business_er.features import feature_schema, pair_features
from business_er.io import Record, iter_id_lists
from business_er.packaging import build_package
from business_er.predict import load_model, predict_baseline
from business_er.storage import file_digest, write_json
from business_er.submission import write_id_lists
from business_er.train import predictions_at_threshold, select_threshold, train_baseline
from business_er.validation import score_files, validate_files


def source_file(path, rows):
    """Write only controlled synthetic records for integration tests."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        writer.writerows(rows)
    return path


def simple_files(tmp_path):
    """Create a two-reference fixture with a singleton and a new country."""
    s1 = source_file(tmp_path / "s1.tsv", [("S1-1", "École", "12 Paris", "France"), ("S1-2", "NA", "", "India")])
    s2 = source_file(tmp_path / "s2.tsv", [("S2-1", "Ecole", "12 Paris", "France")])
    s3 = source_file(tmp_path / "s3.tsv", [("S3-1", "École", "12 Paris", "France")])
    matching, candidates = tmp_path / "m.tsv", tmp_path / "c.tsv"
    write_id_lists(matching, [("S1-2", ()), ("S1-1", ("S2-1",))])
    write_id_lists(candidates, [("S1-1", ("S2-1", "S3-1")), ("S1-2", ())], "candidate_entity_ids")
    return s1, s2, s3, matching, candidates


def test_disk_scoring_reorders_and_rejects_duplicates(tmp_path):
    truth, prediction = tmp_path / "truth.tsv", tmp_path / "pred.tsv"
    write_id_lists(truth, [("S1-1", ("S2-1", "S3-1")), ("S1-2", ())])
    write_id_lists(prediction, [("S1-2", ()), ("S1-1", ("S2-1",))])
    result = score_files(truth, prediction, tmp_path)
    assert result["macro_f05"] == pytest.approx(11 / 12)
    assert result["reference_uniqueness_checked"]
    write_id_lists(prediction, [("S1-1", ()), ("S1-1", ())])
    with pytest.raises(ValueError, match="Duplicate"):
        score_files(truth, prediction, tmp_path)
    write_id_lists(prediction, [("S1-1", ())])
    with pytest.raises(ValueError, match="coverage"):
        score_files(truth, prediction, tmp_path)
    assert not list(tmp_path.glob("business-er-*.sqlite*"))


def test_disk_validation_all_checks_and_arbitrary_order(tmp_path):
    paths = simple_files(tmp_path)
    result = validate_files(*paths, work_dir=tmp_path)
    assert result["references"] == 2 and result["target_ids_checked"]
    assert result["references_by_country"] == {"France": 1, "India": 1}
    matching = paths[3]
    write_id_lists(matching, [("S1-1", ("S2-999",)), ("S1-2", ())])
    with pytest.raises(ValueError, match="Unknown target"):
        validate_files(*paths, work_dir=tmp_path)
    write_id_lists(matching, [("S1-1", ()), ("S1-1", ())])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_files(*paths, work_dir=tmp_path)
    write_id_lists(matching, [("S1-1", ("S2-1",)), ("S1-2", ())])
    write_id_lists(paths[4], [("S1-1", ()), ("S1-2", ())], "candidate_entity_ids")
    with pytest.raises(ValueError, match="outside candidate"):
        validate_files(*paths, work_dir=tmp_path)


def test_duplicate_source_ids_fail_validation(tmp_path):
    paths = simple_files(tmp_path)
    with paths[1].open("a", encoding="utf-8") as stream:
        stream.write("S2-1\tDuplicate\tRoad\tFrance\n")
    with pytest.raises(ValueError, match="Duplicate source"):
        validate_files(*paths, work_dir=tmp_path)


@pytest.mark.parametrize("settings", [
    {"top_k": 0}, {"top_k": True}, {"threads": -1}, {"seed": -1},
    {"learning_rate": float("nan")}, {"num_leaves": 1}, {"max_query_tokens": 101},
])
def test_invalid_config_rejected(settings):
    with pytest.raises(ValueError):
        replace(BaselineConfig(), **settings).validate()


def test_config_unknown_key_and_nonobject(tmp_path):
    path = tmp_path / "config.json"
    for data in ({"top_kk": 10}, [1, 2]):
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text('{"top_k": 1, "top_k": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate configuration"):
        load_config(path)


def test_retrieval_country_fallback_unicode_and_independent_address(tmp_path):
    s1 = source_file(tmp_path / "s1.tsv", [("S1-1", "Completely Renamed", "12 Rue École", "France")])
    s2 = source_file(tmp_path / "s2.tsv", [
        ("S2-1", "Original", "12 Rue Ecole", "France"),
        ("S2-2", "Completely Renamed", "12 Rue Ecole", "India"),
        ("S2-3", "Other", "12 Rue Ecole", ""),
        ("S2-4", "हाई फाइनेंस", "9 मार्ग", "India"),
    ])
    s3 = source_file(tmp_path / "s3.tsv", [("S3-1", "Completely Renamed", "Different place", "France")])
    database = tmp_path / "index.sqlite"
    build_index(s1, s2, s3, database)
    with CandidateIndex(database) as index:
        found = index.retrieve(next(index.iter_references()), 10)
        ids = {candidate.record.entity_id for candidate in found}
        assert ids == {"S2-1", "S2-3", "S3-1"}
        assert next(c for c in found if c.record.entity_id == "S2-1").address_rank > 0
        indic = index.retrieve(Record("S1-9", "हाई फाइनेंस", "", "India"))
        assert "S2-4" in {candidate.record.entity_id for candidate in indic}
        injection = index.retrieve(Record("S1-9", '" OR * NOT', "", "France"))
        assert isinstance(injection, list)
        blank = index.retrieve(Record("S1-9", "Completely Renamed", "", ""))
        assert "S2-2" in {candidate.record.entity_id for candidate in blank}
    with pytest.raises(FileExistsError):
        build_index(s1, s2, s3, database)


def test_features_schema_and_missingness():
    features = pair_features(Record("S1-1", "École", "", "France"), Candidate(Record("S2-1", "Ecole", "", "France"), 1, 0))
    values = dict(zip(feature_schema(), features))
    assert len(features) == len(feature_schema()) and np.isfinite(features).all()
    assert values["folded_name_ratio"] == 1
    assert values["address_exact"] == 0 and values["address_ratio"] == 0
    assert values["reference_address_missing"] == values["target_address_missing"] == 1


def test_threshold_preserves_singletons_and_absent_candidates():
    truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1"}}
    events, scores = [("S1-1", "S2-1"), ("S1-2", "S2-2")], np.asarray([.9, .2])
    threshold, score = select_threshold(truth, events, scores)
    assert score == pytest.approx(2 / 3)
    assert .2 < threshold <= .9
    assert predictions_at_threshold(truth, events, scores, threshold)["S1-3"] == []
    empty_threshold, _ = select_threshold({"S1-1": set()}, [("S1-1", "S2-1")], np.asarray([1.0]))
    assert empty_threshold > 1 and math.isfinite(empty_threshold)


def synthetic_dataset(root, split, size, offset=0):
    """Produce labeled local fixtures; these are never challenge performance data."""
    root.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, truth = [], [], [], []
    for i in range(1, size + 1):
        n = i + offset
        country = ("France" if split == "test" and i % 3 == 0 else "India" if i % 2 else "US")
        name, address = f"Cedar Business {n}", f"{1000+n} River Road Unit {n}"
        sid, a, b = f"S1-{n}", f"S2-{n*2}", f"S3-{n*2}"
        s1.append((sid, name, address, country))
        actual = () if i % 11 == 0 else (a, b)
        truth.append((sid, actual))
        if actual:
            s2.append((a, name.upper(), address, country))
            s3.append((b, name, address.replace("Road", "Rd"), country))
        s2.append((f"S2-{n*2+1}", name, f"{90000+n} Distant Avenue", country))
    paths = [source_file(root / f"{split}_source{number}.tsv", rows) for number, rows in enumerate((s1, s2, s3), 1)]
    truth_path = root / f"{split}_ground_truth.tsv"
    write_id_lists(truth_path, reversed(truth))
    return paths, truth_path


def test_training_inference_validation_packaging_and_tamper_detection(tmp_path):
    train_paths, truth = synthetic_dataset(tmp_path / "train", "train", 140)
    test_paths, _ = synthetic_dataset(tmp_path / "test", "test", 12, 1000)
    train_index, test_index = tmp_path / "train.sqlite", tmp_path / "test.sqlite"
    build_index(*train_paths, train_index)
    build_index(*test_paths, test_index)
    config = BaselineConfig(train_queries=70, tune_queries=20, holdout_queries=20,
                            max_pairs=20000, n_estimators=25, min_child_samples=2, top_k=3)
    model_dir = tmp_path / "model"
    trained = train_baseline(train_index, truth, model_dir, config)
    repeated = tmp_path / "model-repeat"
    train_baseline(train_index, truth, repeated, config)
    for artifact in ("model.txt", "config.json", "metrics.json"):
        assert file_digest(model_dir / artifact) == file_digest(repeated / artifact)
    assert 0 <= trained["metrics"]["groups"]["holdout"]["matching"]["macro_f05"] <= 1
    selected = trained["metrics"]["selected_reference_ids"]
    assert not (set(selected["train"]) & set(selected["tune"]) | set(selected["train"]) & set(selected["holdout"]) | set(selected["tune"]) & set(selected["holdout"]))
    output = tmp_path / "output"
    inferred = predict_baseline(test_index, model_dir, output)
    assert inferred["references"] == 12
    assert inferred["validation"]["references_by_country"]["France"] == 4
    assert inferred["scored_candidates"] == sum(len(ids) for _, ids in iter_id_lists(output / "candidate_pairs.tsv", "candidate_entity_ids"))
    expected = tmp_path / "input_manifest.json"
    write_json(expected, {str(path): {"bytes": path.stat().st_size, "sha256": file_digest(path)} for path in test_paths})
    documentation = tmp_path / "method.md"
    documentation.write_text("**Team Name:** [Required]\n**Team Members:** [Required]\n**Submission Date:** [Required]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Complete the methodology"):
        build_package(output, model_dir, documentation, expected, tmp_path / "invalid.zip")
    documentation.write_text("**Team Name:** Synthetic QA Only\n**Team Members:** Test Fixture\n**Submission Date:** 2026-09-25\n\nSynthetic integration test; not a challenge submission.\n", encoding="utf-8")
    archive_path = tmp_path / "qa.zip"
    build_package(output, model_dir, documentation, expected, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "output/matching_results.tsv" in names
        assert "code/business_entity_resolution/artifacts/model/model.txt" in names
        assert "code/business_entity_resolution/LICENSE" in names
        assert "code/business_entity_resolution/artifacts/model/LICENSE" in names
        assert not any(name.endswith("kaggle.json") or "KGAT" in name or "dataset/" in name for name in names)
        assert archive.testzip() is None
        extracted = tmp_path / "extracted"
        for name in names:
            assert (extracted / name).resolve().is_relative_to(extracted.resolve())
        archive.extractall(extracted)
    # Reproduce both outputs using the runnable source and model inside the ZIP.
    packaged = extracted / "code/business_entity_resolution"
    reproduced = tmp_path / "reproduced"
    process = subprocess.run([sys.executable, "-X", "utf8", str(packaged / "src/run.py"),
                              "predict", "--index", str(test_index), "--model-dir", str(packaged / "artifacts/model"),
                              "--output-dir", str(reproduced)], capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert file_digest(reproduced / name) == file_digest(output / name)
    official = SRC.parents[2] / "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py"
    # In the workspace, also compare with the organizer's unchanged validator.
    # A standalone final ZIP omits the original resource bundle and uses our
    # already-exercised strict validator for its self-contained test suite.
    if official.is_file():
        checked = subprocess.run([sys.executable, "-X", "utf8", str(official), "--matching", str(output / "matching_results.tsv"),
                                  "--candidate", str(output / "candidate_pairs.tsv"), "--test-dir", str(tmp_path / "test"), "--check-ids"],
                                 capture_output=True, text=True, encoding="utf-8")
        assert checked.returncode == 0, checked.stdout + checked.stderr
    model_file = model_dir / "model.txt"
    original = model_file.read_bytes()
    model_file.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(model_dir)
    model_file.write_bytes(original)
    matching_file = output / "matching_results.tsv"
    matching_file.write_bytes(matching_file.read_bytes() + b"S1-999\t\n")
    with pytest.raises(ValueError, match="changed after validation"):
        build_package(output, model_dir, documentation, expected, tmp_path / "tampered.zip")


def test_cli_reports_failure_without_traceback(tmp_path):
    truth = tmp_path / "truth.tsv"
    write_id_lists(truth, [("S1-1", ())])
    result = subprocess.run([sys.executable, "-X", "utf8", str(SRC / "run.py"), "score", "--truth", str(truth),
                             "--predictions", str(tmp_path / "absent.tsv")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 1 and "ERROR" in result.stderr and "Traceback" not in result.stderr
