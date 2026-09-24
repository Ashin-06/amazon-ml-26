"""Full-reference inference with exact scored-candidate export and completion checks."""

from pathlib import Path
from importlib.metadata import version
from dataclasses import asdict
import json
import math
import time

import lightgbm as lgb
import numpy as np

from .blocking import CandidateIndex
from .config import BaselineConfig, config_digest
from .features import feature_schema, pair_features
from .storage import code_manifest, file_digest, staging_directory, write_json
from .validation import validate_files


def load_model(model_dir):
    """Check model/config/code/schema fingerprints before loading text-format weights."""
    model_dir = Path(model_dir).resolve(strict=True)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("pipeline_version") != 1:
        raise ValueError("Incomplete or unsupported model artifact")
    config = BaselineConfig(**manifest["config"]).validate()
    if config_digest(config) != manifest["config_sha256"]:
        raise ValueError("Saved configuration fingerprint mismatch")
    if list(feature_schema()) != manifest["feature_schema"]:
        raise ValueError("Saved feature schema differs from current code")
    if code_manifest() != manifest["code_sha256"]:
        raise ValueError("Source code changed since training; use matching code or retrain")
    for name, expected in manifest["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Dependency version changed since training: {name}; use the pinned environment")
    for name in ("model.txt", "config.json", "metrics.json", "LICENSE"):
        if file_digest(model_dir / name) != manifest["artifacts"].get(name):
            raise ValueError(f"Model artifact hash mismatch: {name}")
    if json.loads((model_dir / "config.json").read_text(encoding="utf-8")) != asdict(config):
        raise ValueError("Configuration file and model manifest disagree")
    threshold = manifest["threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError("Invalid saved decision threshold")
    if json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["threshold"] != threshold:
        raise ValueError("Threshold differs from the recorded tuning result")
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    if model.feature_name() != list(feature_schema()):
        raise ValueError("Booster feature names differ from the saved schema")
    return model, config, manifest


def predict_baseline(index_path, model_dir, output_dir):
    """Score all reference groups, validate both files, then publish the output directory.

    At most one reference's feature matrix is resident. This simple baseline does
    not resume a partial inference stage; failed unpublished stages are retained.
    """
    model, config, model_manifest = load_model(model_dir)
    stage = staging_directory(output_dir)
    started = time.monotonic()
    references = pairs = 0
    with CandidateIndex(index_path) as index:
        source_paths = {}
        for source in ("S1", "S2", "S3"):
            info = index.manifest["inputs"][source]
            path = Path(info["path"])
            if file_digest(path) != info["sha256"]:
                raise ValueError(f"Indexed input changed: {source}")
            source_paths[source] = path
        with (stage / "matching_results.tsv").open("w", encoding="utf-8", newline="") as matching, \
             (stage / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as candidates:
            matching.write("source1_entity_id\tmatched_entity_ids\n")
            candidates.write("source1_entity_id\tcandidate_entity_ids\n")
            for record in index.iter_references():
                found = index.retrieve(record, config.top_k, config.max_query_tokens)
                ids = [candidate.record.entity_id for candidate in found]
                # Export the exact final inference candidate set before scoring.
                candidates.write(record.entity_id + "\t" + ",".join(ids) + "\n")
                accepted = []
                if found:
                    matrix = np.asarray([pair_features(record, candidate) for candidate in found], dtype=np.float32)
                    scores = np.asarray(model.predict(matrix, num_threads=config.threads))
                    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
                        raise ValueError("Model returned malformed/nonfinite scores")
                    accepted = [target for target, score in zip(ids, scores) if score >= model_manifest["threshold"]]
                matching.write(record.entity_id + "\t" + ",".join(accepted) + "\n")
                references += 1
                pairs += len(ids)
        report = validate_files(source_paths["S1"], source_paths["S2"], source_paths["S3"],
                                stage / "matching_results.tsv", stage / "candidate_pairs.tsv", stage)
        for source, path in source_paths.items():
            if file_digest(path) != index.manifest["inputs"][source]["sha256"]:
                raise ValueError(f"Source changed during inference: {source}")
        manifest = {"complete": True, "pipeline_version": 1, "references": references,
                    "scored_candidates": pairs, "validation": report,
                    "input_index": index.manifest, "model_manifest_sha256": file_digest(Path(model_dir) / "manifest.json"),
                    "model_artifacts": model_manifest["artifacts"], "code_sha256": code_manifest(),
                    "elapsed_seconds": time.monotonic() - started,
                    "output_sha256": {name: file_digest(stage / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")}}
        write_json(stage / "manifest.json", manifest)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Output destination appeared during inference; complete stage was retained")
    stage.rename(destination)
    return manifest
