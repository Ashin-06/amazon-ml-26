"""Bounded CPU baseline training with entity-separated tuning and holdout."""

from array import array
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import heapq
import math
import sqlite3
import time

import lightgbm as lgb
import numpy as np

from .blocking import CandidateIndex
from .config import config_digest
from .features import feature_schema, pair_features
from .io import iter_id_lists, parse_id_list
from .metrics import candidate_metrics, macro_f05
from .normalize import normalize_text
from .splits import assign_split
from .storage import code_manifest, file_digest, staging_directory, temporary_database, write_json


def _load_truth(connection, index, path):
    """Audit full label coverage/ownership against the index, storing labels on disk."""
    connection.execute("ATTACH DATABASE ? AS source", (index.path.as_uri() + "?mode=ro",))
    connection.execute("CREATE TABLE truth(sid TEXT PRIMARY KEY, targets TEXT NOT NULL)")
    connection.execute("CREATE TABLE owners(target TEXT PRIMARY KEY, sid TEXT NOT NULL)")
    count = 0
    try:
        for sid, targets in iter_id_lists(path):
            connection.execute("INSERT INTO truth VALUES (?,?)", (sid, ",".join(targets)))
            connection.executemany("INSERT INTO owners VALUES (?,?)", ((target, sid) for target in targets))
            count += 1
            if count % 50000 == 0:
                connection.commit()
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate reference label or target linked to multiple S1 owners") from exc
    checks = [
        ("SELECT sid FROM truth EXCEPT SELECT entity_id FROM source.records WHERE source='S1' LIMIT 1", "Unknown labeled reference"),
        ("SELECT entity_id FROM source.records WHERE source='S1' EXCEPT SELECT sid FROM truth LIMIT 1", "Missing reference label"),
        ("SELECT target FROM owners EXCEPT SELECT entity_id FROM source.records WHERE source IN ('S2','S3') LIMIT 1", "Unknown labeled target"),
    ]
    for query, message in checks:
        row = connection.execute(query).fetchone()
        if row:
            raise ValueError(f"{message}: {row[0]}")
    connection.create_function("norm_country", 1, normalize_text, deterministic=True)
    mismatch = connection.execute("""
        SELECT o.target FROM owners o
        JOIN source.records a ON a.entity_id=o.sid JOIN source.records b ON b.entity_id=o.target
        WHERE norm_country(a.country)<>'' AND norm_country(b.country)<>''
          AND norm_country(a.country)<>norm_country(b.country) LIMIT 1
    """).fetchone()
    if mismatch:
        raise ValueError(f"Cross-country true link invalidates this blocking policy: {mismatch[0]}")
    return count


def _select_queries(connection, config):
    """Keep a bounded, stable min-hash sample per disjoint S1 split and audit balance."""
    limits = {"train": config.train_queries, "tune": config.tune_queries, "holdout": config.holdout_queries}
    heaps = {key: [] for key in limits}
    summary = {key: Counter() for key in limits}
    for sid, country, raw in connection.execute(
        "SELECT r.entity_id,r.country,t.targets FROM source.records r JOIN truth t ON t.sid=r.entity_id WHERE r.source='S1'"
    ):
        group = assign_split(sid, config.seed)
        summary[group]["references"] += 1
        summary[group]["country:" + country] += 1
        summary[group]["singletons"] += not raw
        priority = int.from_bytes(hashlib.blake2b(
            f"query-sample:{config.seed}:{sid}".encode("utf-8"), digest_size=8
        ).digest(), "big")
        item = (-priority, sid)
        heap = heaps[group]
        if len(heap) < limits[group]:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = {key: [sid for _, sid in sorted(heap, reverse=True)] for key, heap in heaps.items()}
    if any(not ids for ids in selected.values()):
        raise ValueError("Each S1 split needs at least one query; increase the input reference pool")
    return selected, {key: dict(value) for key, value in summary.items()}


def predictions_at_threshold(truth, events, scores, threshold):
    """Preserve every query, including no-candidate entities, while selecting scored IDs."""
    if len(events) != len(scores):
        raise ValueError("Score/event lengths differ")
    predictions = {sid: [] for sid in truth}
    for (sid, target), score in zip(events, scores):
        if not math.isfinite(float(score)):
            raise ValueError("Nonfinite model score")
        if score >= threshold:
            predictions[sid].append(target)
    return predictions


def select_threshold(truth, events, scores):
    """Maximize tune-set macro F0.5, preferring the higher threshold on ties."""
    grid = [value / 100 for value in range(101)] + [math.nextafter(1.0, math.inf)]
    results = [(macro_f05(truth, predictions_at_threshold(truth, events, scores, threshold)), threshold)
               for threshold in grid]
    score, threshold = max(results)
    return threshold, score


def _evaluate(truth, events, scores, threshold, countries):
    """Compute overall/per-country macro scores and singleton diagnostics."""
    predictions = predictions_at_threshold(truth, events, scores, threshold)
    singles = [sid for sid in truth if not truth[sid]]
    by_country = {}
    for country in sorted(set(countries.values())):
        subset = {sid: targets for sid, targets in truth.items() if countries[sid] == country}
        by_country[country] = {"references": len(subset), "macro_f05": macro_f05(
            subset, {sid: predictions[sid] for sid in subset})}
    return {"references": len(truth), "macro_f05": macro_f05(truth, predictions),
            "singleton_accuracy": sum(not predictions[sid] for sid in singles) / len(singles) if singles else None,
            "predicted_links": sum(map(len, predictions.values())), "by_country": by_country}


def train_baseline(index_path, ground_truth, model_dir, config):
    """Train on natural candidates; tune threshold separately; publish complete artifacts.

    Input target pools stay complete. Only S1 query groups are sampled. Matrices
    are bounded by max_pairs; no positive is inserted into evaluation retrieval.
    Fixed boosting rounds avoid consulting holdout for early stopping.
    """
    config.validate()
    stage = staging_directory(model_dir)
    started = time.monotonic()
    schema = feature_schema()
    truth_digest = file_digest(ground_truth)
    with CandidateIndex(index_path) as index, temporary_database(stage) as connection:
        _load_truth(connection, index, ground_truth)
        selected, split_summary = _select_queries(connection, config)
        groups = {}
        total_pairs = 0
        for group, ids in selected.items():
            flat, labels = array("f"), array("b")
            truth, candidates, countries, events = {}, {}, {}, []
            for sid in ids:
                record = index.get_record(sid)
                raw = connection.execute("SELECT targets FROM truth WHERE sid=?", (sid,)).fetchone()[0]
                actual = set(parse_id_list(raw))
                truth[sid], countries[sid] = actual, record.country
                found = index.retrieve(record, config.top_k, config.max_query_tokens)
                candidates[sid] = [candidate.record.entity_id for candidate in found]
                if total_pairs + len(found) > config.max_pairs:
                    raise ValueError("max_pairs exceeded; reduce query counts/top_k or deliberately raise the bounded limit")
                for candidate in found:
                    target = candidate.record.entity_id
                    flat.extend(pair_features(record, candidate))
                    labels.append(target in actual)
                    events.append((sid, target))
                total_pairs += len(found)
            matrix = np.frombuffer(flat, dtype=np.float32).reshape((-1, len(schema)))
            groups[group] = {"matrix": matrix, "labels": np.frombuffer(labels, dtype=np.int8),
                             "truth": truth, "candidates": candidates, "countries": countries, "events": events}
        training = groups["train"]
        if set(training["labels"].tolist()) != {0, 1}:
            raise ValueError("Training needs retrieved positives and negatives; inspect the candidate pool")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=config.n_estimators, num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples, learning_rate=config.learning_rate,
            random_state=config.seed, n_jobs=config.threads, deterministic=True, force_col_wise=True,
            verbosity=-1,
        )
        model.fit(training["matrix"], training["labels"], feature_name=list(schema))
        booster = model.booster_
        tuning = groups["tune"]
        tune_scores = booster.predict(tuning["matrix"], num_threads=config.threads) if len(tuning["events"]) else np.empty(0)
        threshold, _ = select_threshold(tuning["truth"], tuning["events"], tune_scores)
        metrics = {"scope": "Query-sampled validation against the entire supplied input index; verify its input hashes/counts.",
                   "split_summary_full_reference_pool": split_summary, "selected_reference_ids": selected,
                   "threshold": threshold, "total_candidate_pairs": total_pairs, "groups": {}}
        for group in ("tune", "holdout"):
            data = groups[group]
            scores = tune_scores if group == "tune" else (
                booster.predict(data["matrix"], num_threads=config.threads) if data["events"] else np.empty(0))
            metrics["groups"][group] = {
                "matching": _evaluate(data["truth"], data["events"], scores, threshold, data["countries"]),
                "blocking": candidate_metrics(data["truth"], data["candidates"]),
            }
        if file_digest(ground_truth) != truth_digest:
            raise ValueError("Ground truth changed during training")
        booster.save_model(str(stage / "model.txt"))
        (stage / "LICENSE").write_bytes((Path(__file__).resolve().parents[2] / "LICENSE").read_bytes())
        write_json(stage / "config.json", asdict(config))
        write_json(stage / "metrics.json", metrics)
        manifest = {"complete": True, "pipeline_version": 1, "config": asdict(config),
                    "config_sha256": config_digest(config), "feature_schema": list(schema),
                    "threshold": threshold, "training_index": index.manifest,
                    "ground_truth_sha256": truth_digest, "code_sha256": code_manifest(),
                    "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "lightgbm", "rapidfuzz")},
                    "model_license": "MIT (included LICENSE); fitted locally using supplied labels with LightGBM",
                    "elapsed_seconds": time.monotonic() - started,
                    "artifacts": {name: file_digest(stage / name) for name in ("model.txt", "config.json", "metrics.json", "LICENSE")}}
        write_json(stage / "manifest.json", manifest)
    destination = Path(model_dir).resolve()
    if destination.exists():
        raise FileExistsError("Model destination appeared during training; complete stage was retained")
    stage.rename(destination)
    return {"model_dir": str(destination), "threshold": threshold, "metrics": metrics,
            "elapsed_seconds": manifest["elapsed_seconds"]}
