"""Challenge-specific entity and macro F0.5, including singleton semantics."""

from itertools import zip_longest
from math import fsum
from collections.abc import Iterable, Mapping


def entity_f05(truth: Iterable[str], prediction: Iterable[str]) -> float:
    """Score sets for one S1; empty/empty is 1, one-empty is 0.

    Duplicates have set semantics here. Structural validation must reject them
    before scoring submission files.
    """
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    if not prediction:
        return 0.0
    tp = len(truth & prediction)
    fp, fn = len(prediction - truth), len(truth - prediction)
    return 5.0 * tp / (5 * tp + 4 * fp + fn)


def _same_keys(truth: Mapping, predictions: Mapping) -> None:
    """Enforce the complete nonempty reference universe before averaging."""
    if not truth:
        raise ValueError("Cannot score an empty reference universe")
    if truth.keys() != predictions.keys():
        raise ValueError("Truth and predictions must contain identical S1 IDs")


def macro_f05(
    truth_by_id: Mapping[str, Iterable[str]],
    predictions_by_id: Mapping[str, Iterable[str]],
) -> float:
    """Score every reference equally; require identical nonempty key sets."""
    _same_keys(truth_by_id, predictions_by_id)
    return fsum(
        entity_f05(truth, predictions_by_id[sid]) for sid, truth in truth_by_id.items()
    ) / len(truth_by_id)


def score_aligned_rows(truth_rows: Iterable, prediction_rows: Iterable) -> dict:
    """Score same-order, previously uniqueness-validated rows in bounded memory.

    Exhausts both iterators. Rejects order/length mismatch. It cannot detect
    repeated aligned S1 IDs without extra storage; perform uniqueness validation
    upstream. Do not use this on differently ordered truth and prediction files.
    """
    sentinel = object()
    count = tp = fp = fn = singletons = singleton_correct = 0
    # Kahan summation keeps accumulated rounding small on millions of entities.
    score_sum = compensation = 0.0
    for truth_row, prediction_row in zip_longest(
        truth_rows, prediction_rows, fillvalue=sentinel
    ):
        if truth_row is sentinel or prediction_row is sentinel:
            raise ValueError("Truth/prediction row counts differ")
        sid, truth = truth_row
        predicted_sid, prediction = prediction_row
        if sid != predicted_sid:
            raise ValueError(f"Unaligned S1 rows: {sid!r} versus {predicted_sid!r}")
        truth, prediction = set(truth), set(prediction)
        tp += len(truth & prediction)
        fp += len(prediction - truth)
        fn += len(truth - prediction)
        singletons += not truth
        singleton_correct += not truth and not prediction
        increment = entity_f05(truth, prediction) - compensation
        updated = score_sum + increment
        compensation = (updated - score_sum) - increment
        score_sum = updated
        count += 1
    if not count:
        raise ValueError("Cannot score an empty reference universe")
    return {
        "references": count, "macro_f05": score_sum / count,
        "true_positive_links": tp, "false_positive_links": fp,
        "false_negative_links": fn, "singletons": singletons,
        "singleton_accuracy": singleton_correct / singletons if singletons else None,
    }


def candidate_metrics(
    truth_by_id: Mapping[str, Iterable[str]],
    candidates_by_id: Mapping[str, Iterable[str]],
) -> dict:
    """Report natural candidate recall and oracle score on a bounded diagnostic set."""
    _same_keys(truth_by_id, candidates_by_id)
    positives = retrieved = candidate_count = 0
    recalls, oracle = [], []
    for sid, truth in truth_by_id.items():
        truth, candidates = set(truth), set(candidates_by_id[sid])
        hits = truth & candidates
        positives += len(truth)
        retrieved += len(hits)
        candidate_count += len(candidates)
        if truth:
            recalls.append(len(hits) / len(truth))
        oracle.append(entity_f05(truth, hits))
    return {
        "references": len(truth_by_id), "positive_links": positives,
        "retrieved_positive_links": retrieved,
        "link_recall": retrieved / positives if positives else None,
        "non_singleton_macro_recall": fsum(recalls) / len(recalls) if recalls else None,
        "oracle_macro_f05": fsum(oracle) / len(oracle),
        "candidate_pairs": candidate_count,
        "mean_candidates": candidate_count / len(truth_by_id),
    }
