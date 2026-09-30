"""Exact per-reference error accounting for frozen decisions, without threshold search."""
import math
from collections import Counter


def error_rows(truth, prediction, references, candidate_hits, countries):
    """Decompose loss into retrieval, later missed matches and false-positive cost.

    candidate_hits counts naturally retrieved TRUE links, not all candidates.
    Predictions may include competitors outside references; those are not scored.
    Every requested reference must have explicit truth, hit count and country.
    No labels or decisions are changed. An empty/empty reference scores one.
    """
    references = list(references)
    if not references or len(set(references)) != len(references):
        raise ValueError("References must be nonempty and unique")
    rows = []
    for sid in references:
        if sid not in truth or sid not in candidate_hits or sid not in countries:
            raise ValueError("Missing truth, retrieval count or country for a reference")
        actual, predicted = set(truth[sid]), set(prediction.get(sid, ()))
        nt, npred, tp = len(actual), len(predicted), len(actual & predicted)
        hits = candidate_hits[sid]
        if not isinstance(hits, int) or isinstance(hits, bool) or not 0 <= tp <= hits <= nt:
            raise ValueError("Inconsistent retrieved-positive count")
        fp, fn = npred - tp, nt - tp
        oracle = 1.0 if nt == 0 else 5.0 * hits / (4 * hits + nt)
        without_fp = 1.0 if nt == 0 else 5.0 * tp / (4 * tp + nt)
        score = (float(npred == 0) if nt == 0 else 5.0 * tp / (4 * npred + nt))
        rows.append(dict(sid=sid, country=countries[sid], true_links=nt,
                         predicted_links=npred, tp=tp, fp=fp, fn=fn,
                         retrieved_true_links=hits, retrieval_misses=nt-hits,
                         later_misses=hits-tp, f05=score, oracle_f05=oracle,
                         retrieval_loss=1-oracle, later_miss_loss=oracle-without_fp,
                         false_positive_loss=without_fp-score))
    return rows


def summarize_errors(rows):
    """Summarize exact scores and additive loss by country, including singletons."""
    rows = list(rows)
    if not rows or len({r['sid'] for r in rows}) != len(rows):
        raise ValueError("Rows must be nonempty and unique")
    def summary(part):
        """Aggregate one nonempty reference population without weighting by pair count."""
        n = len(part)
        result = {'references': n}
        for name in ('true_links','predicted_links','tp','fp','fn','retrieved_true_links',
                     'retrieval_misses','later_misses'):
            result[name] = sum(r[name] for r in part)
        for field, name in (('f05','macro_f05'),('oracle_f05','oracle_macro_f05'),
                            ('retrieval_loss','retrieval_loss'),('later_miss_loss','later_miss_loss'),
                            ('false_positive_loss','false_positive_loss')):
            result[name] = math.fsum(r[field] for r in part)/n
        singletons = [r for r in part if r['true_links'] == 0]
        result.update(singletons=len(singletons),
                      singleton_accuracy=(sum(r['predicted_links']==0 for r in singletons)/len(singletons)
                                          if singletons else None),
                      references_with_fp=sum(r['fp']>0 for r in part),
                      references_with_fn=sum(r['fn']>0 for r in part),
                      exact_match_rate=sum(r['fp']==0 and r['fn']==0 for r in part)/n)
        return result
    return {'all':summary(rows), 'by_country':{
        country:summary([r for r in rows if r['country']==country])
        for country in sorted({r['country'] for r in rows})}}


def reweight_known_countries(report, test_country_counts):
    """Reweight labeled country scores; explicitly exclude unseen countries from accuracy.

    This is a distribution diagnostic. It is neither a prediction of the public
    score nor a measurement of an unlabeled country's accuracy.
    """
    counts = Counter(test_country_counts)
    if not counts or any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("Expected nonnegative integer country counts with a positive total")
    known = {c:n for c,n in counts.items() if c in report['by_country']}
    total, covered = sum(counts.values()), sum(known.values())
    weighted = sum(n*report['by_country'][c]['macro_f05'] for c,n in known.items())
    return {'known_country_reweighted_f05':weighted/covered if covered else None,
            'covered_reference_fraction':covered/total,
            'unlabeled_country_counts':{c:n for c,n in counts.items() if c not in known},
            'scope':'country-mixture diagnostic only; within-country shift remains unknown'}
