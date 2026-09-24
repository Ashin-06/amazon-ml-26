"""A fixed numeric pair-feature schema with no learned statistics or labels."""

from rapidfuzz import fuzz

from .normalize import fold_latin_accents, normalize_text, numeric_tokens


def feature_schema():
    """Return immutable feature order; saved models must match it exactly."""
    return (
        "name_exact", "name_ratio", "name_token_sort", "name_token_set", "name_jaccard", "name_length_ratio",
        "address_exact", "address_ratio", "address_token_sort", "address_token_set", "address_jaccard", "address_length_ratio",
        "folded_name_ratio", "folded_address_ratio", "number_jaccard", "number_conflict",
        "reference_name_missing", "target_name_missing", "reference_address_missing", "target_address_missing",
        "country_equal", "country_unknown", "target_source3", "name_route", "address_route",
        "name_reciprocal_rank", "address_reciprocal_rank", "reference_name_nonascii", "target_name_nonascii",
    )


def _similarities(left, right):
    """Return six nonempty equality/edit/token/length signals; missing is zero."""
    if not left or not right:
        return [0.0] * 6
    a, b = set(left.split()), set(right.split())
    return [float(left == right), fuzz.ratio(left, right) / 100,
            fuzz.token_sort_ratio(left, right) / 100, fuzz.token_set_ratio(left, right) / 100,
            len(a & b) / len(a | b), min(len(left), len(right)) / max(len(left), len(right))]


def pair_features(reference, candidate):
    """Produce finite features for exactly one retrieved pair, retaining missingness."""
    target = candidate.record
    rn, tn = normalize_text(reference.business_name), normalize_text(target.business_name)
    ra, ta = normalize_text(reference.business_address), normalize_text(target.business_address)
    rc, tc = normalize_text(reference.country), normalize_text(target.country)
    numbers_a, numbers_b = numeric_tokens(reference.business_address), numeric_tokens(target.business_address)
    values = _similarities(rn, tn) + _similarities(ra, ta)
    values += [fuzz.ratio(fold_latin_accents(rn), fold_latin_accents(tn)) / 100 if rn and tn else 0,
               fuzz.ratio(fold_latin_accents(ra), fold_latin_accents(ta)) / 100 if ra and ta else 0,
               len(numbers_a & numbers_b) / len(numbers_a | numbers_b) if numbers_a and numbers_b else 0,
               float(bool(numbers_a and numbers_b) and not numbers_a & numbers_b),
               float(not rn), float(not tn), float(not ra), float(not ta),
               float(bool(rc and tc) and rc == tc), float(not rc or not tc),
               float(target.entity_id.startswith("S3-")), float(candidate.name_rank > 0),
               float(candidate.address_rank > 0), 1 / candidate.name_rank if candidate.name_rank else 0,
               1 / candidate.address_rank if candidate.address_rank else 0,
               float(not reference.business_name.isascii()), float(not target.business_name.isascii())]
    if len(values) != len(feature_schema()):
        raise RuntimeError("Feature schema and implementation disagree")
    return values
