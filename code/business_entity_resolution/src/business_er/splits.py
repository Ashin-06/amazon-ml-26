"""Stable group assignment; ownership propagation belongs to preparation."""

import hashlib
import math

from .io import validate_id


def assign_split(
    entity_id: str, seed: int = 2026,
    train_fraction: float = 0.8, tune_fraction: float = 0.1,
) -> str:
    """Assign an S1 group deterministically without relying on row order.

    This helper does not itself propagate labels to targets or stratify groups.
    The preparation stage must enforce those contracts and audit split balance.
    """
    validate_id(entity_id, ("S1",))
    if not (math.isfinite(train_fraction) and math.isfinite(tune_fraction)):
        raise ValueError("Split fractions must be finite")
    if train_fraction <= 0 or tune_fraction < 0 or train_fraction + tune_fraction >= 1:
        raise ValueError("Require positive train and holdout fractions, nonnegative tune")
    digest = hashlib.blake2b(
        f"{seed}\0{entity_id}".encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + tune_fraction:
        return "tune"
    return "holdout"
