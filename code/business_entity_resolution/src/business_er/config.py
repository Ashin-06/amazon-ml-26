"""Strict, reproducible settings for the first offline CPU baseline."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib
import json
import math


@dataclass(frozen=True)
class BaselineConfig:
    """Bounded query sampling and CPU model settings; no account information."""

    seed: int = 2026
    top_k: int = 10
    max_query_tokens: int = 12
    train_queries: int = 10000
    tune_queries: int = 1500
    holdout_queries: int = 1500
    max_pairs: int = 750000
    n_estimators: int = 200
    num_leaves: int = 31
    min_child_samples: int = 30
    learning_rate: float = 0.05
    threads: int = 2

    def validate(self):
        """Reject unknown types/ranges before allocating indexes or model matrices."""
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "learning_rate":
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                    raise ValueError("learning_rate must be finite in (0,1]")
            elif type(value) is not int:
                raise ValueError(f"{field.name} must be an integer")
            elif field.name != "seed" and value <= 0:
                raise ValueError(f"{field.name} must be positive")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        if self.top_k > 1000 or self.max_query_tokens > 100:
            raise ValueError("Retrieval limits exceed supported baseline bounds")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        return self


def load_config(path=None):
    """Read a JSON object with only known setting names; validate all values."""
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys) if path else {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(data) - {field.name for field in fields(BaselineConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return BaselineConfig(**data).validate()


def _unique_keys(pairs):
    """Reject duplicate JSON keys rather than silently selecting the last setting."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def config_digest(config):
    """Hash canonical settings so mismatched runs can be detected."""
    config.validate()
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
