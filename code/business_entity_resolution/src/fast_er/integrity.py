"""Content fingerprints and fail-closed completion records for the fast pipeline."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

PACKAGES = ("anyascii", "duckdb", "lightgbm", "numpy", "pandas", "pyarrow", "rapidfuzz", "scipy", "xgboost")
SPLIT_VERSION = "md5-hold26-v1"


def digest(path):
    """Hash a file in bounded blocks; never load a complete data file into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files_digest(paths):
    """Describe explicitly named files, refusing symlinks and duplicate basenames."""
    result = {}
    for item in paths:
        path = Path(item)
        if path.is_symlink() or not path.is_file() or path.name in result:
            raise ValueError("Missing, linked or ambiguous artifact: " + path.name)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def code_digest():
    """Fingerprint every fast-pipeline module, including its drivers and integrity rules."""
    return files_digest(sorted(Path(__file__).parent.glob("*.py")))


def environment():
    """Record the Python version and libraries that affect numerical or data behavior."""
    return {"python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def require_first_model_candidates(manifest):
    """Reject exports that hide cascade inputs behind a smaller survivor count."""
    inputs = manifest.get('inputs', {})
    pairs, blocked = inputs.get('pairs'), inputs.get('blocked_pairs')
    if (inputs.get('candidate_scope') != 'input_to_first_scoring_model'
            or type(pairs) is not int or type(blocked) is not int
            or pairs < 0 or pairs != blocked):
        raise ValueError('Candidate export must contain every first scoring model input')


def read_json(path):
    """Read UTF-8 JSON without permitting a missing completion manifest."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or linked manifest: " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def seal(directory, filename, inputs, outputs):
    """Publish a completion record only after every explicitly listed output exists."""
    directory = Path(directory)
    meta = {"version": 1, "complete": True, "inputs": inputs,
            "outputs": files_digest(directory / name for name in outputs)}
    with (directory / filename).open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2, sort_keys=True)
    return meta


def verify(directory, filename, inputs=None):
    """Reject incomplete, changed or incompatible artifacts; never bless legacy runs."""
    directory = Path(directory)
    meta = read_json(directory / filename)
    if meta.get("version") != 1 or meta.get("complete") is not True:
        raise ValueError("Incomplete or unsupported manifest: " + filename)
    if inputs is not None and meta.get("inputs") != inputs:
        raise ValueError("Input/code/settings fingerprint mismatch: " + filename)
    names = meta.get("outputs", {})
    if not names or any(Path(n).name != n for n in names):
        raise ValueError("Invalid output allowlist: " + filename)
    if files_digest(directory / name for name in names) != names:
        raise ValueError("Output fingerprint mismatch: " + filename)
    return meta


def stage_inputs(pq_dir, split, gt, settings, lexicon=None):
    """Bind blocking to raw inputs, labels, code, environment, chunks and lexicon."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    paths = [Path(pq_dir) / f"{split}_source{i}.parquet" for i in (1, 2, 3)]
    if gt:
        paths.append(Path(gt))
    converted = verify(pq_dir, "conversion_manifest.json")
    consumed = files_digest(paths)
    if any(converted["outputs"].get(name) != value for name,value in consumed.items()):
        raise ValueError("Stage inputs differ from verified converted inputs")
    return {"conversion": converted, "code": code_digest(), "environment": environment(), "split": split,
            "split_version": SPLIT_VERSION, "files": consumed, "settings": settings,
            "lexicon": digest(lexicon) if lexicon else None}


def verify_stage(work, split):
    """Verify finished normalized/candidate files and the currently selected lexicon."""
    work = Path(work)
    meta = verify(work / split, "stage1_meta.json")
    recorded = meta["inputs"]
    if recorded.get("code") != code_digest() or recorded.get("environment") != environment():
        raise ValueError("Stage code/environment changed; rebuild in a fresh WORK directory")
    if recorded.get("lexicon") != digest(work / "lexicon.json"):
        raise ValueError("Lexicon fingerprint mismatch")
    return meta
