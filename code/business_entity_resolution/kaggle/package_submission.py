"""Build a verified final archive from an explicit selected fast-pipeline run."""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fast_er.integrity import code_digest, digest, read_json, verify, verify_stage, require_first_model_candidates
from fast_er.model import load_models, model_path

DENY = re.compile(r"(?i)(kaggle\.json|access_token|\.env$|secret|credential|token|\.pem$|\.key$|id_rsa)")
TOP = ("README.md", "requirements.txt", "requirements-dev.txt", "LICENSE", "THIRD_PARTY_NOTICES.md")


def check_document(path):
    """Require completed real team fields and refuse draft markers and template placeholders."""
    text = Path(path).read_text(encoding="utf-8")
    if re.search(r"(?i)DRAFT[: ]|\[(?:Your |List |Required|Date|Fill )|\bTODO\b", text):
        raise ValueError("Methodology is still a draft or contains placeholders")
    for field in ("Team Name", "Team Members", "Submission Date"):
        match = re.search(r"^\*\*" + field + r":\*\*\s*([^\r\n]+)", text, re.MULTILINE)
        if not match or not match[1].strip() or "[" in match[1]:
            raise ValueError("Complete methodology field: " + field)


def build_package(output_dir, run_dir, doc, expected_manifest, destination, model_dir="model"):
    """Package only hash-verified model, output, source and explicitly allowed metadata files."""
    out, run, destination = Path(output_dir), Path(run_dir), Path(destination)
    if destination.exists():
        raise FileExistsError("Archive exists; choose a new versioned destination")
    check_document(doc)
    predicted = verify(out, "prediction_manifest.json")
    require_first_model_candidates(predicted)
    stage = verify_stage(run, "test")
    if predicted["inputs"]["stage"] != stage or predicted["inputs"]["code"] != code_digest():
        raise ValueError("Output/test stage/code mismatch")
    ensemble_name = predicted["inputs"].get("ensemble_name")
    if ensemble_name:
        ensemble = verify(model_path(run,ensemble_name), "ensemble_manifest.json")
        if digest(model_path(run,ensemble_name)/"ensemble_manifest.json") != predicted["inputs"].get("ensemble"):
            raise ValueError("Output/ensemble mismatch")
        model_dirs = list(ensemble["inputs"]["members"])
        if {n:digest(model_path(run,n)/"model_manifest.json") for n in model_dirs} != ensemble["inputs"]["members"]:
            raise ValueError("Ensemble member changed")
    else:
        model_dirs = [model_dir]
        if predicted["inputs"]["model"] != digest(model_path(run,model_dir)/"model_manifest.json"):
            raise ValueError("Output/selected model mismatch")
    models = {}
    for name in model_dirs:
        directory = model_path(run,name)
        load_models(directory)
        models[name] = verify(directory,"model_manifest.json")
        if models[name]["inputs"]["lexicon"] != digest(run/"lexicon.json"):
            raise ValueError("Selected model/lexicon mismatch")
    expected = read_json(expected_manifest)
    raw = stage["inputs"]["conversion"]["inputs"]["files"]
    for i in (1,2,3):
        name = f"test_source{i}.tsv"
        entries = [value for key,value in expected.items() if Path(key).name == name]
        if len(entries) != 1 or entries[0] != raw.get(name):
            raise ValueError("Selected run does not use the audited official test data: " + name)
    pkg = Path(__file__).resolve().parents[1]
    files = {"output/"+n:out/n for n in ("matching_results.tsv","candidate_pairs.tsv","prediction_manifest.json")}
    files["Documentation_template.md"] = Path(doc)
    for folder in ("src", "tests"):
        for path in sorted((pkg/folder).rglob("*.py")):
            files["code/business_entity_resolution/"+path.relative_to(pkg).as_posix()] = path
    for name in TOP + ("configs/baseline.json", "kaggle/package_submission.py", "kaggle/build_kernel.py"):
        files["code/business_entity_resolution/"+name] = pkg/name
    for model_name, model in models.items():
        for name in (*model["outputs"], "model_manifest.json"):
            files[f"code/business_entity_resolution/models/{model_name}/{name}"] = model_path(run,model_name)/name
    if ensemble_name:
        for name in ("ensemble_manifest.json","metrics.json"):
            files[f"code/business_entity_resolution/models/{ensemble_name}/{name}"] = model_path(run,ensemble_name)/name
    files["code/business_entity_resolution/models/lexicon.json"] = run/"lexicon.json"
    for name, path in files.items():
        if DENY.search(name) or path.is_symlink() or not path.is_file():
            raise ValueError("Refused missing/linked/sensitive package path: " + name)
        if name.startswith("code/business_entity_resolution/src/") and not path.resolve().is_relative_to(pkg/"src"):
            raise ValueError("Source path escapes the package")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed write has no completion claim; exclusive creation never replaces an existing archive.
    with zipfile.ZipFile(destination,"x",zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():
            archive.write(path,name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity failed")
    return {"archive":str(destination),"files":len(files),"sha256":digest(destination)}


def main():
    """Parse required run, original-input manifest and new archive destination."""
    ap = argparse.ArgumentParser()
    ap.add_argument("team"); ap.add_argument("output_dir")
    ap.add_argument("--run", required=True); ap.add_argument("--doc", default="Documentation_template.md")
    ap.add_argument("--input-manifest", default="docs/research/input_manifest.json")
    ap.add_argument("--destination", required=True)
    ap.add_argument("--model-dir", default="model")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", a.team):
        ap.error("team must contain only letters, digits, underscores or hyphens")
    print(json.dumps(build_package(a.output_dir,a.run,a.doc,a.input_manifest,a.destination,a.model_dir),indent=2))


if __name__ == "__main__":
    main()
