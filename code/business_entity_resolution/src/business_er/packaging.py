"""Allowlisted final archives with model/output/input provenance checks."""

from pathlib import Path
import json
import os
import re
import tempfile
import zipfile

from .predict import load_model
from .storage import code_manifest, file_digest


def build_package(output_dir, model_dir, documentation, expected_inputs_manifest, destination):
    """Package a verified run only after team details and expected test inputs agree.

    The caller supplies the original audited input manifest, not a guessed row
    count. ZIP contents are allowlisted source/config/model/output/doc files;
    original data, cloud tokens and arbitrary workspace files are never included.
    """
    output_dir, model_dir = Path(output_dir).resolve(), Path(model_dir).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Archive already exists; choose a versioned destination")
    _, _, model_manifest = load_model(model_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("pipeline_version") != 1:
        raise ValueError("Inference output has no supported completion manifest")
    for flag in ("target_ids_checked", "reference_uniqueness_checked", "candidate_containment_checked"):
        if manifest.get("validation", {}).get(flag) is not True:
            raise ValueError(f"Inference did not complete required validation: {flag}")
    if manifest["code_sha256"] != code_manifest():
        raise ValueError("Output was generated with different source code")
    if manifest["model_manifest_sha256"] != file_digest(model_dir / "manifest.json"):
        raise ValueError("Outputs and chosen model belong to different runs")
    if manifest["model_artifacts"] != model_manifest["artifacts"]:
        raise ValueError("Output/model artifact fingerprints differ")
    expected = json.loads(Path(expected_inputs_manifest).read_text(encoding="utf-8"))
    for source, name in (("S1", "test_source1.tsv"), ("S2", "test_source2.tsv"), ("S3", "test_source3.tsv")):
        matches = [value for key, value in expected.items() if Path(key).name == name]
        if len(matches) != 1 or matches[0]["sha256"] != manifest["input_index"]["inputs"][source]["sha256"]:
            raise ValueError(f"Inference inputs do not match the audited test data: {source}")
    text = Path(documentation).read_text(encoding="utf-8")
    if "DRAFT: NOT FOR SUBMISSION" in text:
        raise ValueError("Finalize the draft methodology with this run's actual results before packaging")
    for field in ("Team Name", "Team Members", "Submission Date"):
        match = re.search(r"^\*\*" + re.escape(field) + r":\*\*\s*([^\r\n]+)", text, re.MULTILINE)
        if not match or not match[1].strip() or "[" in match[1] or re.search(r"\b(TODO|NOT PROVIDED|PENDING)\b", match[1], re.I):
            raise ValueError(f"Complete the methodology field: {field}")
    if re.search(r"\[(?:Your |List |Required|Date|Fill )", text, re.I):
        raise ValueError("Methodology still contains template placeholders")
    root = Path(__file__).resolve().parents[2]
    files = []
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        path = output_dir / name
        if file_digest(path) != manifest["output_sha256"][name]:
            raise ValueError(f"Output changed after validation: {name}")
        files.append((path, "output/" + name))
    files.append((output_dir / "manifest.json", "output/manifest.json"))
    for relative in code_manifest():
        files.append((root / "src" / relative, "code/business_entity_resolution/src/" + relative))
    for path in sorted((root / "tests").glob("*.py")):
        files.append((path, "code/business_entity_resolution/tests/" + path.name))
    for name in ("README.md", "requirements.txt", "requirements-dev.txt", "configs/baseline.json", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        files.append((root / name, "code/business_entity_resolution/" + name))
    for name in ("manifest.json", "model.txt", "config.json", "metrics.json", "LICENSE"):
        files.append((model_dir / name, "code/business_entity_resolution/artifacts/model/" + name))
    files.append((Path(documentation), "Documentation_template.md"))
    provenance = {Path(key).name: value for key, value in expected.items()
                  if Path(key).name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, relative in files:
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Required archive file is missing or is a symlink: {relative}")
                archive.write(path, relative)
            archive.writestr("input_fingerprints.json", json.dumps(provenance, indent=2))
        with zipfile.ZipFile(name) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive integrity check failed")
        if destination.exists():
            raise FileExistsError("Archive destination appeared during packaging")
        os.rename(name, destination)
    finally:
        if Path(name).exists():
            Path(name).unlink()
    return {"archive": str(destination), "sha256": file_digest(destination), "files": len(files) + 1,
            "references": manifest["references"]}
