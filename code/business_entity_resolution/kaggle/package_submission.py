"""Build <team>_submission.zip in the organizers' layout:
output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/{src,README.md,requirements.txt,...},
Documentation_template.md, and optionally the selected run's model bundle under code/.../model/.

Only allow-listed file types are included; credential-like names are always refused; an existing
archive is never overwritten.

Usage (repo root): python code/business_entity_resolution/kaggle/package_submission.py TEAM OUTPUT_DIR [--run RUN_DIR] [--doc DOC]
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

ALLOWED_EXT = {".py", ".md", ".txt", ".json", ".sh", ".cfg", ".toml"}
DENY = re.compile(r"(?i)(kaggle\.json|access_token|\.env$|secret|credential|token|\.pem$|\.key$|id_rsa)")
RUN_FILES = ("lgb_A.txt", "lgb_B.txt", "lgb2.txt", "metrics.json")

ap = argparse.ArgumentParser()
ap.add_argument("team"); ap.add_argument("output_dir")
ap.add_argument("--run", help="run work dir holding model/ and lexicon.json (e.g. work/v10)")
ap.add_argument("--doc", default="Documentation_template.md")
a = ap.parse_args()

team = re.sub(r"[^A-Za-z0-9_-]+", "_", a.team).strip("_")
if not team:
    sys.exit("team name must contain letters or digits")
doc, out_dir, pkg = Path(a.doc), Path(a.output_dir), Path("code/business_entity_resolution").resolve()
if doc.read_text(encoding="utf-8").lstrip().startswith("<!-- DRAFT"):
    sys.exit("Documentation still marked DRAFT: fill team fields/results and remove the marker first.")

files = {f"output/{n}": out_dir / n for n in ("matching_results.tsv", "candidate_pairs.tsv")}
files["Documentation_template.md"] = doc
for p in sorted(pkg.rglob("*")):
    rel = p.relative_to(pkg)
    if (not p.is_file() or p.is_symlink() or "__pycache__" in rel.parts or DENY.search(rel.as_posix())
            or (p.suffix.lower() not in ALLOWED_EXT and p.name != "LICENSE") or not p.resolve().is_relative_to(pkg)):
        continue
    if rel.parts[0] in ("src", "kaggle", "configs", "tests") or len(rel.parts) == 1:
        files[f"code/business_entity_resolution/{rel.as_posix()}"] = p
if a.run:
    run = Path(a.run)
    for n in RUN_FILES:
        files[f"code/business_entity_resolution/model/{n}"] = run / "model" / n
    files["code/business_entity_resolution/model/lexicon.json"] = run / "lexicon.json"
missing = [str(s) for s in files.values() if not Path(s).is_file()]
if missing:
    sys.exit(f"missing files: {missing}")

dest = Path(f"{team}_submission.zip")
if dest.exists():
    sys.exit(f"{dest} already exists; move it away first (never overwritten)")
with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED) as z:
    for arc, src in files.items():
        z.write(src, arc)
with zipfile.ZipFile(dest) as z:
    names = z.namelist()
bad = [n for n in names if DENY.search(n)]
if bad:
    dest.unlink()
    sys.exit(f"refused: sensitive-looking entries {bad}")
print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
