"""Build <team>_submission.zip in the organizers' layout:
output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/{src,README.md,requirements.txt,...},
Documentation_template.md. Only allow-listed paths are included (no data, tokens or workspace files).

Usage (repo root): python code/business_entity_resolution/kaggle/package_submission.py TEAM OUTPUT_DIR [DOC]
"""
import sys
import zipfile
from pathlib import Path

team, out_dir = sys.argv[1], Path(sys.argv[2])
doc = Path(sys.argv[3] if len(sys.argv) > 3 else "Documentation_template.md")
pkg = Path("code/business_entity_resolution")
if doc.read_text(encoding="utf-8").lstrip().startswith("<!-- DRAFT"):
    sys.exit("Documentation still marked DRAFT: fill team fields/results and remove the marker first.")
files = {f"output/{n}": out_dir / n for n in ("matching_results.tsv", "candidate_pairs.tsv")}
files["Documentation_template.md"] = doc
for p in sorted(pkg.rglob("*")):
    rel = p.relative_to(pkg)
    if p.is_file() and "__pycache__" not in rel.parts and rel.parts[0] in ("src", "kaggle", "configs", "tests") \
            or rel.as_posix() in ("README.md", "requirements.txt", "requirements-dev.txt", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        if p.is_file():
            files[f"code/business_entity_resolution/{rel.as_posix()}"] = p
dest = Path(f"{team}_submission.zip")
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, src in files.items():
        z.write(src, arc)
print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
