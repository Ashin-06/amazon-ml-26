"""Prepare one leaderboard upload (matching TSV + code zip) only if every check passes.

Usage (repo root): python code/business_entity_resolution/kaggle/release.py NAME RUN_OUT_DIR GIT_REF TEST_DIR VALIDATOR
  NAME         e.g. v10  -> submissions/v10_matching_results.tsv, submissions/v10_code.zip
  RUN_OUT_DIR  folder with matching_results.tsv and candidate_pairs.tsv from that run
  GIT_REF      tag/commit whose code produced the run (the zip is built from exactly this ref)
Run it when no training job is active (the organizers' validator loads all test IDs into memory).
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

name, out, ref, test_dir, validator = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
m, c = out / "matching_results.tsv", out / "candidate_pairs.tsv"
here = Path(__file__).resolve().parent


def run(cmd, what):
    """Run one check; stop the release if it does not exit 0."""
    print(f"\n>>> {what}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"RELEASE STOPPED: {what} failed; nothing was copied.")


run([sys.executable, validator, "--matching", str(m), "--candidate", str(c), "--test-dir", test_dir, "--check-ids"],
    "organizers' validator (--check-ids)")
run([sys.executable, str(here / "check_submission.py"), str(m), test_dir, "--candidate", str(c)],
    "byte-level / rule-level checker")
dst = Path("submissions"); dst.mkdir(exist_ok=True)
tsv, zp = dst / f"{name}_matching_results.tsv", dst / f"{name}_code.zip"
for p in (tsv, zp):
    if p.exists():
        sys.exit(f"{p} already exists; choose a new NAME (files are never overwritten)")
shutil.copyfile(m, tsv)
run(["git", "archive", "--format=zip", f"--prefix=code/business_entity_resolution/", "-o", str(zp),
     f"{ref}:code/business_entity_resolution"], f"code zip from git ref {ref}")
for p in (tsv, zp):
    print(f"{p}  {p.stat().st_size / 1e6:.1f} MB  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
print(f"\nREADY TO UPLOAD: {tsv} (Matching Results) + {zp} (Code File)")
