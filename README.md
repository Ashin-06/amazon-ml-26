# Amazon ML Challenge 2026: business entity resolution

Matches records for the same business across noisy names and addresses. Submission v30 scored 0.984718 public macro F0.5, 244th globally.

The dataset and trained models aren't included, because they belong to the challenge. Without them you can run the tests and read the code, but not reproduce the score.

```bash
cd code/business_entity_resolution
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest tests -q
```

- [START_HERE.md](START_HERE.md): where to begin
- [Documentation_template.md](Documentation_template.md): the final method
- [docs/RUNBOOK.md](docs/RUNBOOK.md): commands for the full pipeline
- [docs/QUALITY_REPORT.md](docs/QUALITY_REPORT.md): what was verified, and the limits
- `submissions/`: what was submitted

LightGBM needs the OpenMP runtime: on macOS `brew install libomp`, on Linux `sudo apt install libgomp1` if it's missing.

`pyarrow` is newer here than in the submitted run, to pick up a security fix.
