# Fast rescue model: measured result, 27 September 2026

## Result and selected local artifact

The new v28fast_rescue candidate combines v27's unchanged French rows with a learned India/US rescue matcher. It is prepared, not submitted or publicly evaluated. The confirmed public best remains v26 at 0.98417; one team upload remains.

The bounded CPU pilot, including three-fold comparisons of two small classifiers, took **22.578 seconds**. A standardized logistic regression with C=0.1 won the predeclared tuning comparison; its frozen probability cutoff is 0.69. It was fitted on 9,358 eligible tuning pairs using 42 text/frequency features. IDs are join/split/tie-break keys, never predictive features. No test labels or external business data were used.

| Exact macro F0.5, final India/US split | Score |
|---|---:|
| v26 fixed rescue baseline | 0.9893496186790386 |
| Learned rescue | 0.9895825572394727 |
| Paired difference | +0.0002329385604340051 |
| US | 0.9897916033551672 |
| India | 0.9892709755087349 |

The descriptive paired 95% interval for the overall difference is [0.0000872280, 0.0003786491]. India's interval includes zero. The old rule added 234 final-split pairs (218 true); the learned matcher adds 395 (366 true): 148 additional true links and 13 additional false links. This final split has been reused across earlier experiments; these intervals do not remove selection bias or establish public-test performance. Held-pool ownership also differs from full-pool deployment.

## Why it is faster

The supplied misspelled-street/same-house-number observation becomes a small learned matcher on the existing rescue route, rather than requiring another full feature-generation and neural-training run. It compares raw and normalized names/addresses, token overlap, character similarity, number differences and full-pool name frequency. The model can learn combinations that a single raw-string threshold misses. This does not recover links outside the existing candidate pool.

Full test rescue inference covered 440,105 pairs in 103.156 seconds. The output has 1,732,544 reference rows and 5,793,591 links. Compared with v27, 5,761 references change: 5,077 links added and 1,118 removed. All 259,452 French rows are preserved exactly. The 181,275,851-pair v27 candidate pool is unchanged.

## Validation and limitations

The full independent checker passed original-ID validity, complete reference coverage, output format, target ownership, all candidate IDs and match containment, using bounded DuckDB memory. Two focused export tests passed. Source/model/output hashes are sealed in the manifests. The official validator previously exhausted memory on the candidate file; that is not reported as an official full-validator success. All 12 original supplied-file hashes passed the earlier evening quality check; 32 pre-existing documentation/link issues remain.

This is a measured local improvement, **not a 0.995 model**. There are no French labels, and preserving v27 does not establish the accuracy of its French additions. No leaderboard score can be promised. Do not infer missing true links from predicted link counts or competitor scores. Do not spend the last attempt on an unvalidated extra threshold probe.

## Files and reproduction

- Prepared matching file: `submissions/v28fast_rescue_matching_results.tsv`.
- Exact candidate pool: `work/comp/v25/candidate_pairs_v27.tsv`.
- Review archive: `submissions/v28fast_rescue_code.zip`.
- Design, training, locked final evaluation, inference and audit scripts: `artifacts/after-0982910-20260927/` (`FAST_RESCUE_DESIGN.md`, `fast_rescue.py`, `final_fast_rescue.py`, `export_fast_rescue.py`, `validate_fast_rescue.py`, `audit_fast_rescue_delta.py`).
- Machine-readable evidence: `fast_rescue_v1/audit.json`, `fast_rescue_final_v1/{locked_manifest,audit}.json`, `fast_rescue_test_v1/{manifest,delta_audit}.json`, `fast_rescue_validation_v1/audit.json`.

Commands from the workspace are `python artifacts/after-0982910-20260927/fast_rescue.py`, then `final_fast_rescue.py`, then `export_fast_rescue.py` in that directory. They intentionally reject existing output destinations; use a separate workspace with the same frozen prerequisite artifacts to reproduce, not an overwrite of this release. The export requires the exact v25 base output, v27 French output, normalized supplied records, rescue pairs, frozen stack helper and model described in its manifest. Validation uses the independent checker's original-data paths.

The ZIP is an allowlisted code/model/evidence review archive. It includes the previous v27 code archive content, the new rescue model, and the fitted v25 correction model. It is **not** a complete offline Round-2 submission: upstream neural checkpoints/tokenizers, full data/artifacts and both output TSVs are not all embedded. Organizer clarifications say matching TSV tonight and ZIP in Round 2. Root methodology remains a draft pending final team selection.


## Further attempts after release

Five additional model/retrieval variants failed the tuning adoption checks. v28 remains unchanged; no new final evaluation or leaderboard upload occurred.
