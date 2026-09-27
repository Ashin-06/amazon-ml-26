# Private Kaggle neural experiment: xenc7

Written before implementation. Deadline 27 September 23:59 IST; current time
about 14:20 IST. Two team leaderboard submissions remain. No automatic submission.

Reuse the authorized principal Kaggle account's private hard-pair dataset,
provided raw parquet dataset, frozen xenc4 checkpoint and completed v24 kernel
outputs. Do not upload credentials, whole workspace or any external business
data. No alternative accounts to work around quotas. Existing runs are complete;
the experiment uses a new private kernel, not edits of earlier kernels.

Hypothesis: extra hard-pair training with legal-form/presentation augmentation
and modest weighting of missing-address and address-conflict pairs may improve
robustness. This is a hypothesis; earlier larger models and sibling features
provided only small gains. No 0.995 promise or presumed French accuracy.

Continue xenc4 for one epoch, learning rate 8e-6, batch 128, fp16, max 112 tokens.
Seed 27. Use the existing 2.25M naturally retrieved hard/easy pairs, verifying
again that no S1 holdout identity enters training. 25% of examples receive a
name-only legal-form/presentation perturbation; preserve addresses. Negative
pairs that would become identical after augmentation revert to original text.
Hard cohorts receive weight 1.5 rather than changing labels or inventing truth.
Keep the base model's tokenizer and check the model has <8B parameters.

Score all v24 held-band pairs and full test band, joining by integer record keys
only for alignment, never as predictive features. Label-free inference exports
only keys and neural logits. Save exact checkpoint/tokenizer/environment/source
and input fingerprints. Mask final labels during later tuning calibration;
adoption requires complete-reference macro F0.5 gain on grouped tune and a
frozen final comparison. French generalization remains unlabelled.

The job does not produce an automatic leaderboard upload. A maximum three-hour
kernel run prevents an accidental overnight job. If resource quota or readiness
prevents a timely result, retain the validated existing files and final slot.
