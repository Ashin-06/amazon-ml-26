# xenc7 evaluation contract

Written before downloading or inspecting xenc7 held scores. Use the same
three S1 tuning folds as the saved v24stack. The complete existing final set
has been used previously and will not be presented as an untouched test.

1. Require cloud completion, no held training overlap, expected input/output
   hashes, and exact one-to-one coverage of the v24 neural band. Never fill
   missing logits or take duplicate keys arbitrarily.
2. Read only masked labels and tuning reference totals for selection. Keep
   all 50,141 tune references in the denominator, including empty predictions.
3. Compare two predefined arms with the frozen v24stack OOF at cutoff 0.77:
   logistic regression on old/new neural logits and their stage-2 interactions;
   the same fixed-complexity 320-tree corrector with seven new neural features.
   Use the existing cutoff grid. No early stopping or final-label selection.
4. Advancement requires tune gain >= 0.0001, a positive lower descriptive
   paired 95% bound, and neither labeled country regressing by >0.00005.
   Multiple-experiment uncertainty and the held-pool ownership limit remain.
5. Fit the selected arm using tuning identities only. Freeze model, exact
   features, cutoff, inputs and source hashes before one final comparison.
   Final adoption requires positive gain and a positive lower paired bound;
   otherwise retain v24stack. Do not try more thresholds on final.
6. India/US deployment and France generalization are separate questions.
   Preserve the existing French output for a clean India/US improvement.
   French results have no labeled local accuracy. Any France variant must be
   separately identified; no assumed French gain enters the adoption check.
7. No leaderboard upload is automated. Exactly two slots remain. Candidate
   scope stays unchanged; all accepted pairs must be exported candidates.

19:22 IST amendment, before reading xenc7 evaluation metrics: the user now
reports v26 = 0.98417 and one slot remains. Include the existing 600-tree v25
corrector at its fixed .74 cutoff as an additional control. Reproduce its OOF
with non-tune competitors scored, not zeroed; final ownership includes both
held halves. Also require a positive paired lower bound against this control.
If xenc7 qualifies, evaluate the selected model plus the fixed 1.88 rescue
route jointly; do not add gains from different baselines arithmetically.
