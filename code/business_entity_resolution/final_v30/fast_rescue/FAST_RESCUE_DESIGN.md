# Fast rescue matcher experiment

Written before training, 27 September evening. One submission remains.
Reuse the naturally generated, first-word/house-number rescue candidates from
v26's independent audit (19,805 held pairs). Do not expand retrieval, manufacture
labels, alter v27, or restart existing jobs.

Train two small classifiers: regularized logistic regression and a 200-tree
LightGBM (7 leaves, max depth 3, min child 50, learning rate .04, L2=20).
Use only the tuning S1 identities in three grouped folds. Mask all final labels
before creating feature/label frames. Train on the deployment cohort: targets
unowned by the frozen v25 OOF .74 decisions. Features use only supplied names,
addresses, missingness, digit agreement and whole-pool core-name frequencies.
No identities, row order or country labels enter the classifier.

Choose the probability cutoff on complete-reference tuning macro F0.5, retaining
empty references and target ownership. Compare the complete pipeline against
v25 plus the original fixed 1.88 rescue. Advance only with >=.0001 gain and a
positive descriptive paired lower bound. Lock the selected model/cutoff before
one historically reused-final check. Reject a nonpositive final result; do not
retune on final. French accuracy remains unlabelled and must not be claimed.

One CPU thread and a bounded database cache; measure actual runtime. This is a
small calibrated rescue stage, not a full retrain or a 0.995 promise. If it
passes, stream only existing rescue candidates at inference; exports must still
include exactly the full matcher input candidates.
