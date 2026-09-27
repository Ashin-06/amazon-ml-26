# US-only continuation of the xenc7 experiment

27 September, 19:32 IST. The original all-country adoption gate failed.
The tuning US slice improved while India regressed. This is an additional,
exploratory country-specific selection made on tuning data, not a rescue of
a failed final result: xenc7 final labels have not been evaluated yet.

Keep v25 for India. Use the existing neural_tree arm only for US, keeping its
tune-selected cutoff 0.74 and fixed 320-tree settings. Require US tuning gain
>= 0.0001 and a positive paired lower 95% bound versus the 600-tree v25 control.
Fit on all tuning identities as before; freeze source, weights and cutoff.
Compare once on the historically reused final split, with both held halves
competing together. Include the same fixed 1.88 rescue route for both pipelines.
Deployment requires positive US final gain with positive paired lower bound,
and positive final gain for the complete combined pipeline. If it fails, stop
this branch. No further final thresholds or subgroups. France stays unchanged
from the separately selected French file and remains locally unlabelled.
