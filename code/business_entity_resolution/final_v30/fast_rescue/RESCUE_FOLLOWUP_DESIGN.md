# Further rescue experiments, fixed before execution

Keep the published v28 output intact. Use only the existing natural rescue pairs and masked tuning labels; no new candidate route or French decision changes in this pilot. Reuse frozen numeric features and grouped folds, and compare with v28's OOF predictions at its frozen 0.69 cutoff. Complete-reference macro F0.5 and target ownership remain mandatory.

Three predeclared alternatives: (1) standardized logistic regression C=0.1 with explicit name/address/number interaction features, (2) equal average of the existing linear and seven-leaf tree OOF probabilities, and (3) equal average of existing linear and a new 15-leaf LightGBM (200 trees, depth4, minchild75, L2=30, learning rate .04). No ID or row-order features. Limit compute to one CPU thread.

Select cutoff using tuning-only OOF predictions over .50:.01:.99 and .995. Adoption gate against v28: delta at least .00005, positive lower paired descriptive 95% interval, no country regression. Freeze at most one winner before any further final evaluation. If no arm passes, retain v28 and record all failures; do not search final labels for another cutoff. Existing final split has been reused, so any later check is a regression check rather than a new unbiased estimate.
