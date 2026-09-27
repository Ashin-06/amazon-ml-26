# Response to confirmed public score 0.982910

The user confirms `v24stack_t077_fr0925_matching_results.tsv` scored 0.982910;
fewer than four uploads remain today. Exact remaining count is unknown. This
is +0.000633 over v20's 0.982277, not a 0.99 result. No automatic uploads.

Before implementation: test stronger within-reference sibling evidence on the
existing natural v24 candidates. Original stage-two anchors use first-stage
scores >= .3 and at most four neighbors. The proposed experiment uses up to six
other records with final base score >= .995 (neural logit >=4 when available),
and a second route >= .98 (neural logit >=3 when available). These fixed rules
do not use tuning labels or a previously fitted correction model. Exclude the
query target itself. Similarities must not treat two empty addresses as evidence.

Use existing frozen 63 features plus sibling similarities/counts. Compare two
predeclared regularized LightGBM arms in the same three S1-grouped tune folds,
against the saved v24stack OOF predictions. All references and empty lists remain
in exact macro F0.5. Labels for final remain masked during selection. No numeric
ID or row-order features; no outside lookup or truth-injected candidate.

Only proceed to a locked, historically reused final evaluation when tuning gain
is >= .0002 and the descriptive paired 95% lower bound is positive. Otherwise
reject without spending a leaderboard slot. A tuning gain does not promise a
test gain. Existing files/jobs are unchanged, one CPU thread, bounded memory.

Functions: `context_features` builds label-free sibling features from normalized
provided targets; `fit` trains fixed-complexity models; `pilot` cross-fits and
selects on tune; `final` locks selected model/cutoff before viewing final labels.
All results and source snapshots use unique output directories.
