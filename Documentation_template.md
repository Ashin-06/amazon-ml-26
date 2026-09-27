# ML Challenge 2026: Business Entity Resolution

**Team:** Cardinality  
**Members:** Ashin Murugan, Cherin Yacoob, Neha R, Harisankar G  
**Date:** 27 September 2026  
**Final selected submission:** `v30_frxe2_matching_results.tsv`, public score **0.984718** (best public; used for the
private leaderboard). Round-2 standalone reproduction packaging is still to be completed.

v30 combines the v28fast_rescue India/US decisions with a corrected French cross-encoder input normalization.
v28fast_rescue uses a standardized logistic-regression rescue matcher (C=0.1, cutoff 0.69) on supplied name/address
features; it raised the reused India/US final split from 0.989349619 to 0.989582557. See
docs/FAST_RESCUE_TRAINING_2026-09-27.md and docs/FRANCE_OVERLAY_2026-09-27.md.

## Executive summary

We match every Source-1 record to zero or more Source-2/Source-3 records using
only supplied business names, addresses and country fields. Bounded lexical
blocking feeds two-stage boosted-tree matching, multilingual neural pair
rescoring and a small correction model. Every target has at most one owner;
a reference may own multiple targets. External business registries, geocoding
lookups and hosted inference APIs are not used.

## Candidate generation

Normalization uses offline transliteration, punctuation/number cleanup and
multiple raw, normalized and legal-form-reduced name/address views. A lexicon
learned from permitted training identities supports Indic-script variations.
Country-partitioned name/address token and composite-key routes build candidate
sets against the full supplied target pool, with route-specific frequency and
ranking limits. v24 also includes retrieval for names with missing addresses.

The v26 rescue route adds previously unretrieved pairs sharing the first core
name word and the leading address number. Cheap name edit and address token-set
similarities prune the route at a summed score of 1.5; a fixed 1.88 matcher
threshold accepts the best unowned target links. These two thresholds have
different roles. All retained route candidates, not merely accepted matches,
are added to the exported candidate file. v27 extends this route to France.

## Models and decisions

The frozen v24/v20 pipelines use GPU XGBoost stage-one models trained on
separate S1 groups and a second stage using candidate/competitor context.
Multilingual-e5-small/base-derived cross-encoders provide pair logits on the
uncertain score band. Pretrained weights are fine-tuned on supplied pairs;
the selected checkpoints and their licensing/provenance must accompany the
Round-2 reproduction package.

The v25 correction is a 600-tree LightGBM model (15 leaves, maximum depth 5),
trained on tuning identities with 63 numeric features: base/neural scores,
text agreement, missing fields, address digits, and full-pool name frequency.
Its fixed India/US cutoff is 0.74. IDs and row order are not predictive features.
France uses the frozen French-normalized neural blend at 0.925 and documented
rules; French labels are absent, so local accuracy is not claimed.

v26 includes same-address suffix/rebrand and unique-core French rules. v27 adds
7,096 different-address French name-pattern matches and 2,099 French rescue
matches, preserving v26's India/US rows. These are unlabelled French
extensions, justified by India/US analogues (87-99.9% true), not French labels.

v30 corrects the French cross-encoder input normalization. The earlier table
mapped French generic suffixes onto English words that mark different businesses
in the training data ("& Fils" to "& Sons", "Groupe" to "Group", "Ets" to
"Enterprises"). Measured on the unlabeled test data, & Fils, Associés, Frères,
Groupe and Développement occur at the reference's own address 4-20% of the time,
like the English true-match suffixes Services/Center/Partners (6-18%). Holding,
Participations and Distribution occur there 0.03% of the time, like the English
decoy suffixes. The corrected table maps the first group to Services/Partners
and the second to Group. The same xenc4 cross-encoder rescored the 757,114 French
band pairs, and the unchanged blend reproduced the previous French scores to
1e-7 before the substitution. v30 adds 14,111 French links for targets that
v28fast_rescue left unassigned and removes none. Public score went from 0.98417
(v26) to 0.984718.

## Evaluation and results

The objective is per-reference macro F0.5: `5TP / (true_count + 4predicted_count)`,
with empty/empty equal to one. All references enter its denominator. Base
training S1 identities are excluded from tune/final; correction selection uses
S1-grouped tuning folds. The final set has been reused during development and
is not a fresh untouched test. Held-pool ownership is smaller than full-test
ownership, which limits comparability. No known positives are injected into
evaluation retrieval.

| Measured result | Score |
|---|---:|
| v24stack public, user-confirmed | 0.982910 |
| v26 public, user-confirmed | 0.98417 |
| **v30 public, user-confirmed (final selected)** | **0.984718** |
| v26 joint final India/US audit, 50,689 references | 0.989350 |
| v26 final India / US | 0.989011 / 0.989577 |
| v28fast_rescue / v30 joint final India/US | **0.989583** |
| Fixed rescue gain over the same v25 baseline | +0.000420 |

The fixed rescue accepts 234 additional final pairs, 218 true. v27 had the same
India/US decisions as v26 and was not uploaded. A later neural robustness
experiment (xenc7) and its US-only variant failed adoption checks and are not
selected. Rules tested and rejected on labelled data: same-name + house-number
rescue without a text score, sibling name propagation, rarity-gated and
name-only thresholds, a winner-margin rule and a link-count prior. All gained
at most +0.0001, and some lost up to -0.027.
Candidate oracle scores are upper bounds on their labeled populations, not
trained scores or test forecasts. No 0.995 result is claimed.

## Reproduction and submission status

Both outputs preserve all 1,732,544 references and empty lists. The final v30
file has 5,807,702 links, 0 targets with two owners and 0 cross-country links
(SHA-256 prefix 62228fce34a43fec). The supplied validator passes on the matching
file. The unmodified independent checker and a streaming candidate check confirm
original IDs, format, ownership and containment in
`work/comp/v25/candidate_pairs_v27.tsv` (181,275,851 pairs). The supplied
validator runs out of RAM on the 2.3 GB candidate file on this machine, so the
candidate file was verified by those two independent checks instead. Original
inputs and the validator are unchanged.

The organizer's 27 September clarification requires the matching TSV tonight;
the final ZIP is for Round 2. The current code ZIP is a review bundle, not yet
a complete standalone archive. For the selected v30 (submissions/v30_code.zip is the review bundle),
package its frozen source, all required weights/tokenizers, configuration,
licenses, environment and both TSVs. Preserve hashes and verify reproduction.

See [evening findings](docs/EVENING_098417_FINAL_REVIEW_2026-09-27.md) and
[runbook](docs/RUNBOOK.md) for evidence, exact paths and outstanding limitations.
