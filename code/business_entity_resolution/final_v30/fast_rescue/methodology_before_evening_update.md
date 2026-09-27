<!-- DRAFT: fill team fields and the final selected run's numbers before submission -->
# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name] · **Team Members:** [List all team members] · **Submission Date:** [Date]

## 1. Executive Summary
Blocking + learned matcher, built only on the supplied data. Records are normalized (transliteration, abbreviation
and state canonicalization, alias/phone cleanup) and a **transliteration lexicon learned from the training labels**
maps native-script Indian spellings to English. Candidates come from a bulk DuckDB join over hashed name, address and
**name×address-word** keys. A **two-stage LightGBM** matcher scores pairs (stage 2 sees how each pair compares with
competing candidates), each target is given to at most one reference, and the threshold maximizes macro F0.5.

## 2. Methodology
**Problem analysis.** 2.2M train references vs 10.3M targets (test: 1.73M vs 10.0M, France only in test). Links average
3.5 per reference; 5.6% of references have none; no target belongs to two references and no link crosses countries.
Main noise: legal-suffix/typo/reordering name variants, "Brand DBA Real Name" aliases (2.4% of Source-3 names),
appended phone numbers, Indian names in Devanagari/Bengali/Kannada/Tamil (28% of India S2 names), abbreviated or
reordered addresses, zero-padded house numbers, 3–4% missing target addresses, and very generic Indian names shared
by hundreds of businesses. **Approach type:** blocking + two-stage gradient-boosted classifier + one-owner assignment.

## 3. Candidate Generation (Blocking)
Keys per record, namespaced by country (open set, France included): name tokens, adjacent/whole-name concatenations
(catch @handles/domains), whole core name, **name-token × address-word** composites ("pioneerinfotech|pune": keeps
generic names blockable within their city), address tokens and bigrams. Keys held by >300 targets are dropped (1,000
for whole names). Pairs are scored by summed idf; per reference we keep top-40 overall ∪ top-20 by name ∪ top-20 by
address (≈49 candidates). **Full-train link recall 97.05%, perfect-matcher ceiling F0.5 0.9905.**

## 4. Matching Model
**Features (≈50):** RapidFuzz ratio / token-sort / token-set / partial / Jaro-Winkler on name, core name, raw joined
name and address; digit-token agreement and house-number similarity; missing-address flags; how many records share
the exact name/address; blocking scores and ranks; cluster support (candidates sharing the same name/address).
**Model:** LightGBM (MIT). Stage 1: two models on disjoint reference halves give out-of-fold probabilities. Stage 2 adds
context: the pair's rank and share within its reference and within its source, and the best competing reference's
probability for the same target. **Decision:** probability ≥ τ, then each target goes to its single best reference
(ties: lowest id). τ is tuned on half of a held-out reference set; F0.5 is reported on the other, untouched half.

## 5. Experiments & Results
Holdout = ~100k training references never used for fitting or for the lexicon. "Final-half" = threshold tuned on one
half, scored on the other (same protocol for all rows in the last column).

| Run | Change | Candidate recall / ceiling | Holdout F0.5 (tuned on hold) | Final-half F0.5 |
|---|---|---|---|---|
| v1 | token blocking + single LightGBM | 0.9145 / 0.9656 | 0.9422 | – |
| v2 | + learned lexicon, frequency features, two-stage context | 0.9203 / 0.9680 | 0.9535 | – |
| v3 | + number/ordinal/French normalization, short-token and whole-name keys | 0.9329 / 0.9735 | 0.9578 | – |
| v4 | + name×address keys, larger candidate budget, house-number, raw-name and cluster features | 0.9664 / 0.9890 | 0.9762 | 0.9758 |
| v10 | + short-name composites, whole-name cap, alias & phone cleanup, per-source context, per-reference weights | 0.9705 / 0.9905 | [TBD] | [TBD] |

Also tested: expected-F0.5 subset selection and separate thresholds per country or for the top candidate (no gain);
per-reference sample weights (small gain on a pilot). **Errors:** misses dominate (≈9 missed links per false link);
64% of missed links were never retrieved; nearly half of those have an empty target address (often a generic name). False links are mainly
a different business at the same or neighboring address. France has no labels: its predicted match rate (5.8% empty,
3.4 links/reference) matches US/India.

## 6. Conclusion
Recall was the main lever: at full scale plain token blocking lost 8.5% of true links on generic and native-script
Indian names; a label-learned lexicon and name×city keys recovered most of them. A context-aware second stage, the
one-owner constraint and a threshold tuned directly on macro F0.5 then handle the precision-weighted metric.
No external data, APIs or pretrained weights are used; the model is LightGBM (MIT).

**Code:** `code/business_entity_resolution/` (README has exact commands); full details in `METHODOLOGY_DETAILS.md` (same zip).
