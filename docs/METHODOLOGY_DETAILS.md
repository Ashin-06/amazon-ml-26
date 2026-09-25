<!-- Extended methodology (appendix to Documentation_template.md) -->
# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

A two-stage "blocking + learned matcher" pipeline built only on the supplied data. Candidate generation runs as a bulk
DuckDB set-join over hashed name, address and **name×address-word** keys, after a normalization step that includes a
**transliteration lexicon learned from the training labels** (native-script Indian names → English). A two-stage
LightGBM matcher (stage 2 sees out-of-fold stage-1 probabilities of competing candidates) plus a
"each target belongs to one S1" constraint and an F0.5-tuned threshold produce the final matches.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** 2.21M train S1 references vs 10.3M S2/S3 targets; 1.73M test S1 vs 9.97M targets (France only in test: 15% of test S1).
- **Labels:** 7.64M links, median 3 per S1, 5.6% of S1 are singletons (score 1.0 only if predicted empty).
  26% of targets belong to no S1: realistic distractors. **No target is linked to two S1s** and **no link crosses countries**; both facts are used.
- **Name noise:** legal-suffix swaps (Pvt/Private, LLC/Inc), typos and character swaps (C0nsulting, 5ervices), token reordering, @handles and
  domains (`@paulitawilson`, `adityaagro.com`), doubled tokens, invented DBA names at the same address, and
  **Indian names written in Devanagari/Bengali/Kannada/Tamil…** (28% of India S2 and 18% of India S3 train names are non-ASCII).
- **Address noise:** abbreviations (Rd/Road, R./Rue), state names vs codes (Ohio/OH, Maharashtra/MH/महाराष्ट्र),
  component reordering, missing parts (3–4% of targets have no address), zero-padded house numbers (`014889`),
  ordinal words (`Twentieth Ave`), literal `null`.
- **India names are highly generic** ("Smart Consultancy", "Gold Developers"): hundreds of distinct businesses share a
  name across the country, so the name alone cannot block. This drove the name×address-word keys.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted pair classifier + constrained assignment.
**Core Innovations:** (1) label-learned transliteration lexicon; (2) name×address-word composite blocking keys
(`pioneerinfotech|pune`), which keep generic names blockable within their city; (3) stage-2 matcher with competition
context and one-S1-per-target assignment.

---

## 3. Candidate Generation (Blocking)

- **Normalization** (`prep.py`): anyascii transliteration, lowercase, punctuation removal, canonical abbreviations
  (street types, legal forms, US/Indian states incl. native-script state names, French forms), zero-stripped numbers,
  ordinals→numbers, repeated-token collapse; `core` = name without legal forms.
- **Learned lexicon** (`lexicon.py`): for labelled train pairs whose target name was non-ASCII and has the same token count
  as the S1 name, tokens are aligned by position; `translit → english` pairs kept if seen ≥3× with ≥50% consistency
  (holdout S1s excluded). Full train gives 1,298 entries, e.g. `praivet→pvt`, `lksmi→laxmi`, `kmsltemsi→consultancy`.
- **Blocking keys** (`block.py`, all namespaced by country): name tokens, adjacent-token and whole-name concatenations
  (catch handles/domains/split words), name-token×address-word composites, address tokens and adjacent bigrams.
  Keys held by >300 targets are dropped. Score = Σ idf of shared keys (total, name, address).
- **Kept per S1:** top-40 by total ∪ top-20 by name ∪ top-20 by address score. Name ties prefer targets with an empty
  address (no conflicting evidence).
- **Candidate pairs:** train 106.4M (48 per S1); test [TBD from final run].
- **How true matches were protected (full train split, all 2.21M S1, full target pool):**

| Blocking version | Link recall | Oracle macro F0.5 (perfect matcher on candidates) |
|---|---|---|
| v1: tokens + concatenations | 0.9145 | 0.9656 |
| v4: + lexicon, composite keys, all-position address words, larger budget | **0.9664** | **0.9890** |

---

## 4. Matching Model

**Features (≈55 per pair):**
- Name: rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on normalized name; ratio, token-set,
  no-space ratio and exact match on core name; ratio and partial ratio on the raw joined name (handles vs full names).
- Address: ratio / token-sort / token-set / partial; digit-token Jaccard, common count, conflict flag, first-number
  equality, house-number edit similarity and relative numeric difference; empty-address flag.
- Frequency: how many targets / S1s share this exact core name or address in the country (generic names get less trust).
- Blocking context: total/name/address key scores, number of shared keys, ranks, candidate count, score gap and
  relative score within the S1, number of S1s that retrieved the target and this S1's rank among them.
- Cluster support: how many of the S1's candidates share the target's exact core name / address.
- Stage 2 only: the pair's out-of-fold stage-1 probability, its rank, max and sum within the S1, count above 0.5, and the
  best competing S1's probability for the same target (and the difference).

**Model type:** LightGBM (MIT license), binary logloss, 127 leaves, lr 0.05. Stage 1: two models trained on disjoint halves
of 300k sampled train S1s (out-of-fold probabilities for each other; averaged for everything else). Stage 2: one model on
the same S1s with context features.
**Decision / threshold:** keep pairs with p ≥ τ, then assign each target only to its highest-probability S1. τ maximizes
exact per-S1 macro F0.5 (singletons included) on a hash-selected holdout of ~100k S1s never used for fitting
or for the lexicon. We also tested expected-F0.5 subset selection and separate top-1 thresholds; neither beat a
single tuned τ.

---

## 5. Results & Error Analysis

| Run | Blocking oracle | Holdout macro F0.5 | Public LB |
|---|---|---|---|
| v1 (single-stage, basic blocking) | 0.9656 | 0.9422 | [TBD] |
| v4 (final pipeline) | 0.9890 | [TBD] | [TBD] |

- **Common false positives:** a different business at the same or next-door address (e.g. `1030` vs `1032 Sweet River Dr`);
  same generic name with an empty target address; "Shri"/"Sri"-prefixed sister companies at the same premises.
- **Common false negatives:** generic Indian names with an empty target address (many identical candidates);
  heavily corrupted transliterations the lexicon has not seen; invented DBA names whose address is also truncated.
- France has no labels; the pipeline is country-agnostic (country only namespaces keys), and French address and
  legal forms are normalized.

---

## 6. Conclusion

Most of the gain came from recall: at full scale, generic Indian names and native-script spellings made plain token
blocking lose 8.5% of true links. A lexicon learned from the labels and name×city keys recover most of them
(at 48 instead of 29 candidates per S1). The precision-weighted metric is then handled by a context-aware matcher, a one-S1-per-target
constraint and a threshold tuned directly on macro F0.5.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/fast_er/`: `to_parquet.py` (TSV→parquet), `prep.py`, `lexicon.py`, `block.py`, `model.py`;
drivers `run_stage1.py` (normalize + lexicon + block a split) and `run_stage3.py` (train, tune, predict, write both TSVs).
Exact commands are in `code/business_entity_resolution/README.md`. `kaggle/build_kernel.py` packs the same source into the
Kaggle CPU kernel used for the full runs. No external data, APIs or pretrained weights are used.

### B. Additional Results

Blocking recall on a realistic India slice (176k S1 against all 4.1M India targets):
basic keys 0.908 → + lexicon & composite keys 0.944 → + all address positions 0.948 → + larger budget 0.955 (oracle 0.985).
