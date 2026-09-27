# Research: what would move 0.9847 toward 0.99+ (27 Sep 2026)

Final public score: **0.984718** (v30). This document ranks new ideas. Each is tied to a measured loss bucket and
comes with a validation plan, because France has no labels.

## Where the loss is (measured)

| Bucket | Size | Evidence |
|---|---|---|
| **France** | ~0.005 of public score | India/US holdout ~0.9896; implied France ~0.955-0.96. French predictions have 3.28 links/reference against 3.35-3.38 for India/US; train truth is 3.46 in both countries |
| India/US true links never retrieved | 0.0042 holdout; larger on test | test has 25% more targets per reference (5.8 vs 4.7); per reference, the rescue route found 1.5x more links on test than on the holdout |
| Name-only records whose name is shared by 2+ references | ~0.003 | no content signal picks the owner (raw spelling, case, source, link counts: 55-58%) |
| False positives | 0.0006 | singleton accuracy 99.76% |

## 1. France: domain adaptation (largest gap)

1. **Pseudo-label self-training on French test pairs** (organizer row 865 permits it; see the DADER / DAME /
   self-training literature).
   - **Positives:** French pairs from rule families with >= 90% India/US analog precision (F1-F5, the French
     rescue route) and scores >= 0.995.
   - **Negatives:** French decoy families. These are legal-form swaps at another address (US analog 31%),
     same-address real-word swaps (6x over-represented in France), decoy suffixes (Holding, Participations,
     Distribution) and low scores.
   - **Training:** fine-tune the cross-encoder, or a France-specific stage-2 model, on these pairs. Iterate with
     rising confidence thresholds.
   - **Guard against confirmation bias:** train on some rule families and check agreement on held-out families.
2. **French-ified augmentation of labelled India/US pairs** (Ditto-style augmentation). Rewrite pairs with French
   forms under measured semantics, not literal translation. Use one legal-form mapping per pair (SARL/EURL/SAS
   consistent on both sides), true-style suffixes (& Fils, Groupe, Associés, Développement, Frères), decoy-style
   suffixes (Holding, Participations, Distribution), French street types and accent noise (Ç, à, ï). The prepared
   `work/train_xenc6_local.py` must be corrected first. It maps English "Group" (a decoy) to "Groupe" (true-style)
   and picks legal forms per side.
3. **Domain-adaptive MLM pretraining** of multilingual-e5 on the unlabeled French name/address strings before
   fine-tuning.
4. **Automatic token-class audit (would have found the v30 bug on day 1).** For every token that targets add or
   drop relative to their reference, measure its co-location rate at the reference's own address. True-match
   words run 4-20%, decoy words about 0.03%. Feed the class as a feature and use it to check every normalization
   table in every country.

## 2. Retrieval (India/US, larger on the denser test pool)

5. **Dense/ANN blocking.** Use a contrastive bi-encoder over name and address (SC-Block, UniBlocker, DeepBlocker,
   Sudowoodo) with FAISS top-k, unioned with the lexical routes. Target the typo, rebrand and truncated-address
   misses; the recall-vs-candidates curve can be measured on the holdout.
6. **Density-adaptive caps.** Scale per-reference top-k, or use score-margin cut-offs, by key frequency instead
   of the fixed 80/40/40. Also build a validation pool with test-like decoy density, so caps and cutoffs are tuned
   for the 5.8 targets/reference seen in test.
7. **More learned rescue routes.** Route A (first word + house number, +0.00043) and the learned rescue
   (+0.00023) worked. Next are fuzzy house numbers (1121 vs 1121A), transliterated names with Indic-script
   targets and street-typo keys. Each needs a route-specific matcher trained on tune pairs (rules alone were
   3-47% precise).

## 3. Decisions and structure

8. **Target-first clustering, then linking** (collective, multipartite ER). First cluster S2/S3 duplicates, then
   assign whole clusters one-to-one to references. The exact-duplicate test found little (+0.00001), so this
   needs fuzzy duplicate clusters.
9. **Calibrated per-country decisions.** Fit France thresholds and blend weights with the self-training labels
   (idea 1) instead of fixing 0.925 by hand.

## Tested and rejected today (do not repeat)

- **Sibling name propagation:** 9.6% precise, -0.027.
- **Same-name + house-number rules without a model:** 3% precise.
- **Rarity, margin, name-only thresholds and a link-count prior:** at most +0.0001.
- **Name-competition features:** +0.000006.
- **Bigger correction models:** saturated at v25.
- **Global French cutoff 0.60:** neutral on the leaderboard.
- **Business-centre address filter:** 261 French references affected.

## Sources

- DADER: domain adaptation for deep ER — https://www.researchgate.net/publication/361245800_Domain_Adaptation_for_Deep_Entity_Resolution
- DAME: domain adaptation for matching entities — https://dl.acm.org/doi/abs/10.1145/3488560.3498486
- Ditto: deep entity matching with pre-trained LMs — https://arxiv.org/abs/2004.00584
- Universal dense blocking (UniBlocker) — https://arxiv.org/html/2404.14831
- SC-Block: supervised contrastive blocking — https://arxiv.org/pdf/2303.03132
- One-to-one matching algorithms for ER (VLDB J.) — https://link.springer.com/article/10.1007/s00778-023-00791-3
- Multipartite entity resolution — https://arxiv.org/pdf/2112.03346
- Self-training survey — https://www.sciencedirect.com/science/article/pii/S0925231224016758
