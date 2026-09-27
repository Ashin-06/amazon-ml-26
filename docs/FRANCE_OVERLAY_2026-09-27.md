# France recall overlay (27 Sep 2026)

Public score of `v24stack_t077_fr0925`: **0.98291** (user-reported). That is +0.00063 over 0.982277, close to
the holdout prediction (+0.0007). India/US holdout gains transfer to the leaderboard.

## What the data shows

- **Train structure.** Exactly 3.46 true links per reference in both India and US. 97.7% of name-only targets
  are linked, and a target at the exact address of a reference alone there is 98% true.
- **Largest India/US loss.** Name-only targets whose name is shared by 2+ references. For example, identical
  "Roldan Pelican LLC" references in NC and UT can't be told apart. Holdout recall on name-only true links is 57%.
- **France.** 3.15 predicted links per reference, against 3.33–3.37 for India/US. France has many more decoy
  types at a reference's address than the US:
  - different multi-word names: 13× the US rate;
  - real-vocabulary word swaps ("Faune Club" → "Faune Ecole"): 6×;
  - single words: 3×.
  The models mostly reject these correctly.
- **French true-match patterns the models reject.** In India/US these patterns are 87–99.9% true:

  | Pattern (France example) | France accepted | India/US accepted |
  |---|---|---|
  | Same address, generic suffix ("X SARL & Fils", "X Groupe", "& Associés", "Développement") | 58.6% | 99.8–99.9% |
  | Lone address, made-up rebrand token ("onyxpyra") | 54% | 83–90% |
  | Name-only target with a unique core name | 90.5% | 96–97% |

  The generic suffix is the analogue of India/US "+ Services / Center / Partners".
- **French suffix classes.** Suffix words split cleanly by how often they appear at the reference's own address:
  - *true-style* (4–20%): fils, associes, frs, groupe, developpement, svc;
  - *decoy-style* (0.03%, like India/US group/ent/ind/sol/bros/intl): holding, participations, distribution, intl.

  The models already reject the decoy-style words (0 accepted).
- **Legal-form swaps.** "X LLC" vs "X Inc" at another address is only 31% true in US (India 91%), so the overlay
  never adds legal-form swaps.
- **Checked and not a loss.** Dotted French legal forms (S.A.R.L. → "s a r l") pollute the core for 68.5k French
  targets, but those targets are assigned at the normal rate. French accents normalize correctly.

## Files (validator + checker exit 0; only French rows changed; nothing removed)

| File | Rules | Links added |
|---|---|---|
| `submissions/v24stackFRa_matching_results.tsv` | F1 same-address suffix (7,182), F2 lone-address rebrand token (2,386), F4 unique-core name-only (1,037) | 10,605 |
| `submissions/v24stackFRb_matching_results.tsv` | FRa + F3 different-address identical content, legal not swapped, score ≥ 0.6 (2,860) | 13,465 |

- Code: `work/fr_overlay.py`, bundled in `submissions/v24stackFR_code.zip`.
- Every added pair is in the exported v24 candidate set, and no target that is already owned is reassigned.

## Limits

- France has no labels, so precision comes from the India/US analogues.
- Expected leaderboard effect is small: about +0.0003 for FRa, and FRb adds little over FRa with more risk.
- 0.99 is not reachable from these: the France recall gap outside these patterns is either unretrieved or below
  score 0.3.
