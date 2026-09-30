# Language and S1–S2/S3 link audit — 28 September 2026

This audit uses the supplied training ground truth and the train/test source tables. The machine-readable scan is in `artifacts/final-package-20260928/language_links_audit.json`; the read-only scripts used to produce and cross-check it are beside that file. The packaged methodology was updated from these measurements. The frozen v30 model, candidate set and leaderboard matching file were not changed.

## Link structure

The training set contains 2,206,821 S1 references: 883,188 India and 1,323,633 US. There are 3,059,843 links in India and 4,578,522 in the US, or 7,638,365 in total. Splitting every nonempty `matched_entity_ids` field and counting distinct target IDs gave the same total in each country; there were no malformed linked IDs and no target linked to two references.

| Country/source | Target rows | Linked rows | Linked share | References with >=1 link |
| --- | ---: | ---: | ---: | ---: |
| India S2 | 2,017,799 | 1,480,545 | 73.4% | 768,187 (87.0%) |
| India S3 | 2,115,547 | 1,579,298 | 74.7% | 776,635 (87.9%) |
| US S2 | 3,016,817 | 2,213,074 | 73.4% | 1,150,889 (86.9%) |
| US S3 | 3,170,056 | 2,365,448 | 74.6% | 1,163,910 (87.9%) |

Both countries have the same four leading patterns of S2/S3 links per S1 reference: (1,1), (1,2), (2,1), (2,2). There are 49,351 India and 73,896 US references with no linked targets. Mean linked records per reference are 1.67 from S2 and 1.79 from S3, rounded. These are training-label measurements; France has no labels in the supplied training set.

## Scripts and normalization

The India S2/S3 train and test names and addresses contain characters in all nine scanned Indic Unicode blocks: Devanagari, Kannada, Bengali, Gujarati, Gurmukhi, Odia, Tamil, Telugu and Malayalam. For example, the train S2 names contain 269,424 Devanagari-script and 37,211 Kannada-script rows; train S3 names contain 158,003 and 21,995 respectively. A row may contain more than one script, so per-script counts are not additive. The complete per-source, per-field counts are in the JSON audit.

The packaged `fast_er.prep.norm` transliterates from non-Latin text into ASCII with `anyascii`, then lowercases and normalizes tokens. Direct checks with the packaged code gave `राम मार्केटिंग प्राइवेट लिमिटेड` -> `ram marketimg praivet ltd`, `ಕನ್ನಡ` -> `knnd`, and `École Français, Nord` with French address mapping -> `ecole francais frhdf`. The training-only lexicon in `fast_er.lexicon` corrects recurring transliterations; the phonetic skeleton supports remaining variants. These transformations are normalization, not reliable translation of the meaning of an unseen business name.

French test names and addresses include non-ASCII characters. The same normalizer folds accents, while known French region/departement aliases are harmonized as whole address components. No French precision or recall can be calculated from the supplied labels.

## Scope of package change

Only `Documentation_template.md` and the regenerated package manifest are changed inside the ZIP. The selected matching and candidate files retain their SHA-256 hashes: `62228fce34a43fec5c4ca7a21efca1b5aee67fa47216bdcb278c52b9455ebc23` and `4162dba9b0881649032b608c1e50cbf3b740de9d1ee3b7e9283391366f66700f`. Full clean-room pipeline reproduction remains unverified; archive entry verification does not prove a full replay.
