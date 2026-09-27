from pathlib import Path
import hashlib,json
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('D:/Amazon ML 26');H=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
a=read(H/'rescue_followup_v1/audit.json');b=read(H/'second_word_route_v1/audit.json');c=read(H/'second_word_adaptation_v1/audit.json');e=read(H/'v28_tuning_error_budget.json');release=read(H/'fast_rescue_release.json')
assert a['selected'] is None and not b['eligible'] and not c['eligible']
assert sha(Path(release['matching']))==release['matching_sha256']
assert sha(Path(release['review_zip']))==release['review_zip_sha256']
loss=1-e['score'];pct=[100*e[k]/loss for k in ['rejected_retrieved_true_loss','unretrieved_true_loss','false_positive_loss']]
report=f'''# Further model and retrieval tests, 27 September 2026

## Decision

Five further alternatives were evaluated on tuning identities only. None passed its predeclared adoption gate. Keep **v28fast_rescue_matching_results.tsv** unchanged. Its previously measured reused-final India/US score remains **0.989582557**; its public score is unknown. Confirmed public best remains v26, **0.98417**. One leaderboard attempt remains. No upload, final-label evaluation or production replacement occurred during these follow-ups.

## Results — tuning only, not new final scores

Frozen v28 tuning baseline: **{a['baseline_v28_tune']:.12f}** over 50,141 references with exact macro F0.5, including empty references.

| Alternative | Tuning score | Difference from v28 | Decision |
|---|---:|---:|---|
| Equal linear / seven-leaf tree blend | {a['results']['linear_tree_blend']['score']:.12f} | +0.000044823 | Reject: small gain, interval crosses zero |
| Explicit name/address/number interactions | {a['results']['interactions']['score']:.12f} | -0.000016434 | Reject: both countries regress |
| Equal linear / wider-tree blend | {a['results']['linear_wider_tree_blend']['score']:.12f} | +0.000037036 | Reject: small gain, interval crosses zero |
| New second-word + house-number retrieval, frozen v28 matcher | {b['score']:.12f} | -0.000101099 | Reject: false matches, India regression |
| Same new route with grouped retraining | {c['score']:.12f} | +0.000046113 | Reject: small gain, interval crosses zero |

The three-model comparison took {a['seconds']:.3f} seconds. The new-route pilot took {b['seconds']:.3f} seconds; adaptation took {c['seconds']:.3f} seconds, all bounded CPU work. The route examined 4,613,506 raw pairs, yielding 13,094 new candidates and 64 true pairs. The original model accepted 43, of which 24 were true. Retraining accepted 38, of which 31 were true, but its paired descriptive interval was [-0.000007012, +0.000099239]. These are adaptive tuning results, not independent confirmation.

## What is actually limiting the current matcher

The tuning-only error decomposition accounts exactly for 1 minus the v28 score. It removes false positives first, then accepts all retrieved true pairs, then restores unretrieved true pairs. This decomposition order matters; these are oracle accounting steps, not deployable rules.

| Source of score loss | Absolute macro loss | Share of remaining loss |
|---|---:|---:|
| Rejected true pairs already in candidates | {e['rejected_retrieved_true_loss']:.9f} | {pct[0]:.1f}% |
| True pairs never retrieved | {e['unretrieved_true_loss']:.9f} | {pct[1]:.1f}% |
| False accepted pairs | {e['false_positive_loss']:.9f} | {pct[2]:.1f}% |

There are 175 false-positive links and 5,269 false-negative links in this tuning view, including 1,875 unretrieved true links. The candidate oracle is {e['candidate_oracle']:.9f}; it assumes perfect decisions and is **not** an attainable model score or a public-test ceiling. Even eliminating every retrieval miss without changing the other decisions is not equivalent to achieving that oracle. Reaching 0.995 from this tuning score would require recovering roughly half the remaining macro loss. These small blend/route changes do not demonstrate that.

The next substantial experiment should target discriminating currently rejected candidates, using supplied text/context and grouped evaluation. An architecture change needs measured evidence that it separates those positives from hard negatives; extra compute or threshold changes alone do not establish this. French accuracy cannot be established without labels. Repeatedly searching the reused final split would give misleading confidence and was avoided in these follow-ups.

## Reproduction and integrity

All isolated artifacts are under `artifacts/after-0982910-20260927/`:

- `RESCUE_FOLLOWUP_DESIGN.md`, `rescue_followup.py`, `rescue_followup_v1/audit.json`.
- `SECOND_WORD_ROUTE_DESIGN.md`, `second_word_route.py`, `second_word_route_v1/audit.json`.
- `ROUTE_ADAPTATION_DESIGN.md`, `adapt_second_word.py`, `second_word_adaptation_v1/audit.json`.
- `v28_tuning_error_budget.py`, `v28_tuning_error_budget.json`.

Scripts use the frozen feature arrays, models and matching source named in the prior release; unique output directories deliberately refuse overwrites. Grouped folds and original tuning masks are retained. No test labels, external entity data, final-label tuning or known-positive injection into retrieval were used. Input/output IDs and row positions are not model features.

Published v28 matching SHA-256: `{release['matching_sha256']}`. Code/model review ZIP SHA-256: `{release['review_zip_sha256']}`. Both were rechecked unchanged. No newly trained model was deployed. Existing full-file validation remains applicable to the unchanged matching file. The ZIP remains a review archive, not a complete standalone Round-2 package.
'''
rp=ROOT/'docs/V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md';rp.write_text(report,encoding='utf-8')
note='\n\n## Further attempts after release\n\nFive additional model/retrieval variants failed the tuning adoption checks. v28 remains unchanged; no new final evaluation or leaderboard upload occurred. See [follow-up results](V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md), including the current error budget.\n'
p=ROOT/'docs/FAST_RESCUE_TRAINING_2026-09-27.md';p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
for rel in ['START_HERE.md','docs/IMPLEMENTATION_STATUS.md','docs/RUNBOOK.md','code/business_entity_resolution/README.md']:
 p=ROOT/rel;s=p.read_text(encoding='utf-8');first,rest=s.split('\n',1)
 link='docs/V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md' if rel=='START_HERE.md' else ('../../docs/V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md' if rel.startswith('code/') else 'V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md')
 p.write_text(first+f'\n\n**Follow-up experiments:** five additional variants tested; none passed adoption. v28 remains unchanged, with one upload unused. [Results and error budget]({link}).\n'+rest,encoding='utf-8')
state=read(ROOT/'submissions/current_release_status.json');state['as_of_ist']=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat();state['latest_followup']={'report':str(rp.relative_to(ROOT)),'alternatives_tested':5,'adopted':0,'final_labels_opened':False,'published_files_unchanged':True}
for p in [ROOT/'submissions/current_release_status.json',H/'CURRENT_STATE.json']:p.write_text(json.dumps(state,indent=2),encoding='utf-8')
print(json.dumps(state['latest_followup'],indent=2))
if __name__=='__main__':pass
