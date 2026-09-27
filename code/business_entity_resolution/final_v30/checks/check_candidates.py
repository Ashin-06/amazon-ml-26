"""Streaming check of candidate_pairs.tsv against the official format rules (low memory):
header, exactly 2 tab fields, one row per test S1 (no extras/duplicates), S2-/S3- IDs only, no duplicate ID
within a list, no quotes/CR/BOM; and every match in MATCHING is a candidate of the same reference."""
import sys, csv
cand, match, s1 = sys.argv[1:4]
refs = [r['entity_id'] for r in csv.DictReader(open(s1, encoding='utf-8'), delimiter='\t', quoting=csv.QUOTE_NONE)]
want = set(refs); m = {r[0]: [x for x in r[1].split(',') if x] for r in csv.reader(open(match, encoding='utf-8'), delimiter='\t', quoting=csv.QUOTE_NONE)}
bad = 0; seen = set(); pairs = 0
with open(cand, 'rb') as fb:
    if fb.read(3) == b'\xef\xbb\xbf': print('BOM'); bad += 1
with open(cand, encoding='utf-8', newline='') as f:
    head = f.readline()
    if head != 'source1_entity_id\tcandidate_entity_ids\n': print('header', repr(head)); bad += 1
    for n, line in enumerate(f, 2):
        if '\r' in line or '"' in line: bad += 1; print('CR/quote line', n); break
        parts = line.rstrip('\n').split('\t')
        if len(parts) != 2: bad += 1; print('fields line', n); break
        s, ids = parts[0], [x for x in parts[1].split(',') if x] if parts[1] else []
        if s not in want or s in seen: bad += 1; print('ref', s); break
        seen.add(s); pairs += len(ids)
        if len(set(ids)) != len(ids) or any(not (x.startswith('S2-') or x.startswith('S3-')) for x in ids): bad += 1; print('ids line', n); break
        if s in m and not set(m[s]) <= set(ids): bad += 1; print('match outside candidates', s); break
print('rows', len(seen), 'of', len(want), 'candidate pairs', pairs)
print('CANDIDATE FILE OK' if bad == 0 and seen == want else 'CANDIDATE FILE FAILED')
sys.exit(0 if bad == 0 and seen == want else 1)
