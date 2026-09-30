"""OUT = BASE + links that NEW has and BASE lacks, only for targets BASE assigns to nobody (BASE links never removed).
usage: python union_additions.py BASE.tsv NEW.tsv OUT.tsv"""
import sys, csv
base, new, out = sys.argv[1:4]
rd = lambda p: [r for r in csv.reader(open(p, encoding='utf-8', newline=''), delimiter='\t', quoting=csv.QUOTE_NONE)]
b, n = rd(base), {r[0]: r[1] for r in rd(new)}
owned = set(t for r in b[1:] for t in r[1].split(',') if t)
added = 0
with open(out, 'w', encoding='utf-8', newline='') as fo:
    wr = csv.writer(fo, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n'); wr.writerow(b[0])
    for s, m in b[1:]:
        have = [t for t in m.split(',') if t]
        extra = [t for t in n.get(s, '').split(',') if t and t not in owned and t not in have]
        owned.update(extra); added += len(extra)
        wr.writerow([s, ','.join(have + extra)])
print('links added', added)
