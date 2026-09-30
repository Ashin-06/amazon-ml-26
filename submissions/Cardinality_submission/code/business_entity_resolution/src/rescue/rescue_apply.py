"""Apply the rescue route: matches += accepted rescue pairs (score >= T, unowned target, best reference per
target); candidates += every rescue-route pair with score >= PRE (the route's blocking output).
usage: python rescue_apply.py BASE_MATCH.tsv BASE_CAND.tsv PAIRS.parquet T OUT_MATCH.tsv OUT_CAND.tsv"""
import sys, csv, pandas as pd
bm, bc, pp, T, om, oc = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6]
x = pd.read_parquet(pp)
a = x[(x.score >= T) & ~x.owned].sort_values(['tid', 'score', 'sid'], ascending=[True, False, True]).drop_duplicates('tid')
add = a.groupby('se').te.apply(sorted).to_dict(); cand = x.groupby('se').te.apply(sorted).to_dict()
def rewrite(src, dst, extra):
    n = 0
    with open(src, encoding='utf-8', newline='') as fi, open(dst, 'w', encoding='utf-8', newline='') as fo:
        rd = csv.reader(fi, delimiter='\t', quoting=csv.QUOTE_NONE); wr = csv.writer(fo, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
        wr.writerow(next(rd))
        for row in rd:
            if row[0] in extra:
                have = [i for i in row[1].split(',') if i]; seen = set(have)
                row = [row[0], ','.join(have + [i for i in extra[row[0]] if i not in seen])]; n += 1
            wr.writerow(row)
    return n
print('match refs changed', rewrite(bm, om, add), 'links added', len(a))
print('candidate refs changed', rewrite(bc, oc, cand), 'candidate pairs added', len(x))
