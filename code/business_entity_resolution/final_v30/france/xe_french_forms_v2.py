# French cross-encoder input normalization, v2 (27 Sep). v1 mapped French generic suffixes onto English words that
# are DECOY markers in the US/India training data ("& Fils" -> "& Sons", "Groupe" -> "Group", "Ets" -> "Enterprises";
# at a shared address those English additions are ~0% true), so the cross-encoder read the French true-match pattern
# "X SARL & Fils" as a different business. Measured in the test data: & Fils / Associes / Freres / Groupe /
# Developpement appear at the reference's own address 4-20% of the time, like the English true-match suffixes
# Services / Center / Partners (6-18%); Holding / Participations / Distribution only 0.03%, like the English decoys.
# v2 maps the first group to Services / Partners and the second to Group; legal forms as in v1; the "Ets" name prefix
# is dropped. French pairs only; hand-written table, no external data.
# Usage: python work/xe_french_forms_v2.py TEST_BAND.parquet NORM_TEST_DIR OUT.parquet
import re, sys, duckdb, pandas as pd
FORMS = [
    (r"s\.?\s?a\.?\s?s\.?\s?u\.?", "Inc"), (r"s\.?\s?e\.?\s?l\.?\s?a\.?\s?r\.?\s?l\.?", "LLC"), (r"s\.?\s?a\.?\s?r\.?\s?l\.?", "LLC"),
    (r"e\.?\s?u\.?\s?r\.?\s?l\.?", "LLC"), (r"s\.?\s?a\.?\s?s\.?", "Inc"), (r"s\.?\s?c\.?\s?i\.?", "LLC"), (r"s\.?\s?n\.?\s?c\.?", "LP"),
    (r"s\.?\s?c\.?\s?p\.?", "LLP"), (r"s\.?\s?a\.?", "Corp"),
    (r"(?:&|et)\s+fils", "Services"), (r"(?:&|et)\s+fr[eè]res", "Services"), (r"fr[eè]res", "Services"),
    (r"(?:&|et)\s+associ[eé]s", "Partners"), (r"associ[eé]s", "Partners"),
    (r"groupe", "Services"), (r"d[ée]veloppement", "Services"),
    (r"participations", "Group"), (r"holding", "Group"), (r"distribution", "Group"),
    (r"compagnie", "Company"), (r"cie", "Co"), (r"[ée]tablissements|ets", ""),
]
PAT = [(re.compile(r"(?<![\w.])" + p + r"(?![\w])", re.IGNORECASE), r) for p, r in FORMS]
def norm_name(text):
    name, sep, addr = text.partition(" | ")
    for p, r in PAT:
        name = p.sub(r, name)
    return (re.sub(r"\s+", " ", name).strip() or text.partition(" | ")[0]) + sep + addr
if __name__ == "__main__":
    band, norm, out = sys.argv[1:4]
    con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=4")
    fr = con.execute(f"SELECT rid FROM read_parquet('{norm}/norm_s1.parquet') WHERE country = 'france'").df()
    con.register('fr', fr)
    d = con.execute(f"SELECT b.sid, b.tid, b.p, b.a, b.b FROM read_parquet('{band}') b SEMI JOIN fr ON fr.rid=b.sid").df()
    a0, b0 = d.a.copy(), d.b.copy()
    d["a"], d["b"] = d.a.map(norm_name), d.b.map(norm_name)
    d.to_parquet(out, index=False)
    print(f"{out}: {len(d):,} French band pairs; {((d.a != a0) | (d.b != b0)).mean():.1%} with a changed side")
    for x, y in list(zip(b0[(d.b != b0)].head(8), d.b[(d.b != b0)].head(8))):
        print("   ", x[:60], "->", y[:60])
