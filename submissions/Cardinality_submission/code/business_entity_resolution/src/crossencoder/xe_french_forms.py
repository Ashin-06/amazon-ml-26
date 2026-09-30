# Cross-encoder input normalization for French records only: the cross-encoder was trained on US/India pairs, where it
# learned that legal forms and generic suffixes ("LLC", "Inc", "& Sons", "Group") being added or dropped do not change
# identity, but it has never seen the French forms, so it scores "Lille Compagnie" vs "Lille Compagnie SASU" as a
# mismatch (worklog 20). This maps French legal forms/suffixes to the English ones it knows, on both sides of French
# pairs, before scoring. US/India inputs are untouched. A hand-written normalization table (like the pipeline's street
# and legal-form tables), no external data.
# Usage: python work/xe_french_forms.py TEST_BAND.parquet NORM_TEST_DIR OUT_DIR   -> OUT_DIR/<name>_fr_band.parquet
import re, sys, duckdb, pandas as pd
FORMS = [  # (regex on the raw text, replacement); dotted and plain spellings; order matters (longest first)
    (r"s\.?\s?a\.?\s?s\.?\s?u\.?", "Inc"), (r"s\.?\s?e\.?\s?l\.?\s?a\.?\s?r\.?\s?l\.?", "LLC"), (r"s\.?\s?a\.?\s?r\.?\s?l\.?", "LLC"),
    (r"e\.?\s?u\.?\s?r\.?\s?l\.?", "LLC"), (r"s\.?\s?a\.?\s?s\.?", "Inc"), (r"s\.?\s?c\.?\s?i\.?", "LLC"), (r"s\.?\s?n\.?\s?c\.?", "LP"),
    (r"s\.?\s?c\.?\s?p\.?", "LLP"), (r"s\.?\s?a\.?", "Corp"),
    (r"(?:&|et)\s+fils", "& Sons"), (r"(?:&|et)\s+associ[eé]s", "& Associates"), (r"associ[eé]s", "Associates"),
    (r"compagnie", "Company"), (r"cie", "Co"), (r"groupe", "Group"), (r"[ée]tablissements|ets", "Enterprises"),
    (r"d[ée]veloppement", "Development"), (r"participations", "Holdings"), (r"holding", "Holdings"),
]
PAT = [(re.compile(r"(?<![\w.])" + p + r"(?![\w])", re.IGNORECASE), r) for p, r in FORMS]
def norm_name(text):
    name, sep, addr = text.partition(" | ")
    for p, r in PAT:
        name = p.sub(r, name)
    return re.sub(r"\s+", " ", name).strip() + sep + addr
if __name__ == "__main__":
    band, norm, out = sys.argv[1:4]
    con = duckdb.connect(); con.execute("SET memory_limit='2GB'; SET threads=4")
    fr = set(con.execute(f"SELECT rid FROM read_parquet('{norm}/norm_s1.parquet') WHERE country = 'france'").df().rid)
    d = pd.read_parquet(band)
    d = d[d.sid.isin(fr)].reset_index(drop=True)
    a0, b0 = d.a.copy(), d.b.copy()
    d["a"], d["b"] = d.a.map(norm_name), d.b.map(norm_name)
    changed = ((d.a != a0) | (d.b != b0)).mean()
    name = band.replace("\\", "/").rsplit("/", 1)[-1].replace("_band.parquet", "_fr_band.parquet")
    d.to_parquet(f"{out}/{name}", index=False)
    print(f"{out}/{name}: {len(d):,} French band pairs; {changed:.1%} with a changed side")
    for x, y in list(zip(a0[(d.a != a0)].head(6), d.a[(d.a != a0)].head(6))):
        print("   ", x[:70], "->", y[:70])
