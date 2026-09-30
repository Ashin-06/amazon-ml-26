"""Replace the French rows of a test-band cross-encoder score file with the scores computed on the French-normalized
text, giving the band file used by the France blend.
usage: python merge_french_xe.py TEST_BAND_XE.parquet FR_BAND_XE.parquet OUT.parquet"""
import sys
import pandas as pd

test_xe, fr_xe, out = sys.argv[1:4]
t = pd.read_parquet(test_xe)
f = pd.read_parquet(fr_xe)
m = t.merge(f, on=["sid", "tid"], how="left", suffixes=("", "_fr"), validate="one_to_one")
fr = m.xe_fr.notna()
if fr.sum() != len(f):
    raise SystemExit("every French pair must be in the test band")
m["xe"] = m.xe_fr.fillna(m.xe)
m[["sid", "tid", "xe"]].to_parquet(out, index=False)
print(f"French pairs rescored {int(fr.sum()):,} -> {out}")
