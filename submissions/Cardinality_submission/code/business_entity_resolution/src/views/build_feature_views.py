"""Normalized record views (norm_s1/2/3.parquet) read by the correction, rescue and France stages: the pipeline's
own normalization (fast_er.prep.prepare) followed by a GIVEN transliteration lexicon (fast_er.lexicon.apply_lexicon),
without blocking. The final submission's views were built with models/lexicon_feature_views.json.
usage: python src/views/build_feature_views.py PQ_DIR OUT_DIR LEXICON.json [train|test]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_er.lexicon import apply_lexicon  # noqa: E402
from fast_er.prep import prepare  # noqa: E402

def main():
    pq, out, lex = sys.argv[1:4]
    split = sys.argv[4] if len(sys.argv) > 4 else "test"
    Path(out).mkdir(parents=True, exist_ok=True)
    shutil.copy(lex, Path(out) / "lexicon.json")
    prepare(split, pq_dir=pq, out_dir=out, workers=1)
    apply_lexicon(out, split)
    print("views written to", Path(out) / split)

if __name__ == "__main__":
    main()
