"""Stage 1b: learned transliteration lexicon (training labels only, holdout S1s excluded, no external data).

Many India targets spell English business names in native scripts; after anyascii
"Perfect Engineering Private Limited" reads "prphekt imjiniyrimg praivet limitet".
For labelled train pairs whose target name was non-ASCII and has the same token
count as the S1 name, tokens are aligned by position and (translit -> english)
counts are kept when frequent and consistent. The lexicon is then applied to
every non-ASCII-named record (train and test) before blocking/features.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .model import hold_rule
from .prep import core_of


def build_lexicon(work, gt_path, hold_mod=22, min_count=3, min_share=0.5, split="train"):
    w = Path(work) / split
    con = duckdb.connect()
    con.execute(f"CREATE VIEW rec AS SELECT * FROM read_parquet('{w}/norm_s*.parquet')")
    rows = con.execute(f"""
        SELECT a.name, b.name FROM (
          SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
          FROM '{gt_path}' WHERE coalesce(matched_entity_ids, '') <> '') g
        JOIN rec a ON a.entity_id = g.s JOIN rec b ON b.entity_id = g.t
        WHERE b.na AND NOT {hold_rule('a.entity_id', hold_mod)}""").fetchall()
    pair, tot = defaultdict(Counter), Counter()
    for s_name, t_name in rows:
        st, tt = s_name.split(), t_name.split()
        if len(st) != len(tt):
            continue
        for x, y in zip(tt, st):
            tot[x] += 1
            if x != y:
                pair[x][y] += 1
    lex = {}
    for x, c in pair.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / tot[x] >= min_share:
            lex[x] = y
    (Path(work) / "lexicon.json").write_text(json.dumps(lex, indent=0, sort_keys=True))
    print(f"[lexicon] {len(lex):,} entries from {len(rows):,} transliterated train pairs", flush=True)
    return lex


def apply_lexicon(work, split):
    """Rewrite name/core of non-ASCII-named records in work/<split>/norm_s*.parquet (idempotent via marker)."""
    w = Path(work) / split
    marker = w / "lexicon_applied"
    if marker.exists():
        return
    lex = json.loads((Path(work) / "lexicon.json").read_text())
    changed = 0
    for path in sorted(w.glob("norm_s*.parquet")):
        t = pq.read_table(path)
        na = t.column("na").to_pylist()
        names = t.column("name").to_pylist()
        cores = t.column("core").to_pylist()
        for i, flag in enumerate(na):
            if flag:
                new = " ".join(lex.get(x, x) for x in names[i].split())
                if new != names[i]:
                    names[i], cores[i] = new, core_of(new)
                    changed += 1
        t = t.set_column(t.schema.get_field_index("name"), "name", pa.array(names))
        t = t.set_column(t.schema.get_field_index("core"), "core", pa.array(cores))
        pq.write_table(t, str(path) + ".tmp")
        Path(str(path) + ".tmp").replace(path)
    marker.write_text(str(changed))
    print(f"[lexicon] {split}: rewrote {changed:,} transliterated names", flush=True)
