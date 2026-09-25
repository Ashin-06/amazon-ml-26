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

from .model import hold_rule, lit, normalized_sql
from .prep import core_of, skeleton
from .validation import load_truth
from .integrity import digest


def build_lexicon(work, gt_path, hold_mod=22, min_count=3, min_share=0.5, split="train"):
    """Learn deterministic token mappings from audited non-hold labels only."""
    if split != "train" or min_count < 1 or not 0 < min_share <= 1:
        raise ValueError("Lexicon needs train labels and valid count/share settings")
    if (Path(work) / "lexicon.json").exists():
        raise FileExistsError("Lexicon already exists")
    w = Path(work) / split
    con = duckdb.connect()
    con.execute(f"CREATE VIEW rec AS SELECT * FROM read_parquet({normalized_sql(w)})")
    load_truth(con, gt_path)
    cursor = con.execute(f"""
        SELECT a.name, b.name FROM (
          SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
          FROM read_parquet({lit(gt_path)}) WHERE coalesce(matched_entity_ids, '') <> '') g
        JOIN rec a ON a.entity_id = g.s JOIN rec b ON b.entity_id = g.t
        WHERE b.na AND NOT {hold_rule('a.entity_id', hold_mod)}""")
    pair, tot = defaultdict(Counter), Counter()
    seen = 0
    while rows := cursor.fetchmany(100_000):
        seen += len(rows)
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
        y, n = min(c.items(), key=lambda item: (-item[1], item[0]))
        if n >= min_count and n / tot[x] >= min_share:
            lex[x] = y
    (Path(work) / "lexicon.json").write_text(json.dumps(lex, indent=0, sort_keys=True), encoding="utf-8")
    print(f"[lexicon] {len(lex):,} entries from {seen:,} transliterated train pairs", flush=True)
    con.close()
    return lex


def apply_lexicon(work, split):
    """Rewrite learned names and persist post-lexicon skeletons in bounded 100k-row batches."""
    w = Path(work) / split
    marker = w / "lexicon_applied"
    if marker.exists():
        raise FileExistsError("Use the stage driver to verify completed lexicon application")
    lex = json.loads((Path(work) / "lexicon.json").read_text(encoding="utf-8"))
    changed = 0
    for path in (w / f"norm_s{i}.parquet" for i in (1,2,3)):
        source = pq.ParquetFile(path)
        schema = source.schema_arrow.append(pa.field("skel", pa.string()))
        with pq.ParquetWriter(str(path) + ".tmp", schema) as writer:
            for batch in source.iter_batches(batch_size=100_000):
                t = pa.Table.from_batches([batch])
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
                t = t.append_column("skel", pa.array([skeleton(c) for c in cores], type=pa.string()))
                writer.write_table(t)
        source.close()
        Path(str(path) + ".tmp").replace(path)
    marker.write_text(json.dumps({"changed": changed, "lexicon": digest(Path(work) / "lexicon.json")}), encoding="utf-8")
    print(f"[lexicon] {split}: rewrote {changed:,} transliterated names", flush=True)
