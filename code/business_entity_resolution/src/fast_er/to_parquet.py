"""Convert explicit UTF-8 TSV inputs to string-only parquet in a fresh, verified directory."""
import csv
import sys
from pathlib import Path
import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.integrity import code_digest, environment, files_digest, seal
from fast_er.db import lit


def convert(data, out):
    """Preserve every string and empty field; seal all seven original-input fingerprints."""
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError("Conversion destination exists; use a fresh versioned directory")
    paths = [data / split / f"{split}_source{i}.tsv" for split in ("train", "test") for i in (1,2,3)]
    paths.append(data / "train/train_ground_truth.tsv")
    before = files_digest(paths)
    out.mkdir(parents=True, exist_ok=False)
    outputs = []
    with duckdb.connect() as con:
        con.execute("SET threads=1; SET memory_limit='2GB'")
        for path in paths:
            with path.open(encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
            expected = (["source1_entity_id", "matched_entity_ids"] if "ground_truth" in path.name else
                        ["entity_id", "business_name", "business_address", "country"])
            if columns != expected:
                raise ValueError("Unexpected TSV header: " + path.name)
            name = path.stem + ".parquet"
            force = "[" + ",".join(lit(c) for c in columns) + "]"
            con.execute(f"""COPY (SELECT * FROM read_csv({lit(path)}, delim='\\t', header=true,
                quote='', escape='', all_varchar=true, encoding='utf-8', strict_mode=true,
                force_not_null={force})) TO {lit(out / name)} (FORMAT parquet)""")
            outputs.append(name)
    if files_digest(paths) != before:
        raise ValueError("TSV inputs changed during conversion")
    return seal(out, "conversion_manifest.json",
                {"files": before, "code": code_digest(), "environment": environment()}, outputs)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
