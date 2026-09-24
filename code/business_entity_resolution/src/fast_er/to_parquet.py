"""Stage 0: convert the supplied TSVs to parquet (strings kept verbatim, empty fields -> NULL).

Usage: python to_parquet.py DATASET_DIR OUT_DIR   (DATASET_DIR contains train/ and test/)
"""
import sys
import time
from pathlib import Path

import duckdb

if __name__ == "__main__":
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for split in ("train", "test"):
        names = [f"{split}_source{i}" for i in (1, 2, 3)] + (["train_ground_truth"] if split == "train" else [])
        for n in names:
            dst = out / f"{n}.parquet"
            if dst.exists():
                continue
            t = time.time()
            con.execute(f"""COPY (SELECT * FROM read_csv('{data / split / (n + ".tsv")}', delim='\\t', header=true,
                            quote='', escape='', all_varchar=true, strict_mode=true)) TO '{dst}' (FORMAT parquet)""")
            print(n, con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0], f"{time.time() - t:.0f}s", flush=True)
