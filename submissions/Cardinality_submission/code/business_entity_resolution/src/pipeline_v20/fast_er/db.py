"""Shared DuckDB/SQL helpers used by candidate generation (stage 1) and the matcher (stage 3)."""
import re
from pathlib import Path


def lit(value):
    """SQL string literal with quotes escaped (paths may contain apostrophes)."""
    return "'" + str(value).replace("'", "''") + "'"


def normalized_sql(directory):
    """Explicit three-file parquet expression; never ingest an unmanifested wildcard match."""
    return "[" + ",".join(lit(Path(directory) / f"norm_s{i}.parquet") for i in (1,2,3)) + "]"


def configure(con, mem, threads, temp_dir):
    """Validated DuckDB resource settings."""
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", str(mem)):
        raise ValueError(f"memory limit must look like '8GB', got {mem!r}")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError(f"threads must be a positive int, got {threads!r}")
    con.execute(f"SET memory_limit={lit(mem)}; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={lit(temp_dir)}")
    if threads:
        con.execute(f"SET threads={threads}")


def hold_rule(col, hold_mod):
    """SQL predicate selecting holdout S1s (~1/hold_mod of references)."""
    if type(hold_mod) is not int or hold_mod < 2:
        raise ValueError(f"hold_mod must be an int >= 2, got {hold_mod!r}")
    return f"(md5_number_lower({col} || 'hold26') % {hold_mod} = 0)"
