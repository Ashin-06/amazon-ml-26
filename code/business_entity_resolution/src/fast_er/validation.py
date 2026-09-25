"""Disk-backed schema, reference coverage, label and candidate integrity gates."""


def validate_records(con):
    """Require unique integer internal keys, valid string source IDs and country fields."""
    bad = con.execute("""SELECT count(*) FROM rec WHERE rid IS NULL OR rid < 0 OR rid != trunc(rid)
        OR src IS NULL OR src NOT IN (1,2,3) OR entity_id IS NULL OR country IS NULL
        OR NOT regexp_full_match(entity_id, 'S' || src::VARCHAR || '-[0-9]+')""").fetchone()[0]
    n, nr, ni = con.execute("SELECT count(*), count(DISTINCT rid), count(DISTINCT entity_id) FROM rec").fetchone()
    if bad or n != nr or n != ni or not con.execute("SELECT count(*) FROM rec WHERE src=1").fetchone()[0]:
        raise ValueError("record integrity: invalid/duplicate keys or no S1 references")


def validate_pairs(con, table):
    """Reject duplicates, unknown or reversed endpoints and cross-country candidates."""
    if table not in ("c", "sc", "gt"):
        raise ValueError("Unsupported pair table")
    bad = con.execute(f"""SELECT count(*) FROM {table} c LEFT JOIN rec a ON a.rid=c.sid
        LEFT JOIN rec b ON b.rid=c.tid WHERE a.rid IS NULL OR b.rid IS NULL
        OR a.src != 1 OR b.src NOT IN (2,3) OR a.country != b.country""").fetchone()[0]
    dup = con.execute(f"SELECT count(*) - count(DISTINCT (sid,tid)) FROM {table}").fetchone()[0]
    if bad or dup:
        raise ValueError(f"{table} pair integrity: {bad} invalid endpoints/countries; {dup} duplicate pairs")


def load_truth(con, gt_path):
    """Require exactly one label row per S1, including empty lists, and unique target ownership."""
    from .model import lit
    validate_records(con)
    con.execute(f"CREATE OR REPLACE TABLE labels AS SELECT * FROM read_parquet({lit(gt_path)})")
    n_lab, unique_lab = con.execute("SELECT count(*), count(DISTINCT source1_entity_id) FROM labels").fetchone()
    missing = con.execute("""SELECT count(*) FROM rec a LEFT JOIN labels l ON l.source1_entity_id=a.entity_id
        WHERE a.src=1 AND l.source1_entity_id IS NULL""").fetchone()[0]
    unknown = con.execute("""SELECT count(*) FROM labels l LEFT JOIN rec a
        ON a.entity_id=l.source1_entity_id AND a.src=1 WHERE a.rid IS NULL""").fetchone()[0]
    if missing or unknown or n_lab != unique_lab:
        raise ValueError("label integrity: duplicate, missing or unknown S1 label rows")
    con.execute("""CREATE OR REPLACE TABLE gt_raw AS SELECT source1_entity_id s,
        unnest(string_split(matched_entity_ids, ',')) t FROM labels
        WHERE coalesce(matched_entity_ids, '') <> ''""")
    con.execute("""CREATE OR REPLACE TABLE gt AS SELECT a.rid sid, b.rid tid FROM gt_raw g
        LEFT JOIN rec a ON a.entity_id=g.s AND a.src=1
        LEFT JOIN rec b ON b.entity_id=g.t AND b.src IN (2,3)""")
    validate_pairs(con, "gt")
    links, unique_targets = con.execute("SELECT count(*), count(DISTINCT tid) FROM gt").fetchone()
    if links != unique_targets:
        raise ValueError("label integrity: a target has multiple reference owners")
    return {"links": links, "label_rows": n_lab, "targets_with_2plus_owners": 0}
