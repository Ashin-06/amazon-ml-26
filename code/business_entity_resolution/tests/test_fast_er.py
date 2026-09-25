"""Regression tests for the fast_er submission pipeline (run: python -m pytest code/business_entity_resolution/tests/test_fast_er.py -q)."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fast_er.block import block  # noqa: E402
from fast_er.model import check_scores, decide, hold_rule, lit, macro_f05, write_outputs  # noqa: E402
from fast_er.prep import clean_name, core_of, norm  # noqa: E402


def test_norm_canonicalizes_noise():
    assert norm("Arihant 5ervices  Private Limited") == "arihant 5ervices pvt ltd"
    assert norm("6451 Old Church Way, Reynoldsburg, Ohio", states=True) == "6451 old church way reynoldsburg oh"
    assert norm("014889 Twentieth Ave, Nature Nature") == "14889 20th ave nature"
    assert norm("Cascade Ctr, DC, null", states=True) == "cascade ctr dc"
    assert norm("Door No 12, MG Road, Bombay, Keralam", states=True) == norm("H.No 12, MG Rd, Mumbai, Kerala", states=True)
    assert norm("Unit 5, 10 Oak St, Kent County", states=True) == "5 10 oak st kent"
    assert norm(None) == "" and norm("") == ""
    assert norm("अरिहंत") != ""  # native script is transliterated, not dropped
    assert core_of(norm("Taran It Pvt Ltd")) == "taran it"
    # French region vs departement components share one code; street names stay untouched
    assert norm("4 Rue du Nord, Lille, Nord", states=True, france=True) == "4 r du nord lille frhdf"
    assert norm("12 Main St, Nord, ND", states=True) == "12 mn st nord nd"  # not France: untouched
    assert norm("19 R. Berthelot, Merignac, Gironde", states=True, france=True) == norm("19 R. Berthelot, Merignac, Nouvelle-Aquitaine", states=True, france=True)


def test_clean_name_phone_and_alias():
    assert clean_name("MW Management Private  Limited - 2067865001").strip() == "MW Management Private  Limited"
    assert clean_name("Veowex a/k/a Sbs & Brothers Private Limited") == "Sbs & Brothers Private Limited"
    assert clean_name("Zetatavo DBA Surya It Pvt Ltd") == "Surya It Pvt Ltd"
    assert clean_name("Kaka Traders") == "Kaka Traders"  # 'aka' inside a word is not a marker
    assert clean_name("Ciraveo née Dunkerque Musique SAS") == "Dunkerque Musique SAS"
    assert clean_name("NEE College SAS") == "NEE College SAS"  # French marker needs a preceding brand
    assert clean_name(None) is None


def test_decide_ties_threshold_and_ownership():
    # target 10 tied between sid 2 and sid 1 -> lowest sid wins (same rule as the SQL writer)
    d = decide(np.array([2, 1, 1]), np.array([10, 10, 11]), np.array([0.9, 0.9, 0.5]), 0.6)
    assert d == {1: [10]}
    # a stronger competitor takes the target
    d = decide(np.array([1, 2]), np.array([7, 7]), np.array([0.8, 0.95]), 0.5)
    assert d == {2: [7]}


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        decide(np.array([1]), np.array([2]), np.array([np.nan]), 0.5)
    with pytest.raises(ValueError):
        check_scores([0.2, 1.5])
    with pytest.raises(ValueError):
        check_scores([0.2], thr=float("nan"))


def test_macro_f05_edge_cases():
    truth = {1: {10, 11}, 2: set(), 3: {30}}
    pred = {1: [10, 12], 2: [], 3: []}
    # s1: tp=1 fp=1 fn=1 -> 5/(5+4+1)=0.5 ; s2 singleton empty -> 1 ; s3 missed -> 0
    assert macro_f05(truth, pred, [1, 2, 3]) == pytest.approx((0.5 + 1 + 0) / 3)
    assert macro_f05({}, {}, [7]) == 1.0  # singleton with empty prediction
    with pytest.raises(ValueError):
        macro_f05({}, {}, [])
    with pytest.raises(ValueError):
        hold_rule("entity_id", 0)


def _rec(con):
    con.execute("""CREATE TABLE rec AS SELECT *, 'us' AS country FROM (VALUES
        (0, 1, 'S1-1'), (1, 1, 'S1-2'), (2, 1, 'S1-3'),
        (3, 2, 'S2-10'), (4, 3, 'S3-20'), (5, 2, 'S2-30')) t(rid, src, entity_id)""")


def test_write_outputs_every_s1_and_sql_tie_rule(tmp_path):
    con = duckdb.connect()
    _rec(con)
    sc = pd.DataFrame({"sid": [0, 1, 0, 1], "tid": [3, 3, 4, 5], "p": [0.9, 0.9, 0.2, 0.7]})
    sc.to_parquet(tmp_path / "scores.parquet", index=False)
    n_match, n_s1 = write_outputs(con, tmp_path / "scores.parquet", 0.5, tmp_path, chunks=2)
    rows = dict(l.rstrip("\n").split("\t") for l in (tmp_path / "matching_results.tsv").read_text().splitlines()[1:])
    cands = dict(l.rstrip("\n").split("\t") for l in (tmp_path / "candidate_pairs.tsv").read_text().splitlines()[1:])
    assert n_s1 == 3 and set(rows) == {"S1-1", "S1-2", "S1-3"}
    assert rows == {"S1-1": "S2-10", "S1-2": "S2-30", "S1-3": ""}  # tie on S2-10 -> lowest sid, as decide()
    assert cands["S1-3"] == "" and set(cands["S1-1"].split(",")) == {"S2-10", "S3-20"}
    py = decide(sc.sid.values, sc.tid.values, sc.p.values, 0.5)
    assert py == {0: [3], 1: [5]} and n_match == 2


def test_write_outputs_rejects_nan_and_duplicates(tmp_path):
    con = duckdb.connect()
    _rec(con)
    pd.DataFrame({"sid": [0], "tid": [3], "p": [np.nan]}).to_parquet(tmp_path / "s.parquet", index=False)
    with pytest.raises(ValueError):
        write_outputs(con, tmp_path / "s.parquet", 0.5, tmp_path)
    pd.DataFrame({"sid": [0, 0], "tid": [3, 3], "p": [0.9, 0.9]}).to_parquet(tmp_path / "d.parquet", index=False)
    with pytest.raises(ValueError):
        write_outputs(con, tmp_path / "d.parquet", 0.5, tmp_path)


def test_sql_literal_paths_with_apostrophes(tmp_path):
    d = tmp_path / "O'Brien dir"
    d.mkdir()
    pd.DataFrame({"x": [1, 2]}).to_parquet(d / "f.parquet", index=False)
    assert duckdb.connect().execute(f"SELECT sum(x) FROM read_parquet({lit(d / 'f.parquet')})").fetchone()[0] == 3


def test_block_finds_renamed_and_reordered_matches(tmp_path):
    split = tmp_path / "toy"
    split.mkdir()
    rows = [  # rid, src, entity_id, country, name, core, addr, na, rawn
        (0, 1, "S1-1", "us", "pioneer infotech pvt ltd", "pioneer infotech", "45 hill rd pune mh", False, ""),
        (1, 1, "S1-2", "us", "blue river llc", "blue river", "9 lake st austin tx", False, ""),
        (2, 2, "S2-1", "us", "pioneer infotech", "pioneer infotech", "pune mh 45 hill rd", False, ""),
        (3, 3, "S3-1", "us", "drexecto", "drexecto", "45 hill rd pune mh", False, ""),
        (4, 2, "S2-2", "us", "blue river", "blue river", "", False, ""),
        (5, 3, "S3-2", "in", "blue river", "blue river", "9 lake st austin tx", False, ""),  # other country
    ]
    df = pd.DataFrame(rows, columns=["rid", "src", "entity_id", "country", "name", "core", "addr", "na", "rawn"])
    for s in (1, 2, 3):
        df[df.src == s].to_parquet(split / f"norm_s{s}.parquet", index=False)
    out = block("toy", work=tmp_path, cap=10, k=5, k_field=5, chunks=2, threads=1, mem="1GB")
    c = duckdb.connect().execute(f"SELECT sid, tid FROM read_parquet({lit(out)})").fetchall()
    assert (0, 2) in c and (0, 3) in c and (1, 4) in c  # reordered address, renamed DBA, empty-address target
    assert (1, 5) not in c  # never across countries


def test_skeleton_merges_transliterations_but_not_unrelated_names():
    from fast_er.prep import skeleton
    assert skeleton("lakshmi traders") == skeleton("lksmi traders")
    assert skeleton("perfect") == skeleton("prphekt")
    assert skeleton("blue river") != skeleton("green valley")
    assert skeleton("12 main") .startswith("12")  # numbers kept verbatim


def test_clean_name_removes_target_generator_noise():
    assert clean_name("Uppal Global Private Limited (ID: 49107)") == "Uppal Global Private Limited"
    assert clean_name("Foster, Chiasson & Tucker Rogers #67631") == "Foster, Chiasson & Tucker Rogers"
    assert clean_name("Ma Vidyalaya | www.mavidyala.com") == "Ma Vidyalaya"
    assert clean_name("M/s People Cure Private Limited") == "People Cure Private Limited"
    assert clean_name("Smt Lakshmi Traders") == "Lakshmi Traders"
    assert clean_name("Arihant 5ervices 6roup hea1th c0rp") == "Arihant services group health corp"
    assert clean_name("Tata lnc lndia") == "Tata inc india"
    assert clean_name("1st Choice 24hr 7eleven B2B") == "1st Choice 24hr 7eleven B2B"  # genuine digits kept
    assert clean_name("Novifaye formerly known as Tedyndyne Crown LLC") == "Tedyndyne Crown LLC"
    assert clean_name("Synzephnex fka Tamar Beebe Limited") == "Tamar Beebe Limited"
    assert clean_name("Dr") == "Dr"


def test_domain_names_are_segmented_with_reference_vocabulary():
    from fast_er import prep
    vocab = {"regional": 9, "creative": 7, "towing": 3, "vidyalaya": 5, "badyne": 1, "fou": 2, "dai": 2, "ton": 4,
             "clean": 50, "marine": 30, "systems": 80, "marin": 2, "esystems": 1}
    assert prep.segment("regionalcreativetowing", vocab) == "regional creative towing"
    assert prep.segment("hvidyalaya", vocab) == "h vidyalaya"
    assert prep.segment("badyne", vocab) == "badyne"  # known word stays whole
    assert prep.segment("foudaiton", vocab) == "foudaiton"  # only short pieces: kept
    assert prep.segment("cleanmarinesystems", vocab) == "clean marine systems"  # common words win
    old = prep.VOCAB
    try:
        prep.VOCAB = vocab
        assert prep.clean_name("REGIONALCREATIVETOWING.COM") == "regional creative towing"
        assert prep.clean_name("-- regional-creativetowing.co.in") == "regional creative towing"
    finally:
        prep.VOCAB = old


def test_house_number_delta_features():
    from fast_er.model import _num_feats
    f = _num_feats(["353 oak st", "1234 elm st", "9 pine rd"], ["355 oak st", "1334 elm st", "pine rd"])
    assert f[0, 8] > 0 and f[0, 9] == 1.0 and f[1, 9] == 1.0  # 353/355 and 1234/1334: one digit differs
    assert abs(f[1, 8] - np.log1p(100)) < 1e-5 and f[2, 8] == -1 and f[2, 9] == -1


def test_sql_pair_features_match_python_definitions():
    """_pairs computes counts/lengths/digit tokens in SQL; features() must equal the reference Python formulas."""
    from fast_er import model
    con = duckdb.connect()
    recs = [(0, 1, "acme dental", "acme dental", "353 oak st 60601"), (1, 2, "acme dental", "acme dental", "355 oak st 60601"),
            (2, 3, "acme dent", "acme dent", "1234 elm st"), (3, 2, "zed", "zed", ""), (4, 3, "acme dental", "acme dental", "0353 oak st 353"),
            (5, 2, "x co", "x", "12 12 main 7"), (6, 3, "acme dental", "acme dental", "353 oak st 60601")]
    con.execute("CREATE TABLE rec (rid INTEGER, src TINYINT, name VARCHAR, core VARCHAR, addr VARCHAR)")
    con.executemany("INSERT INTO rec VALUES (?, ?, ?, ?, ?)", recs)
    con.execute("""CREATE TABLE rec2 AS SELECT *, name AS rawn, core AS skel, regexp_extract_all(addr, '[0-9]+') AS nums,
                   1 AS fc_t, 1 AS fc_s, 1 AS fa_t FROM rec""")
    con.execute("DROP TABLE rec; ALTER TABLE rec2 RENAME TO rec")
    base = ", ".join(f"0.0 AS {b}" for b in model.BASE if b not in ("tsrc", "s_fct", "s_fcs", "s_fat", "t_fct", "t_fcs", "t_fat"))
    con.execute(f"CREATE TABLE c AS SELECT 0 AS sid, rid AS tid, {base} FROM rec WHERE rid > 0")
    df = model._pairs(con, "TRUE")
    F = model.features(df)
    col = lambda k: F[:, model.FEATS.index(k)]
    ref = model._num_feats(df.saddr.tolist(), df.taddr.tolist())
    for j, k in enumerate(["num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt", "t_num_cnt",
                           "hn_ratio", "hn_rel", "hn_logdiff", "hn_sub1"]):
        np.testing.assert_allclose(col(k), ref[:, j], atol=1e-6, err_msg=k)
    np.testing.assert_array_equal(col("sup_core"), df.groupby(["sid", "tcore"]).sid.transform("size").to_numpy())
    np.testing.assert_array_equal(col("c_exact"), (df.scor == df.tcore).to_numpy())
    np.testing.assert_array_equal(col("t_addr_empty"), (df.taddr == "").to_numpy())
    np.testing.assert_array_equal(col("len_a_t"), df.taddr.str.len().to_numpy())
    assert col("sup_addr")[(df.taddr == "").to_numpy()].tolist() == [-1]
    by = {int(r): i for i, r in enumerate(df.tid)}
    assert col("ft_lev")[by[2]] == 0 and col("ft_pre")[by[2]] == 1 and col("ft_suf")[by[2]] == 1  # acme / acme
    assert col("ft_lev")[by[5]] == 4 and col("ft_pre")[by[5]] == 0  # acme vs x
    assert col("hn_in_t")[by[4]] == 1 and col("hn_t_in_s")[by[4]] == 0  # 353 in [0353, 353]; 0353 not in [353, 60601]
    assert col("hn_in_t")[by[3]] == -1  # empty target address



def test_candidate_slices_cover_every_reference_once():
    from fast_er import model
    con = duckdb.connect()
    con.execute("CREATE TABLE c AS SELECT (i * 7919) % 1000 AS sid, i AS tid FROM range(5000) t(i) ORDER BY sid")
    seen = []
    for k in range(7):
        seen += con.execute(f"SELECT sid FROM c WHERE {model._slice(con, 7, k)}").df().sid.tolist()
    assert sorted(seen) == sorted(con.execute("SELECT sid FROM c").df().sid.tolist())
