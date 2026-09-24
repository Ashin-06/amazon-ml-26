"""Stage 1: normalize every record of one split into work/<split>/norm_s{1,2,3}.parquet.

Columns: rid (int, row id within split), src (1/2/3), entity_id, country,
name (normalized), core (name without legal/stop words), addr (normalized),
na (original name was non-ASCII, i.e. transliterated).
Non-ASCII text (Indic scripts, French accents) is transliterated with anyascii
so cross-script pairs share a Latin view. No external lookups.
"""
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from anyascii import anyascii

NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Canonical short forms. Applied token-by-token after lowercasing.
TOKEN_MAP = {
    # street types (US / India / France)
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave", "lane": "ln",
    "court": "ct", "drive": "dr", "boulevard": "blvd", "bd": "blvd", "place": "pl", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "square": "sq", "trail": "trl",
    "north": "n", "south": "s", "east": "e", "west": "w", "suite": "ste", "apartment": "apt",
    "floor": "fl", "flr": "fl", "building": "bldg", "number": "no", "num": "no", "nos": "no",
    "house": "h", "hno": "h", "sector": "sec", "near": "nr", "opposite": "opp", "cross": "crs",
    "main": "mn", "nagar": "ngr", "colony": "col", "chemin": "ch", "route": "rte",
    # legal forms
    "private": "pvt", "pvt.": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp",
    "company": "co", "and": "&", "et": "&", "services": "svc", "service": "svc",
    "international": "intl", "associates": "assoc", "association": "assoc", "brothers": "bros",
    "center": "ctr", "centre": "ctr", "enterprises": "ent", "enterprise": "ent",
    "industries": "ind", "industry": "ind", "technologies": "tech", "technology": "tech",
    "solutions": "sol", "management": "mgmt", "development": "dev", "developers": "dev",
    "saint": "st", "sainte": "ste",
}
# Multi-word phrases replaced before tokenizing (US + India states, a few common ones).
STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    # anyascii forms of native-script state names (mined from supplied India target addresses)
    "mharastr": "mh", "dilli": "dl", "uttr prdes": "up", "krnatk": "ka", "tmilnatu": "tn", "pscimbng": "wb",
    "gujrat": "gj", "telmgan": "tg", "hriyana": "hr", "rajsthan": "rj", "kerlm": "kl", "mdhy prdes": "mp",
    "amdhrprdes": "ap", "pmjab": "pb", "od'isa": "od",
}
STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATES), key=len, reverse=True)) + r")\b")
# Dropped from the 'core' name view (legal forms, generic filler).
LEGAL = {"pvt", "ltd", "inc", "llc", "corp", "co", "lp", "llp", "plc", "pc", "pllc", "the", "&", "of", "a", "k",
         "com", "www", "in", "net", "org", "sa", "sas", "sarl", "eurl", "sci", "public", "india", "group",
         "dba", "aka", "d", "b", "l", "p", "c", "s"}


def norm(text, states=False):
    """Lowercase, transliterate, strip punctuation, canonicalize tokens."""
    if not isinstance(text, str) or not text:
        return ""
    if not text.isascii():
        text = anyascii(text)
    text = text.lower().replace("&", " and ")
    if states:
        text = STATE_RE.sub(lambda m: STATES[m.group(1)], text)
    return " ".join(TOKEN_MAP.get(t, t) for t in NON_ALNUM.sub(" ", text).split())


def _chunk(cols):
    ids, names, addrs, countries = cols
    name = [norm(x) for x in names]
    return (ids, [norm(x) for x in countries], name, [core_of(n) for n in name],
            [norm(x, states=True) for x in addrs], [not (x or "").isascii() for x in names])


def core_of(name):
    """Name without legal forms / filler words (falls back to the full name)."""
    return " ".join(t for t in name.split() if t not in LEGAL) or name


def prepare(split, pq_dir="work/pq", out_dir="work", workers=8):
    """Stream each source in 100k-row batches; one parquet per source, global rid offsets."""
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    rid = 0
    with Pool(workers) as pool:
        for src in (1, 2, 3):
            path = out / f"norm_s{src}.parquet"
            pf = pq.ParquetFile(f"{pq_dir}/{split}_source{src}.parquet")
            n = pf.metadata.num_rows
            if path.exists():
                rid += n
                continue
            t = time.time()
            batches = ((b.column("entity_id").to_pylist(), b.column("business_name").to_pylist(),
                        b.column("business_address").to_pylist(), b.column("country").to_pylist())
                       for b in pf.iter_batches(batch_size=100_000))
            tmp = path.with_suffix(".tmp")
            writer = None
            for ids, country, name, core, addr, na in pool.imap(_chunk, batches):
                table = pa.table({"rid": pa.array(range(rid, rid + len(ids)), pa.int32()),
                                  "src": pa.array([src] * len(ids), pa.int8()),
                                  "entity_id": ids, "country": country, "name": name, "core": core, "addr": addr,
                                  "na": pa.array(na, pa.bool_())})
                writer = writer or pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table)
                rid += len(ids)
            writer.close()
            tmp.rename(path)
            print(f"{split} S{src}: {n:,} rows {time.time() - t:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    assert norm("Arihant 5ervices  Private Limited") == "arihant 5ervices pvt ltd"
    assert norm("6451 Old Church Way, Reynoldsburg, Ohio", states=True) == "6451 old church way reynoldsburg oh"
    assert norm("B+ Retail & Co.") == "b retail & co"
    for s in sys.argv[1:] or ["train", "test"]:
        prepare(s)
