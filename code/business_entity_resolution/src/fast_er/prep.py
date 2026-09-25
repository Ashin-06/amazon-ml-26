"""Stage 1: normalize every record of one split into work/<split>/norm_s{1,2,3}.parquet.

Columns: rid (int, row id within split), src (1/2/3), entity_id, country,
name (normalized), core (name without legal/stop words), addr (normalized),
na (original name was non-ASCII, i.e. transliterated), rawn (raw name, no spaces/mapping).
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
    # French address forms (test-only country)
    "rue": "r", "allee": "all", "impasse": "imp", "cours": "crs", "quai": "qu", "faubourg": "fbg",
    "compagnie": "co", "cie": "co",
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
# France (test-only): references give the region, targets often the departement. Both map to one region
# code, like US state names vs codes. Only whole comma-separated components are mapped, so street names
# such as "Rue du Nord" are untouched.
FR_REGIONS = {
    "frhdf": ["hauts de france", "nord", "pas de calais", "somme", "aisne", "oise"],
    "frnaq": ["nouvelle aquitaine", "gironde", "landes", "dordogne", "lot et garonne", "pyrenees atlantiques",
              "charente", "charente maritime", "deux sevres", "vienne", "haute vienne", "creuse", "correze"],
    "frpdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
}
FR_REGION_OF = {name: code for code, names in FR_REGIONS.items() for name in names}
FR_COMPONENT = re.compile(r"[\s\-]+")
# Dropped from the 'core' name view (legal forms, generic filler).
LEGAL = {"pvt", "ltd", "inc", "llc", "corp", "co", "lp", "llp", "plc", "pc", "pllc", "the", "&", "of", "a", "k",
         "com", "www", "in", "net", "org", "sa", "sas", "sarl", "eurl", "sci", "public", "india", "group",
         "dba", "aka", "d", "b", "l", "p", "c", "s", "de", "du", "des", "la", "le", "les", "france",
         # generic words the target generator appends/drops ("X [Services]", "X Inc Center", "X Partners")
         "svc", "ctr", "partners"}


ORDINALS = {w: f"{i}{'th' if 10 <= i % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')}"
            for i, w in enumerate(["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                                   "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
                                   "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
                                   "twentieth"], 1)}
ORDINALS.update({"thirtieth": "30th", "fortieth": "40th", "fiftieth": "50th"})
NULLS = {"null", "none", "nan"}


def _tok(t):
    """Canonicalize zero padding, abbreviations and ordinal tokens."""
    if t.isdigit():
        return t.lstrip("0") or "0"  # "014889" == "14889" (zero-padding noise)
    return TOKEN_MAP.get(t) or ORDINALS.get(t, t)


def norm(text, states=False, france=False):
    """Lowercase, transliterate, strip punctuation, canonicalize tokens, drop repeated tokens."""
    if not isinstance(text, str) or not text:
        return ""
    if not text.isascii():
        text = anyascii(text.replace("°", " ").replace("º", " "))  # "N°5" -> "N 5", not "Ndeg5"
    text = text.lower().replace("&", " and ")
    if states:
        if france and "," in text:  # France only: whole-component region/departement -> region code
            text = ",".join(FR_REGION_OF.get(FR_COMPONENT.sub(" ", c).strip(), c) for c in text.split(","))
        text = STATE_RE.sub(lambda m: STATES[m.group(1)], text)
    out = []
    for t in map(_tok, NON_ALNUM.sub(" ", text).split()):
        if states and t in NULLS:  # literal "null"/"none" noise inside addresses
            continue
        if not out or out[-1] != t:  # "nature nature" -> "nature"
            out.append(t)
    return " ".join(out)


def raw_joined(text):
    """Transliterated, lowercase, alphanumerics only, no spaces, no abbreviation mapping."""
    if not isinstance(text, str) or not text:
        return ""
    return NON_ALNUM.sub("", (anyascii(text) if not text.isascii() else text).lower())


PHONE = re.compile(r"[\s\-]*\+?\d[\d\- ]{6,}\d")
ALIAS = re.compile(r"(?i)(?<![A-Za-z0-9])(?:a/k/a|aka|f/k/a|fka|d/b/a|dba|trading as|t/a|also known as|formerly known as|formerly|doing business as|operating as)(?=\W)[:\s]*")


FR_ALIAS = re.compile(r"(?i)(?<![A-Za-z0-9])(?:n[ée]e|anciennement)(?=\W)[:\s]*")


# Target-name noise seen in the supplied data (reference names almost never carry it):
# "(ID: 49107)" tags, "| www.x.com" tails, "M/s"/"Shri"/"Smt"/"Mr"/"Dr" prefixes, OCR digits inside
# words ("5ervices", "6lobal", "hea1th", "c0rp") and "ln" for "in" ("lnc", "lndia"), and names
# replaced by a glued domain ("regionalcreativetowing.com").
ID_TAG = re.compile(r"(?i)\s*[(\[]?\bid\b\s*[:#]?\s*\d+\s*[)\]]?\s*$")
URL_TAIL = re.compile(r"(?i)\s*\|\s*(?:https?://)?(?:www\.)?[\w-]+\.(?:com|net|org|co\.in|in|co)\S*\s*$")
HONORIFIC = re.compile(r"(?i)^\W*(?:m/s|sri|shri|smt|mrs|mr|dr)\b\.?\s+(?=\S)")
OCR_WORD = re.compile(r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*[01568])[A-Za-z0-9]+\b")
OCR_MAP = str.maketrans("01568", "olsgb")
OCR_KEEP = re.compile(r"[2-479]|^\d+(?:st|nd|rd|th)$", re.I)  # other digits / ordinals are genuine
LN_WORD = re.compile(r"\b[lL](?=[nN][A-Za-z])")
DOMAIN = re.compile(r"(?i)^\W*(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|co\.in|in|co)\W*$")
VOCAB = frozenset()  # lowercase words of the split's reference names; set per worker by prepare()


def _ocr(m):
    """Map OCR digits 0/1/5/6/8 in a word to o/l/s/g/b unless the word looks genuinely numeric."""
    w = m.group()
    digits = sum(c.isdigit() for c in w)
    return w if OCR_KEEP.search(w) or digits > len(w) - digits else w.translate(OCR_MAP)


def segment(stem, vocab=None):
    """Split a glued domain stem into reference-name words, single letters as initials
    ("regionalcreativetowing" -> "regional creative towing", "hvidyalaya" -> "h vidyalaya").
    Longer words are preferred (score = sum of squared lengths); the stem is kept unchanged when it is a
    known word itself or cannot be covered mostly (>= 60%) by words of 4+ letters."""
    vocab = VOCAB if vocab is None else vocab
    n = len(stem)
    if not vocab or n < 5 or n > 60 or stem in vocab:
        return stem
    best = [None] * (n + 1)
    best[0] = (0, ())
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, min(n, i + 30) + 1):
            w = stem[i:j]
            if j - i > 1 and w not in vocab:
                continue
            cand = (best[i][0] + (j - i) ** 2 * (j - i > 1), best[i][1] + (w,))
            if best[j] is None or cand[0] > best[j][0]:
                best[j] = cand
    parts = best[n][1] if best[n] else ()
    if len(parts) < 2 or sum(len(p) == 1 for p in parts) > 2 or sum(len(p) for p in parts if len(p) >= 4) < 0.6 * n:
        return stem
    return " ".join(parts)


def clean_name(text):
    """Drop appended phone numbers; for 'Brand DBA Real Name' keep the part after the alias marker;
    remove generator noise (ID tags, URL tails, honorific prefixes, OCR digits) and split domain names."""
    if not isinstance(text, str) or not text:
        return text
    original = text
    text = PHONE.sub(" ", text)
    if not text.strip():
        return original
    parts = ALIAS.split(text, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        text = parts[1]
    else:
        parts = FR_ALIAS.split(text, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            text = parts[1]
    text = HONORIFIC.sub("", URL_TAIL.sub("", ID_TAG.sub("", text))).strip() or text
    text = LN_WORD.sub("i", OCR_WORD.sub(_ocr, text))
    m = DOMAIN.match(text)
    return " ".join(segment(p) for p in m.group(1).lower().split("-") if p) if m else text


def name_vocab(path):
    """Lowercase transliterated words (2+ chars) of every reference name in one raw parquet."""
    vocab = set()
    for b in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=["business_name"]):
        for x in b.column("business_name").to_pylist():
            if isinstance(x, str) and x:
                vocab.update(w for w in NON_ALNUM.split((x if x.isascii() else anyascii(x)).lower()) if len(w) >= 2)
    return frozenset(vocab)


def _init_worker(vocab):
    """Pool initializer: give each worker the reference-name vocabulary used by segment()."""
    global VOCAB
    VOCAB = vocab


def _chunk(cols):
    """Normalize one batch while preserving a separate original-name view."""
    ids, names, addrs, countries = cols
    name = [norm(clean_name(x)) for x in names]
    country = [norm(x) for x in countries]
    return (ids, country, name, [core_of(n) for n in name],
            [norm(x, states=True, france=c == "france") for x, c in zip(addrs, country)],
            [not (x or "").isascii() for x in names],
            [raw_joined(x) for x in names])


SKEL_PAIRS = (("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("sh", "s"),
              ("ch", "c"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "j"), ("w", "v"), ("x", "ks"))
VOWELS = re.compile(r"[aeiouyh]")
REPEATS = re.compile(r"(.)\1+")


def skeleton(text):
    """Phonetic consonant skeleton of a normalized name: similar sounds merged, vowels dropped, repeats
    collapsed. Transliteration variants the lexicon has not learned still line up
    ("lakshmi"/"lksmi" -> "lksm", "perfect"/"prphekt" -> "prfkt")."""
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.append(tok)
            continue
        for a, b in SKEL_PAIRS:
            tok = tok.replace(a, b)
        tok = REPEATS.sub(r"\1", VOWELS.sub("", tok))
        if tok:
            out.append(tok)
    return " ".join(out)


def core_of(name):
    """Name without legal forms / filler words (falls back to the full name)."""
    return " ".join(t for t in name.split() if t not in LEGAL) or name


def prepare(split, pq_dir="work/pq", out_dir="work", workers=8):
    """Stream each source in 100k-row batches; one parquet per source, global rid offsets."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=False)
    rid = 0
    vocab = name_vocab(f"{pq_dir}/{split}_source1.parquet")  # unlabeled input text of this split only
    print(f"{split}: reference-name vocabulary {len(vocab):,} words", flush=True)
    with Pool(workers, initializer=_init_worker, initargs=(vocab,)) as pool:
        for src in (1, 2, 3):
            path = out / f"norm_s{src}.parquet"
            pf = pq.ParquetFile(f"{pq_dir}/{split}_source{src}.parquet")
            n = pf.metadata.num_rows
            expected = {"entity_id", "business_name", "business_address", "country"}
            if set(pf.schema_arrow.names) != expected or any(not pa.types.is_string(f.type) and not pa.types.is_large_string(f.type) for f in pf.schema_arrow):
                raise ValueError("Raw parquet must contain the four source columns as strings")
            t = time.time()
            batches = ((b.column("entity_id").to_pylist(), b.column("business_name").to_pylist(),
                        b.column("business_address").to_pylist(), b.column("country").to_pylist())
                       for b in pf.iter_batches(batch_size=100_000))
            tmp = path.with_suffix(".tmp")
            schema = pa.schema([("rid", pa.int32()), ("src", pa.int8()), ("entity_id", pa.string()),
                ("country", pa.string()), ("name", pa.string()), ("core", pa.string()), ("addr", pa.string()),
                ("na", pa.bool_()), ("rawn", pa.string())])
            writer = pq.ParquetWriter(tmp, schema)
            for ids, country, name, core, addr, na, rawn in pool.imap(_chunk, batches):
                table = pa.table({"rid": pa.array(range(rid, rid + len(ids)), pa.int32()),
                                  "src": pa.array([src] * len(ids), pa.int8()),
                                  "entity_id": ids, "country": country, "name": name, "core": core, "addr": addr,
                                  "na": pa.array(na, pa.bool_()), "rawn": rawn}, schema=schema)
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
    assert norm("014889 Twentieth Ave, Nature Nature") == "14889 20th ave nature"
    assert norm("00 Second St") == "0 2nd st"
    assert clean_name("MW Management Private  Limited - 2067865001").strip() == "MW Management Private  Limited"
    assert clean_name("Veowex a/k/a Sbs & Brothers Private Limited") == "Sbs & Brothers Private Limited"
    assert clean_name("Lumkelo formerly: East Stanley Inc") == "East Stanley Inc"
    assert clean_name("Media Aka Services") == "Services"  # rare genuine 'aka' word: acceptable loss
    for s in sys.argv[1:] or ["train", "test"]:
        prepare(s)
