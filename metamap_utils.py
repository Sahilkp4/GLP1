"""MetaMapLite support for stage 1: run MetaMapLite over posts and map UMLS CUIs to MedDRA LLTs.

MetaMapLite (Java, NLM) finds UMLS concepts in free text. The installed dataset (2022AA
Level 0+4+9 / "USAbase") does not contain MedDRA, so CUIs are linked to MedDRA through their
UMLS synonyms: every string of a CUI (SNOMED CT, MeSH, NCI, CHV, HPO, ...) is matched against
the current MedDRA LLT names from meddra_utils.load_meddra, and one LLT is chosen per CUI.

UMLS data is licensed (UTS): keep MetaMap/ local and do not share or publish it.

Usage:
  python metamap_utils.py --crosswalk           # build MetaMap/cui_to_meddra.tsv, print coverage
  python metamap_utils.py --text "I feel nauseous and dizzy"   # run MetaMapLite on one text
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MML_DIR = os.path.join(HERE, "MetaMap", "public_mm_lite")
CACHE_DIR = os.path.join(HERE, "MetaMap", "cache")
CROSSWALK_PATH = os.path.join(HERE, "MetaMap", "cui_to_meddra.tsv")
MML_VERSION = "3.6.2rc8"
INDEX_SUBDIR = os.path.join("data", "ivf", "2022AA", "USAbase")

# Semantic types searched by MetaMapLite (symptoms, diseases, findings, injuries, ...).
SYMPTOM_STS = ["sosy", "dsyn", "fndg", "patf", "mobd", "inpo", "acab", "anab", "cgab",
               "neop", "emod"]
# When an entity has several candidate concepts, prefer these semantic types in this order.
ST_PRIORITY = {st: i for i, st in enumerate(["sosy", "dsyn", "mobd", "patf", "inpo", "fndg"])}

BATCH = 2000  # posts per JVM call


# --------------------------------------------------------------------------------------
# Running MetaMapLite
# --------------------------------------------------------------------------------------
def _sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _utf16_to_py(text):
    """Java offsets count UTF-16 code units; astral characters (emoji) take two. Return a
    list mapping each UTF-16 offset to the Python string index."""
    out = []
    for i, ch in enumerate(text):
        out.append(i)
        if ord(ch) > 0xFFFF:
            out.append(i)
    out.append(len(text))
    return out


def _java_cmd(mml_dir, filelist, sts, heap):
    idx = os.path.join(mml_dir, INDEX_SUBDIR)
    if not os.path.isdir(idx):
        sys.exit(f"MetaMapLite index not found at {idx}")
    return ["java", f"-Xmx{heap}", "-Dfile.encoding=UTF-8",
            "-Dlog4j.configurationFile=" + os.path.join(mml_dir, "config", "log4j2.xml"),
            "-cp", os.path.join(mml_dir, "target", f"metamaplite-{MML_VERSION}-standalone.jar"),
            "gov.nih.nlm.nls.ner.MetaMapLite",
            "--indexdir=" + idx,
            "--modelsdir=" + os.path.join(mml_dir, "data", "models"),
            "--specialtermsfile=" + os.path.join(mml_dir, "data", "specialterms.txt"),
            "--outputformat=json", "--restrict_to_sts=" + ",".join(sts),
            "--filelistfn=" + filelist]


def _parse(text, entities):
    """MetaMapLite JSON entities -> hits with Python character offsets."""
    u2p = _utf16_to_py(text)
    hits, dropped = [], 0
    for ent in entities:
        s16, n16 = ent["start"], ent["length"]
        if s16 + n16 >= len(u2p):
            dropped += 1
            continue
        start, end = u2p[s16], u2p[s16 + n16]
        want = ent["matchedtext"].lower()
        if text[start:end].lower() != want:  # realign if MetaMapLite shifted the offset
            lo = max(0, start - 5)
            k = text.lower().find(want, lo, end + 5 + len(want))
            if k < 0:
                dropped += 1
                continue
            start, end = k, k + len(want)
        concepts = {}
        for ev in ent.get("evlist", []):
            ci = ev["conceptinfo"]
            concepts.setdefault(ci["cui"], {
                "cui": ci["cui"], "preferred_name": ci.get("preferredname", ""),
                "concept_string": ci.get("conceptstring", ""),
                "semtypes": ci.get("semantictypes", []), "sources": ci.get("sources", [])})
        if concepts:
            hits.append({"start": start, "end": end, "text": text[start:end],
                         "concepts": list(concepts.values())})
    return hits, dropped


def run_metamaplite(texts, mml_dir=MML_DIR, cache_dir=CACHE_DIR, sts=SYMPTOM_STS, heap="8g"):
    """Run MetaMapLite over an iterable of texts. Return a list of hit lists, aligned with
    texts. Results are cached by text hash, so only new texts reach the JVM."""
    texts = list(texts)
    work = os.path.join(cache_dir, "mml")
    os.makedirs(work, exist_ok=True)
    shas = [_sha(t) for t in texts]
    todo = {}
    for t, h in zip(texts, shas):
        if t.strip() and not os.path.exists(os.path.join(work, h + ".json")):
            todo[h] = t
    todo = list(todo.items())
    if todo:
        print(f"MetaMapLite: {len(todo):,} new texts ({len(texts) - len(todo):,} cached)",
              file=sys.stderr)
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        paths = []
        for h, t in chunk:
            p = os.path.join(work, h + ".txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(t)
            paths.append(p)
        filelist = os.path.join(work, "filelist.txt")
        with open(filelist, "w") as fh:
            fh.write("\n".join(paths) + "\n")
        res = subprocess.run(_java_cmd(mml_dir, filelist, sts, heap), capture_output=True,
                             text=True)
        if res.returncode != 0:
            sys.exit(f"MetaMapLite failed (exit {res.returncode}):\n{res.stderr[-3000:]}")
        for (h, _), p in zip(chunk, paths):
            out = p[:-4] + ".json"
            if not os.path.exists(out):  # no entities found -> no output file
                with open(out, "w") as fh:
                    fh.write("[]")
            os.remove(p)
        print(f"  MetaMapLite: {min(b + BATCH, len(todo)):,}/{len(todo):,}", file=sys.stderr)

    results, dropped = [], 0
    for t, h in zip(texts, shas):
        if not t.strip():
            results.append([])
            continue
        with open(os.path.join(work, h + ".json"), encoding="utf-8") as fh:
            raw = fh.read().strip() or "[]"
        hits, d = _parse(t, json.loads(raw))
        results.append(hits)
        dropped += d
    if dropped:
        print(f"[warn] MetaMapLite: {dropped} hits dropped (offsets did not match text)",
              file=sys.stderr)
    return results


# --------------------------------------------------------------------------------------
# CUI -> MedDRA crosswalk
# --------------------------------------------------------------------------------------
_SUFFIX_RE = re.compile(
    r"\s*\((finding|disorder|symptom|situation|morphologic abnormality|observable entity|"
    r"qualifier value|event)\)$"
    r"|,?\s+(nos|unspecified|\[ambiguous\])$", re.I)
_PREFIX_RE = re.compile(r"^[a-z]\s+-\s+", re.I)  # SNOMED "D - Diarrhoea"


def normalize(s):
    """Variants of a UMLS string that may equal a MedDRA LLT name (exact form excluded)."""
    s = s.lower().strip()
    out = set()
    t = _PREFIX_RE.sub("", s)
    prev = None
    while prev != t:
        prev, t = t, _SUFFIX_RE.sub("", t).strip()
    out.add(t)
    if t.count(",") == 1:  # "pain, abdominal" -> "abdominal pain"
        a, b = (x.strip() for x in t.split(","))
        if a and b:
            out.add(f"{b} {a}")
    out.discard(s)
    return out


def _postings(index_dir, table):
    pattern = os.path.join(index_dir, "indices", table, "*postings.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No postings files for {table} under {index_dir}")
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line.rstrip("\n").split("|")


def _semtypes(index_dir):
    """{cui: set(semtypes)} for all CUIs; cuist/postings is one long 'CUI|st' run."""
    out = defaultdict(set)
    path = os.path.join(index_dir, "indices", "cuist", "postings")
    with open(path, encoding="ascii", errors="replace") as fh:
        data = fh.read()
    for m in re.finditer(r"(C\d{7})\|([a-z]{4})", data):
        out[m.group(1)].add(m.group(2))
    return out


def build_crosswalk(vocab, index_dir=os.path.join(MML_DIR, INDEX_SUBDIR), out=CROSSWALK_PATH):
    """Write {cui -> one MedDRA LLT} as TSV and return it (see load_crosswalk).

    Ranking among a CUI's matching strings (lower is better):
      0 = the CUI's preferred name equals an LLT name
      1/2 = an exact synonym equals an LLT that is / is not itself a PT
      3/4 = a normalized synonym ("Diarrhea (finding)" -> "diarrhea") equals a PT / non-PT LLT
    Ties go to the PT matched by most strings, then to the shortest LLT name."""
    print(f"Building CUI -> MedDRA crosswalk from {index_dir} ...", file=sys.stderr)
    cands = defaultdict(dict)  # cui -> {llt_key: (rank, string, src)}

    def add(cui, key, rank, string, src):
        llt = vocab[key]
        rank += 0 if llt["llt_code"] == llt["pt_code"] else 1
        old = cands[cui].get(key)
        if old is None or rank < old[0]:
            cands[cui][key] = (rank, string, src)

    for r in _postings(index_dir, "cuisourceinfo"):
        if len(r) < 6:
            continue
        cui, string, src = r[0], r[3], r[4]
        low = string.lower()
        if low in vocab:
            add(cui, low, 1, string, src)
        else:
            for v in normalize(string):
                if v in vocab:
                    add(cui, v, 3, string, src)

    pref = {}
    for r in _postings(index_dir, "cuiconcept"):
        if len(r) >= 2 and r[0] in cands:
            pref[r[0]] = r[1]

    rows = {}
    for cui, opts in cands.items():
        p = pref.get(cui, "").lower()
        pt_votes = Counter(vocab[k]["pt_code"] for k in opts)

        def score(item):
            key, (rank, _, _) = item
            return (0 if key == p else rank, -pt_votes[vocab[key]["pt_code"]], len(key), key)
        key, (rank, string, src) = min(opts.items(), key=score)
        method = "preferred" if key == p else \
            {1: "synonym_pt", 2: "synonym", 3: "normalized_pt", 4: "normalized"}[rank]
        rows[cui] = {"cui": cui, "umls_name": pref.get(cui, ""), "llt_key": key,
                     "llt_name": vocab[key]["llt_name"], "pt_name": vocab[key]["pt_name"],
                     "map_method": method, "umls_string": string, "umls_src": src}

    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["cui", "umls_name", "llt_key", "llt_name", "pt_name", "map_method", "umls_string",
            "umls_src"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t")
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda x: x["cui"]))

    sts = _semtypes(index_dir)
    symptom_cuis = {c for c, s in sts.items() if s & set(SYMPTOM_STS)}
    mapped = symptom_cuis & set(rows)
    print(f"Crosswalk: {len(rows):,} CUIs -> MedDRA, written to {out}", file=sys.stderr)
    print(f"  symptom-type CUIs mapped: {len(mapped):,}/{len(symptom_cuis):,} "
          f"({len(mapped) / max(1, len(symptom_cuis)):.1%})", file=sys.stderr)
    print("  methods:", dict(Counter(r["map_method"] for r in rows.values())), file=sys.stderr)
    return rows


def load_crosswalk(vocab, path=CROSSWALK_PATH, rebuild=False):
    """Return {cui: crosswalk row}; build the TSV first if it is missing. Rows whose LLT is no
    longer in vocab (e.g. a different MedDRA version) are skipped."""
    if rebuild or not os.path.exists(path):
        return build_crosswalk(vocab, out=path)
    with open(path, encoding="utf-8") as fh:
        rows = {r["cui"]: r for r in csv.DictReader(fh, delimiter="\t")
                if r["llt_key"] in vocab}
    print(f"Loaded {len(rows):,} CUI -> MedDRA mappings from {path}", file=sys.stderr)
    return rows


def choose_concept(concepts, crosswalk):
    """Pick one concept for an entity: a MedDRA-mapped one if any, by semantic-type priority.
    Returns (concept, crosswalk_row or None)."""
    def st_rank(c):
        return min((ST_PRIORITY.get(s, len(ST_PRIORITY)) for s in c["semtypes"]),
                   default=len(ST_PRIORITY))
    ranked = sorted(concepts, key=lambda c: (c["cui"] not in crosswalk, st_rank(c)))
    best = ranked[0]
    return best, crosswalk.get(best["cui"])


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crosswalk", action="store_true", help="(re)build the CUI -> MedDRA TSV")
    ap.add_argument("--text", help="run MetaMapLite on one text and show the MedDRA mapping")
    ap.add_argument("--meddra-dir", default=os.path.join(HERE, "meddra_29_1"))
    args = ap.parse_args()
    from meddra_utils import load_meddra
    vocab = load_meddra(args.meddra_dir)
    xw = load_crosswalk(vocab, rebuild=args.crosswalk)
    if args.text:
        for h in run_metamaplite([args.text])[0]:
            c, row = choose_concept(h["concepts"], xw)
            target = f"{row['llt_name']} (PT {row['pt_name']}, {row['map_method']})" if row \
                else "UNMAPPED"
            print(f"{h['text']!r:25} {c['cui']} {c['preferred_name']} {c['semtypes']} -> {target}")


if __name__ == "__main__":
    main()
