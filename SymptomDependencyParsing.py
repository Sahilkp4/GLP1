"""Extract drug-attributed, non-negated symptoms from GLP-1 Reddit posts and code them to MedDRA.

Pipeline per post (input columns: drug_name, body):
  1. Tag the target drug: case-insensitive regex -> doc.char_span(alignment_mode="expand")
     -> merge into a single PROPN token before parsing. Other drug names are tagged too,
     so we know which drug was mentioned most recently.
  2. Symptom candidates: PhraseMatcher(attr="LOWER") over current MedDRA LLT names, plus a
     rapidfuzz edit-distance fallback for misspellings ("nausua" -> "Nausea"). With --metamap,
     MetaMapLite UMLS concepts are added on text the MedDRA matcher did not cover; each CUI
     is mapped to a MedDRA LLT through its UMLS synonyms (metamap_utils.py).
  3. Drug-linked sentences: a sentence counts if it names the drug, or if an anaphor in it
     ("It gave me...", "This medication made me...") refers back to the drug, found by
     coreferee and/or dependency rules. Links only carry while the target is the most
     recently mentioned drug.
  4. Dependency-based attribution + negation: clause-verb `neg` child, "no" determiners
     (propagated across and/or coordination), "without"/"free of"/"no longer".
  5. Surviving symptoms are coded to MedDRA LLT -> PT -> HLT -> HLGT -> SOC (codes + names,
     primary SOC path).

Usage:
  python SymptomDependencyParsing.py --selftest
  python SymptomDependencyParsing.py --input glp1_single_drug_5pct_subset.csv \
      --nrows 50 --debug        # uses meddra_29_1/ (export it with mdb_export.py)
  python SymptomDependencyParsing.py --nrows 50 --debug --metamap   # + MetaMapLite
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd
import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

from meddra_utils import load_meddra
from metamap_utils import MML_DIR, choose_concept

# MedDRA coding written for every symptom: LLT -> PT -> HLT -> HLGT -> SOC (primary SOC path).
MEDDRA_COLS = ["llt_code", "llt_name", "pt_code", "pt_name", "hlt_code", "hlt_name",
               "hlgt_code", "hlgt_name", "soc_code", "soc_name", "soc_abbrev"]
# UMLS fields, filled for MetaMapLite candidates only.
UMLS_COLS = ["cui", "umls_name", "semtypes", "map_method"]

try:
    from wordfreq import zipf_frequency
except ImportError:  # fall back to vector OOV check
    zipf_frequency = None

# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------
GLP1_DRUGS = [  # same list as make_subset.py
    "semaglutide", "ozempic", "wegovy", "rybelsus",
    "tirzepatide", "mounjaro", "zepbound",
    "liraglutide", "saxenda", "victoza",
    "dulaglutide", "trulicity",
    "exenatide", "byetta", "bydureon",
    "lixisenatide", "adlyxin",
    "retatrutide",
]
OTHER_DRUGS = [
    "metformin", "glucophage", "insulin", "lantus", "humalog", "novolog", "tresiba",
    "jardiance", "farxiga", "invokana", "januvia", "glipizide", "glimepiride",
    "phentermine", "topiramate", "topamax", "qsymia", "contrave", "naltrexone",
    "bupropion", "wellbutrin", "zofran", "ondansetron", "phenergan", "reglan",
    "miralax", "colace", "dulcolax", "imodium", "pepto", "tums", "omeprazole",
    "prilosec", "pantoprazole", "famotidine", "pepcid", "levothyroxine", "synthroid",
    "lisinopril", "losartan", "atorvastatin", "lipitor", "spironolactone",
    "birth control", "testosterone", "trt", "ibuprofen", "advil", "tylenol", "acetaminophen",
]

# Nouns that, with a determiner, refer back to the medication ("this medication").
DRUG_NOUNS = {"medication", "medicine", "med", "meds", "drug", "shot", "injection", "pen",
              "jab", "dose", "stuff", "pill", "glp-1", "glp1"}
ANAPHOR_DETS = {"this", "the", "that", "my", "these"}
ANAPHOR_PRONOUNS = {"it", "this", "that"}
ANAPHOR_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "dative"}
PLEONASTIC_HEADS = {"be", "seem", "appear", "take", "happen", "look"}
TIME_WORDS = {"week", "day", "month", "year", "hour", "minute", "time", "while"}

NEG_DET_WORDS = {"no", "zero", "nor", "neither"}
DEGREE_NOUNS = {"level", "amount", "degree", "intensity", "kind", "type"}

Token.set_extension("drug_role", default=None, force=True)  # "target" | "other" | None
Doc.set_extension("target_drug", default=None, force=True)


def demo_vocab():
    """Tiny placeholder vocabulary for --selftest only (no official codes)."""
    names = {"Nausea": "Nausea", "Headache": "Headache", "Dizzy": "Dizziness",
             "Dizziness": "Dizziness", "Diarrhea": "Diarrhoea", "Vomiting": "Vomiting",
             "Fatigue": "Fatigue", "Constipation": "Constipation", "Heartburn": "Dyspepsia"}
    vocab = {n.lower(): {"llt_code": "DEMO", "llt_name": n, "pt_code": "DEMO", "pt_name": pt,
                         "soc_name": "DEMO"} for n, pt in names.items()}
    # Terms for the term-level filters, with their real MedDRA SOCs.
    for n, pt, soc in [("Weight", "Weight", "Investigations"),
                       ("Weight loss", "Weight decreased", "Investigations"),
                       ("Surgery", "Surgery", "Surgical and medical procedures"),
                       ("Eructation", "Eructation", "Gastrointestinal disorders"),
                       ("ADD", "Attention deficit hyperactivity disorder", "Psychiatric disorders")]:
        vocab[n.lower()] = {"llt_code": "DEMO", "llt_name": n, "pt_code": "DEMO", "pt_name": pt,
                            "soc_name": soc}
    return vocab


# --------------------------------------------------------------------------------------
# spaCy pipeline
# --------------------------------------------------------------------------------------
def _drug_regex(names):
    names = sorted(set(names), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in names) + r")(?![\w-])",
                      re.IGNORECASE)


ALL_DRUGS_RE = _drug_regex(GLP1_DRUGS + OTHER_DRUGS)
NOT_INSULIN_DRUG_RE = re.compile(r"\s+(resistan\w*|sensitiv\w*|levels?|spikes?|production|response)\b", re.I)


def tag_and_merge_drugs(doc):
    """Step 1 (runs on the tokenized doc, before parsing)."""
    target = (doc._.target_drug or "").lower()
    spans = []
    for m in ALL_DRUGS_RE.finditer(doc.text):
        if m.group(1).lower() == "insulin" and NOT_INSULIN_DRUG_RE.match(doc.text, m.end()):
            continue  # "insulin resistance", "insulin levels" are not the drug
        span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
        if span is not None:
            spans.append((span, "target" if m.group(1).lower() == target else "other"))
    keep = {(s.start, s.end) for s in filter_spans([s for s, _ in spans])}
    spans = [(s, role) for s, role in spans if (s.start, s.end) in keep]
    with doc.retokenize() as retok:
        for span, _ in spans:
            if len(span) > 1:
                retok.merge(span, attrs={"POS": "PROPN", "TAG": "NNP"})
    # After merging, token indices changed; re-find by character offset.
    for span, role in spans:
        tok = doc.char_span(span.start_char, span.end_char, alignment_mode="expand")
        if tok is not None:
            for t in tok:
                t._.drug_role = role
                t.pos_, t.tag_ = "PROPN", "NNP"
    return doc


@Language.component("drug_attr_fix")
def drug_attr_fix(doc):
    """Re-assert PROPN/NNP + lemma after the tagger, and mark drugs as PRODUCT entities
    so coreferee treats them as non-person antecedents for "it"."""
    drug_spans = []
    for t in doc:
        if t._.drug_role:
            t.pos_, t.tag_, t.lemma_ = "PROPN", "NNP", t.lower_
            drug_spans.append(Span(doc, t.i, t.i + 1, label="PRODUCT"))
    if drug_spans:
        doc.ents = filter_spans(drug_spans + [e for e in doc.ents])
    return doc


def build_nlp(model="en_core_web_lg", use_coreferee=True):
    nlp = spacy.load(model)
    nlp.add_pipe("drug_attr_fix", last=True)
    coref_ok = False
    if use_coreferee:
        try:
            import coreferee  # noqa: F401
            nlp.add_pipe("coreferee")
            coref_ok = True
        except Exception as e:  # pragma: no cover
            print(f"[warn] coreferee unavailable ({e}); using dependency rules only",
                  file=sys.stderr)
    return nlp, coref_ok


# --------------------------------------------------------------------------------------
# Step 2: symptom candidates
# --------------------------------------------------------------------------------------
class SymptomFinder:
    def __init__(self, nlp, vocab, max_dist_short=1, max_dist_long=2, common_zipf=3.0,
                 crosswalk=None):
        self.vocab = vocab
        self.crosswalk = crosswalk or {}  # CUI -> MedDRA LLT (metamap_utils.load_crosswalk)
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        names = list(vocab)
        for i in range(0, len(names), 5000):
            self.matcher.add("LLT", list(nlp.tokenizer.pipe(names[i:i + 5000])))
        # Second pass on LEMMA so inflected forms ("headaches", "migraines") still match.
        # Each lemma pattern maps back to its LLT name.
        self.lemma_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        self.lemma_key = {}
        with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
            for name, pdoc in zip(names, nlp.pipe(names, batch_size=1000)):
                key = f"LLT::{name}"
                self.lemma_key[nlp.vocab.strings.add(key)] = name
                self.lemma_matcher.add(key, [pdoc])
        # Fuzzy index: bucket LLT names by word count for unigram/bigram lookups.
        self.by_words = {1: [], 2: []}
        for n in names:
            k = len(n.split())
            if k in self.by_words and n.replace(" ", "").isalpha():
                self.by_words[k].append(n)
        self.max_dist_short, self.max_dist_long = max_dist_short, max_dist_long
        self.common_zipf = common_zipf
        self.llt_words = {w for n in names for w in n.split()}

    def _is_real_word(self, tok):
        w = tok.lower_
        if w in self.llt_words:
            return True
        if zipf_frequency is not None:
            return zipf_frequency(w, "en") >= self.common_zipf
        return not tok.is_oov

    def find(self, doc, mml_hits=None):
        exact =[Span(doc, s, e) for _, s, e in self.matcher(doc)]
        exact = [s for s in filter_spans(exact) if not any(t._.drug_role for t in s)]
        cands = [{"span": s, "llt": self.vocab[s.text.lower()], "match_type": "exact",
                  "fuzzy_dist": 0} for s in exact]
        covered = {t.i for s in exact for t in s}

        lemma_hits = [Span(doc, s, e, label=m) for m, s, e in self.lemma_matcher(doc)]
        for s in filter_spans(lemma_hits):
            if any(t.i in covered or t._.drug_role for t in s):
                continue
            cands.append({"span": s, "llt": self.vocab[self.lemma_key[s.label]],
                          "match_type": "lemma", "fuzzy_dist": 0})
            covered.update(t.i for t in s)

        fuzzy = []
        for n in (2, 1):
            for i in range(len(doc) - n + 1):
                toks = doc[i:i + n]
                if any(t.i in covered or t._.drug_role or not t.is_alpha or t.is_stop
                       for t in toks):
                    continue
                if all(self._is_real_word(t) for t in toks):
                    continue  # only look for misspellings
                text = " ".join(t.lower_ for t in toks)
                if len(text.replace(" ", "")) < 5:
                    continue
                max_d = self.max_dist_short if len(text) < 8 else self.max_dist_long
                hit = process.extractOne(text, self.by_words[n],
                                         scorer=DamerauLevenshtein.distance,
                                         score_cutoff=max_d)
                if hit:
                    fuzzy.append({"span": toks, "llt": self.vocab[hit[0]],
                                  "match_type": "fuzzy", "fuzzy_dist": int(hit[1])})
                    covered.update(t.i for t in toks)
        return cands + fuzzy + self._metamap(doc, mml_hits or [], covered)

    def _metamap(self, doc, hits, covered):
        """MetaMapLite concepts on tokens the MedDRA passes did not cover (MedDRA first)."""
        spans, hit_of = [], {}
        for h in hits:
            span = doc.char_span(h["start"], h["end"], alignment_mode="expand")
            if span is None or (span.start, span.end) in hit_of:
                continue
            spans.append(span)
            hit_of[(span.start, span.end)] = h
        out = []
        for span in filter_spans(spans):  # overlapping hits: keep the longest
            if any(t.i in covered or t._.drug_role for t in span):
                continue
            concept, row = choose_concept(hit_of[(span.start, span.end)]["concepts"],
                                          self.crosswalk)
            umls = {"cui": concept["cui"], "umls_name": concept["preferred_name"],
                    "semtypes": ",".join(concept["semtypes"]),
                    "map_method": row["map_method"] if row else ""}
            llt = self.vocab[row["llt_key"]] if row else {f: "" for f in MEDDRA_COLS}
            out.append({"span": span, "llt": llt, "match_type": "metamap", "fuzzy_dist": 0,
                        "umls": umls, "umls_string": concept["concept_string"],
                        "unmapped": row is None})
            covered.update(t.i for t in span)
        return out


# --------------------------------------------------------------------------------------
# Step 3: drug-linked sentences
# --------------------------------------------------------------------------------------
def _is_pleonastic_it(tok):
    if tok.lower_ != "it":
        return False
    head = tok.head
    if head.lemma_ not in PLEONASTIC_HEADS:
        return False
    for c in head.children:
        if c.dep_ in {"ccomp", "xcomp", "csubj"}:
            return True  # "it is hard to eat", "it takes time to adjust"
        if any(g.dep_ in {"xcomp", "ccomp"} for g in c.children):
            return True
        if c.dep_ in {"attr", "npadvmod", "dobj"} and c.lemma_ in TIME_WORDS:
            return True  # "it's been 3 weeks"
    return False


def is_anaphor(tok):
    if tok.dep_ not in ANAPHOR_DEPS:
        return False
    if tok.pos_ == "PRON" and tok.lower_ in ANAPHOR_PRONOUNS and tok.tag_ != "WDT":
        return not _is_pleonastic_it(tok)
    if tok.pos_ in {"NOUN", "PROPN"} and tok.lemma_.lower() in DRUG_NOUNS | {tok.lower_}:
        if tok.lower_ in DRUG_NOUNS or tok.lemma_.lower() in DRUG_NOUNS:
            return any(c.dep_ in {"det", "poss"} and c.lower_ in ANAPHOR_DETS
                       for c in tok.children)
    return False


def coref_target_tokens(doc, coref_ok):
    """Token indices that coreferee puts in a chain with the target drug."""
    if not coref_ok or not doc.has_extension("coref_chains") or doc._.coref_chains is None:
        return set()
    out = set()
    for chain in doc._.coref_chains:
        mentions = [list(m.token_indexes) for m in chain]
        if any(doc[i]._.drug_role == "target" for m in mentions for i in m):
            out.update(i for m in mentions for i in m if not doc[i]._.drug_role)
    return out


def link_sentences(doc, coref_ok, mode="both", scope="word"):
    """Return list of (sent, anchors[list of Token], methods[set], linked[bool], active[bool]).

    anchors are the target drug's mentions in the sentence: the drug name ("explicit"), or a
    word referring back to it ("coreferee" chain, or "dep_rule" for it/this/this medication).
    active: the target was the most recently mentioned drug when the sentence started.

    scope="word" (default): a sentence with anchors is "linked", but process_doc then requires
    each symptom to depend on an anchor in the parse tree. A sentence without anchors is
    linked only while the target drug is active ("carryover"). Before the drug's first
    mention, a pronoun can still point forward to it ("dep_rule_cataphora": "I think it's
    increasing my migraines. I didn't realize Ozempic was a shot.").
    scope="sentence": a sentence is linked only via an explicit mention or an anaphor.
    scope="post": each post names exactly one target drug, so every sentence is linked
    by default ("post_scope") unless a NON-target drug was the most recent drug mentioned
    or the sentence itself names one (e.g. "I'm on phentermine...", "Miralax helped").
    """
    coref_idx = coref_target_tokens(doc, coref_ok) if mode in {"both", "coreferee"} else set()
    use_rules = mode in {"both", "rules"}
    first_drug = next((t._.drug_role for t in doc if t._.drug_role), None)
    last_drug = None
    out = []
    for sent in doc.sents:
        last_at_start = last_drug
        anchors, methods = [], set()
        for tok in sent:
            if tok._.drug_role == "target":
                anchors.append(tok)
                methods.add("explicit")
                last_drug = "target"
            elif tok._.drug_role == "other":
                last_drug = "other"
            elif last_drug == "target":  # link carries only while target is most recent
                if tok.i in coref_idx and tok.pos_ in {"PRON", "NOUN", "PROPN"}:
                    anchors.append(tok)
                    methods.add("coreferee")
                elif use_rules and is_anaphor(tok):
                    anchors.append(tok)
                    methods.add("dep_rule")
            elif scope == "word" and last_drug is None and first_drug == "target" and \
                    use_rules and tok.pos_ == "PRON" and is_anaphor(tok):
                anchors.append(tok)  # pronoun before the drug's first mention
                methods.add("dep_rule_cataphora")
                last_drug = "target"  # the pronoun makes the drug active from here on
        linked = bool(anchors)
        active = last_at_start == "target"
        if not linked and scope == "word" and active:
            linked = True
            methods.add("carryover")
        if not linked and scope == "post" and last_at_start != "other" and \
                not any(t._.drug_role == "other" for t in sent):
            linked = True
            methods.add("post_scope")
        out.append((sent, anchors, methods, linked, active))
    return out


# --------------------------------------------------------------------------------------
# Step 4: attribution + negation via the dependency tree
# --------------------------------------------------------------------------------------
def tree_distance(a, b):
    path_a = {}
    t, d = a, 0
    while True:
        path_a[t.i] = d
        if t.head.i == t.i:
            break
        t, d = t.head, d + 1
    t, d = b, 0
    while True:
        if t.i in path_a:
            return path_a[t.i] + d
        if t.head.i == t.i:
            return 99
        t, d = t.head, d + 1


def _coord_head(tok):
    """First conjunct of a coordination ("no headache or nausea" -> headache)."""
    while tok.dep_ == "conj" and tok.head.i != tok.i:
        tok = tok.head
    return tok


def _has_neg_child(tok):
    return any(c.dep_ == "neg" for c in tok.children)


def _is_clause_verb(t):
    # Gerund symptoms ("without vomiting") are VERB/pcomp; keep climbing past them.
    return t.pos_ in {"VERB", "AUX"} and not (t.tag_ == "VBG" and t.dep_ in {"pcomp", "pobj"})


def clause_verb(tok):
    """The verb heading tok's clause (or the sentence root for verbless fragments)."""
    while not _is_clause_verb(tok) and tok.head.i != tok.i:
        tok = tok.head
    return tok


# Arcs that join clauses describing the same situation: "took Ozempic AND woke up with
# nausea", "started TO FEEL sick", "it made me tired, fatigue IS real", "AFTER I took it".
# relcl is left out: a relative clause describes a noun, not the drug's effect.
CLAUSE_LINKS = {"conj", "xcomp", "ccomp", "advcl", "parataxis"}


def clause_group(verb):
    """Clause verbs reachable from verb through CLAUSE_LINKS arcs (up or down)."""
    group, todo = {verb.i}, [verb]
    while todo:
        v = todo.pop()
        nxt = [c for c in v.children if c.dep_ in CLAUSE_LINKS]
        if v.dep_ in CLAUSE_LINKS and v.head.i != v.i:
            nxt.append(v.head)
        for n in nxt:
            n = clause_verb(n)
            if n.i not in group:
                group.add(n.i)
                todo.append(n)
    return group


def drug_dependency(span, anchors):
    """Word-level link: is the matched symptom connected to a drug anchor in the parse tree?
    Returns (anchor, relation) for the closest connected anchor, or None.
      direct        - one is the other's ancestor ("nausea from Ozempic", "Ozempic nausea")
      same_clause   - same clause verb ("Ozempic gave me nausea", "On Wegovy my nausea is bad")
      linked_clause - clauses joined by CLAUSE_LINKS ("... and it is causing heartburn")"""
    root = span.root
    root_verb = clause_verb(root)
    group = None
    best = None
    for a in anchors:
        if a in root.ancestors or root in a.ancestors:
            rel = "direct"
        elif clause_verb(a) == root_verb:
            rel = "same_clause"
        else:
            group = group if group is not None else clause_group(root_verb)
            rel = "linked_clause" if clause_verb(a).i in group else None
        if rel:
            d = tree_distance(root, a)
            if best is None or d < best[0]:
                best = (d, a, rel)
    return (best[1], best[2]) if best else None


# Personal experience: who had the symptom?
PERSONAL_PRONOUNS = {"i", "me", "myself", "he", "him", "himself", "she", "her", "herself"}
GENERIC_PERSONS = {"you", "yourself", "ya", "u", "we", "us", "ourselves", "they", "them",
                   "themselves", "one", "anyone", "anybody", "everyone", "everybody",
                   "someone", "somebody", "nobody", "people", "patient", "user", "folk",
                   "man", "woman", "person", "ppl"}
PERSONAL_POSS = {"my", "his", "her"}
# Nouns that name a specific person when possessed ("my husband"; not "my entire life").
PERSON_NOUNS = {"husband", "wife", "partner", "spouse", "boyfriend", "girlfriend", "fiance",
                "fiancee", "mom", "mum", "mother", "dad", "father", "parent", "son",
                "daughter", "kid", "child", "baby", "sister", "brother", "sibling", "friend",
                "bf", "gf", "aunt", "uncle", "grandma", "grandmother", "grandpa",
                "grandfather", "cousin", "niece", "nephew", "coworker", "colleague",
                "roommate", "neighbor", "neighbour", "family", "cat", "dog"}
GENERIC_POSS = {"your", "their", "our", "ur"}
EXPERIENCER_DEPS = ["dobj", "dative", "iobj", "obj", "nsubj", "nsubjpass"]  # objects first
INHERIT_DEPS = {"conj", "xcomp", "advcl", "ccomp"}  # clauses that can share a subject


def _person_kind(tok):
    """'personal' | 'generic' | None for a candidate experiencer token."""
    if tok._.drug_role or is_anaphor(tok):
        return None
    w, lemma = tok.lower_, tok.lemma_.lower()
    if w in PERSONAL_PRONOUNS:
        return "personal"
    if w in GENERIC_PERSONS or lemma in GENERIC_PERSONS:
        return "generic"
    if tok.pos_ in {"NOUN", "PROPN"}:
        poss = {c.lower_ for c in tok.children if c.dep_ == "poss"}
        if poss & PERSONAL_POSS and lemma in PERSON_NOUNS:
            return "personal"  # "my husband"
        if poss & GENERIC_POSS:
            return "generic"
        if tok.ent_type_ == "PERSON":
            return "personal"  # a named person
    return None


def experiencer(span):
    """Return (kind, text): kind is 'personal', 'generic' or 'implicit' (no subject found,
    e.g. "Nauseous all day on Ozempic." -> treated as the poster)."""
    in_span = {t.i for t in span}
    # (a) a possessive on the symptom: "my nausea" / "your nausea"
    for tok in {*span, _coord_head(span.root)}:
        for c in tok.children:
            if c.dep_ == "poss":
                if c.lower_ in PERSONAL_POSS:
                    return "personal", c.text
                if c.lower_ in GENERIC_POSS:
                    return "generic", c.text
    # (b) subjects/objects from the symptom up to its clause verb, then up through clauses
    # that share a subject ("he took Ozempic and woke up with nausea" -> he)
    path, tok = [], span.root
    while True:
        path.append(tok)
        if _is_clause_verb(tok) or tok.head.i == tok.i:
            break
        tok = tok.head
    while True:
        for node in path:
            for dep in EXPERIENCER_DEPS:
                for c in node.children:
                    if c.dep_ == dep and c.i not in in_span:
                        kind = _person_kind(c)
                        if kind:
                            return kind, c.text
        v = path[-1]
        if v.dep_ not in INHERIT_DEPS or v.head.i == v.i:
            break
        path = [clause_verb(v.head)]
    return "implicit", ""


def negation_reason(span):
    root = span.root
    # (a) determiner negation on the noun, on any token of a multiword span ("no" can attach
    # to a compound: "no sulfur burps"), or on the head of its coordination
    for tok in {*span, _coord_head(root)}:
        for c in tok.children:
            if c.lower_ in NEG_DET_WORDS and c.dep_ in {"det", "neg", "nummod", "amod",
                                                        "advmod", "preconj", "cc"}:
                if c.dep_ == "cc" and c.lower_ not in {"nor", "neither"}:
                    continue
                return "negated:det"
    # (b) climb to the clause verb; check "without"/"free of" along the way
    tok = root
    comparative = False
    while not _is_clause_verb(tok) and tok.head.i != tok.i:
        tok = tok.head
        if tok.lower_ == "without":
            return "negated:without"
        if tok.lower_ == "of" and tok.head.lower_ in {"free", "rid"}:
            return "negated:without"
        # "never experienced THIS LEVEL OF nausea" asserts the symptom (intensity
        # comparison), so the clause negation does not apply.
        if tok.lemma_.lower() in DEGREE_NOUNS and \
                any(c.dep_ in {"det", "amod"} and c.lower_ in {"this", "that", "such"}
                    for c in tok.children):
            comparative = True
    verb = tok
    if comparative:
        return None
    if _has_neg_child(verb) or any(_has_neg_child(c) for c in verb.children if c.dep_ in {"aux", "auxpass"}):
        return "negated:clause"
    for c in verb.children:  # "no longer have nausea", "no more nausea"
        if c.dep_ == "advmod" and c.lower_ in {"longer", "more"} and \
                any(g.lower_ == "no" for g in c.children):
            return "negated:no_longer"
    return None


# --------------------------------------------------------------------------------------
# Step 4b: did the person actually take the drug? (uptake)
# --------------------------------------------------------------------------------------
# Verbs whose object is the drug: "I started Saxenda", "took my shot", "never did Ozempic".
USE_VERBS = {"take", "start", "use", "inject", "try", "restart", "begin", "continue",
             "switch", "titrate", "prescribe", "put", "do", "stop", "quit", "discontinue",
             "pause", "skip", "miss", "dose", "be", "go", "stay", "get", "come"}
STOP_VERBS = {"stop", "quit", "discontinue", "pause"}
# Verbs with the drug as subject that imply someone took it: "It made me dizzy".
EFFECT_VERBS = {"give", "make", "cause", "increase", "trigger", "leave", "curb", "suppress",
                "reduce", "kill", "help", "hit"}
# Prepositions whose object is the drug and imply being on it: "on/since/after/from Ozempic".
ON_PREPS = {"on", "since", "after", "from"}
INTENT_VERBS = {"hope", "want", "plan", "wait", "consider", "wish", "need", "decide",
                "intend", "like", "love", "ask", "scare", "afraid", "nervous", "debate"}
FUTURE_AUX = {"will", "'ll", "wo", "shall", "would", "should", "might", "may", "gonna"}
HYPOTHETICAL_MARKS = {"if", "unless"}       # always hypothetical
CONDITIONAL_MARKS = {"once", "when", "until"}  # hypothetical only in the present tense
TAKEN = ("current", "past", "stopped")


def _future_or_hypothetical(v):
    """True if clause verb v describes an intended/future/hypothetical event."""
    t = v
    while True:
        for c in t.children:
            if c.dep_ in {"aux", "auxpass"} and c.lower_ in FUTURE_AUX:
                return True  # "will start", "should I take"
            if c.dep_ == "advmod" and c.lower_ in {"soon", "tomorrow", "eventually"}:
                return True  # "Starting Ozempic soon"
            if c.dep_ == "npadvmod" and any(g.lower_ in {"next", "coming"} for g in c.children):
                return True  # "next week"
            if c.dep_ == "mark" and (c.lower_ in HYPOTHETICAL_MARKS or (
                    c.lower_ in CONDITIONAL_MARKS and "Pres" in t.morph.get("Tense")
                    and t.tag_ != "VBG")):
                return True  # "if I take it", "once I stop the meds"
        if t.dep_ in {"xcomp", "ccomp"} and t.head.i != t.i:
            h = t.head
            if h.lemma_.lower() in INTENT_VERBS or \
                    (h.lemma_ in {"go", "think"} and h.tag_ == "VBG"):
                return True  # "hoping to be put on", "going to start", "thinking of trying"
            t = h
            continue
        return False


def _is_negated_verb(v):
    return _has_neg_child(v) or any(_has_neg_child(c) for c in v.children
                                    if c.dep_ in {"aux", "auxpass"}) or \
        any(c.lower_ == "never" for c in v.children)


def _person_class(kind, who):
    """Map an experiencer to 'poster' | 'other' | None (generic)."""
    if kind == "generic":
        return None
    if kind == "implicit" or who.lower() in {"i", "me", "my", "myself", "mine"}:
        return "poster"
    return "other"


def uptake_mentions(doc, sent_info):
    """Find mentions showing someone took (or did not take) the target drug.
    Returns a list of dicts: {person: poster|other, status, evidence}.
    status: current | past | stopped (taken) or future | negated (not taken)."""
    anchors = [a for info in sent_info for a in info[1]]
    out = []
    for a in anchors:
        t = _coord_head(a)  # "on phentermine and mounjaro" -> phentermine's slot
        if t.dep_ in {"compound", "amod", "nmod"} and t.head.lemma_.lower() in DRUG_NOUNS:
            t = t.head      # "my Ozempic shot"
        kind, prep = None, None
        if t.dep_ in {"dobj", "obj"} and t.head.lemma_.lower() in USE_VERBS:
            v, kind = t.head, "use"
        elif t.dep_ == "pobj" and t.head.lower_ in ON_PREPS | {"off", "to", "with"}:
            prep = t.head
            v = clause_verb(prep)
            if prep.lower_ in ON_PREPS:
                kind = "on"
            elif prep.lower_ == "off":
                kind = "off"
            elif v.lemma_.lower() in USE_VERBS:  # "switched to", "started with"
                kind = "use"
        elif t.dep_ == "nsubj" and t.head.lemma_.lower() in EFFECT_VERBS:
            v, kind = t.head, "effect"
        poss = [c for c in (*a.children, *t.children) if c.dep_ == "poss"]
        if kind is None and any(c.lower_ in PERSONAL_POSS for c in poss):
            v, kind = clause_verb(t), "poss"  # "my ozempic arrived", "his dose"
        if kind is None:
            continue
        who_kind, who = experiencer(doc[a.i:a.i + 1])
        person = _person_class(who_kind, who)
        if person is None:
            continue  # "you can take it", "people on Ozempic"
        lemma = v.lemma_.lower()
        if kind == "on" and (prep.head != v or
                             lemma not in {"be", "go", "stay", "get", "start", "put"}):
            # "On Ozempic I never experienced...", "Week 3 on Ozempic: my headache isn't bad"
            # still mean the person is on it; only "I'm (not) on it" takes the verb's status
            status = "current"
        elif _future_or_hypothetical(v):
            status = "future"
        elif _is_negated_verb(v):
            status = "negated"  # "never did Ozempic", "I'm not on it"
        elif kind == "off" or lemma in STOP_VERBS:
            status = "stopped"
        elif "Past" in v.morph.get("Tense"):
            status = "past"
        else:
            status = "current"
        out.append({"person": person, "status": status,
                    "evidence": f"{status}: {v.text} {a.text} ({who or who_kind})"})
    return out


def uptake_gate(symptom_kind, symptom_who, mentions):
    """(ok, label): ok if the symptom's experiencer took the drug. An implicit experiencer
    ("Again was feeling nauseous") matches anyone in the post who took it."""
    person = _person_class(symptom_kind, symptom_who)
    pool = [m for m in mentions if symptom_kind == "implicit" or m["person"] == person]
    taken = [m for m in pool if m["status"] in TAKEN]
    if taken:
        return True, taken[0]["evidence"]
    for status in ("future", "negated"):
        hit = next((m for m in pool if m["status"] == status), None)
        if hit:
            return False, hit["evidence"]
    return False, "none"


MODIFIED_NOUNS = DRUG_NOUNS | {"killer", "reliever", "relief", "med", "medication"}
IDIOM_RE = re.compile(r"\b(sick|tired) (and (sick|tired) )?of\b|\bof being (sick and )?(sick|tired)\b")


def non_experience_reason(span):
    """Mentions that are not symptoms the poster experienced."""
    root = span.root
    # "sick and tired of being sick and tired", "tired of dieting" -> fed up, not fatigue
    doc = span.doc
    window = doc[max(0, span.start - 4):min(len(doc), span.end + 3)].text.lower()
    if span.text.lower() in {"tired", "sick"} and IDIOM_RE.search(window):
        return "idiom"
    # "on Ozempic for weight loss", "medication for weight loss" -> indication / goal
    if root.head.lower_ == "for" and root.dep_ in {"pobj", "pcomp"}:
        return "indication"
    # "pain meds", "nausea medication" -> the term only modifies another noun
    if root.dep_ in {"compound", "amod", "nmod"} and root.head.lemma_.lower() in MODIFIED_NOUNS:
        return "modifier"
    return None


# Term-level filters: MedDRA covers far more than adverse events (tests, procedures, social
# circumstances...), so these drop LLTs that cannot be a side effect, whatever the context.
EXCLUDED_SOCS = {"Surgical and medical procedures", "Social circumstances", "Product issues"}
# An Investigations PT only describes a side effect when it reports a change ("Weight
# decreased"), not a bare test name ("Weight", "Blood glucose") or a result ("... positive").
INVESTIGATION_CHANGE_RE = re.compile(
    r"\b(increased|decreased|abnormal|fluctuation|irregular|prolonged|shortened)\b", re.I)
MAX_ACRONYM_LETTERS = 5
# PTs that say "a side effect happened" without naming one ("side effects" -> Adverse event).
GENERIC_PTS = {"Adverse event", "Adverse drug reaction"}


def _letters(s):
    return "".join(ch for ch in s if ch.isalpha())


def _is_acronym(term):
    term = _letters(term)
    return term.isupper() and len(term) <= MAX_ACRONYM_LETTERS


def term_reason(span, llt, umls_string=None):
    """Reject an LLT match because of the term itself (not its context). umls_string is the
    UMLS synonym MetaMapLite matched, for MetaMapLite candidates."""
    if llt.get("soc_name") in EXCLUDED_SOCS:
        return "excluded_soc"
    if llt.get("soc_name") == "Investigations" and \
            not INVESTIGATION_CHANGE_RE.search(llt.get("pt_name", "")):
        return "investigation_no_change"
    if llt.get("pt_name") in GENERIC_PTS:
        return "generic_term"
    # Short all-caps terms ("ADD", "PSA", "AIDS"; MetaMapLite's "PLAN", "CMPD") only count
    # when written in capitals, so "add"/"aids"/"plan" are not read as diagnoses.
    if (_is_acronym(llt.get("llt_name", "")) or (umls_string and _is_acronym(umls_string))) \
            and not _letters(span.text).isupper():
        return "lowercase_acronym"
    return None


def attribute_to_target(span, anchors, sent):
    """If another drug is in the same sentence, keep the symptom only if it is at least as
    close (in the dependency tree) to a target anchor as to the other drug."""
    others = [t for t in sent if t._.drug_role == "other"]
    if not others:
        return True
    if not anchors:
        return False
    d_target = min(tree_distance(span.root, a) for a in anchors)
    d_other = min(tree_distance(span.root, o) for o in others)
    return d_target <= d_other


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def process_doc(doc, finder, coref_ok, mode, scope="word", mml_hits=None, uptake=True):
    kept, rejected = [], []
    sent_info = link_sentences(doc, coref_ok, mode, scope)
    check_uptake = uptake and scope == "word"
    mentions = uptake_mentions(doc, sent_info) if check_uptake else []
    sent_of = {}
    for k, info in enumerate(sent_info):
        for t in info[0]:
            sent_of[t.i] = k
    for c in finder.find(doc, mml_hits):
        span = c["span"]
        sent, anchors, methods, linked, active = sent_info[sent_of[span.start]]
        rec = {"sentence": sent.text.strip(), "symptom_text": span.text,
               "match_type": c["match_type"], "fuzzy_dist": c["fuzzy_dist"],
               "link_method": "+".join(sorted(methods)), **c["llt"], **c.get("umls", {})}
        if c.get("unmapped"):  # MetaMapLite concept with no MedDRA equivalent
            rejected.append({**rec, "reason": "metamap_unmapped"})
            continue
        term = term_reason(span, c["llt"], c.get("umls_string"))
        if term:
            rejected.append({**rec, "reason": term})
            continue
        if scope == "word":
            # Word level: the symptom itself must depend on a drug anchor in the sentence.
            # Sentences with no anchor keep symptoms only while the target drug is active.
            if anchors:
                dep = drug_dependency(span, anchors)
                if dep is None:
                    rejected.append({**rec, "reason": "not_drug_dependent"})
                    continue
                rec.update(drug_anchor=dep[0].text, dep_relation=dep[1])
            elif active:
                rec.update(drug_anchor="", dep_relation="carryover")
            else:
                rejected.append({**rec, "reason": "not_drug_linked"})
                continue
        elif not linked:
            rejected.append({**rec, "reason": "not_drug_linked"})
            continue
        if not attribute_to_target(span, anchors, sent):
            rejected.append({**rec, "reason": "attributed_to_other_drug"})
            continue
        if scope == "word":
            kind, who = experiencer(span)
            rec["experiencer"] = who or kind
            if kind == "generic":  # "you will be hungry", "people get headaches"
                rejected.append({**rec, "reason": "not_personal"})
                continue
        neg = negation_reason(span) or non_experience_reason(span)
        if neg:
            rejected.append({**rec, "reason": neg})
            continue
        if check_uptake:  # did this person actually take the drug?
            ok, evidence = uptake_gate(kind, who, mentions)
            rec["uptake"] = evidence
            if not ok:
                status = evidence.split(":")[0] if evidence != "none" else "none"
                rejected.append({**rec, "reason": f"no_uptake:{status}"})
                continue
        kept.append(rec)
    return kept, rejected


def run(texts_drugs, nlp, finder, coref_ok, mode="both", scope="word", mml_hits=None,
        uptake=True):
    """texts_drugs: (text, drug) pairs. mml_hits: optional list of MetaMapLite hit lists,
    aligned with texts_drugs (character offsets into the same text)."""
    hits = iter(mml_hits) if mml_hits is not None else None

    def make_docs():
        for text, drug in texts_drugs:
            doc = nlp.make_doc(text)  # drug merging below keeps doc.text, so offsets hold
            doc._.target_drug = drug
            yield tag_and_merge_drugs(doc)
    for doc in nlp.pipe(make_docs(), batch_size=32):
        yield doc, process_doc(doc, finder, coref_ok, mode, scope,
                               next(hits) if hits is not None else None, uptake)


def _fake_hit(text, phrase, cui, name, st="sosy"):
    """A synthetic MetaMapLite hit for --selftest (no Java needed)."""
    i = text.index(phrase)
    return {"start": i, "end": i + len(phrase), "text": phrase,
            "concepts": [{"cui": cui, "preferred_name": name, "concept_string": name,
                          "semtypes": [st], "sources": []}]}


def selftest(nlp, coref_ok, mode, scope="word"):
    # Fake crosswalk: C9000001 -> Eructation, C0027497 -> Nausea; C9000002 is unmapped.
    xw = {"C9000001": {"llt_key": "eructation", "map_method": "preferred"},
          "C0027497": {"llt_key": "nausea", "map_method": "preferred"},
          "C9000003": {"llt_key": "headache", "map_method": "synonym"}}
    finder = SymptomFinder(nlp, demo_vocab(), crosswalk=xw)
    mml_cases = [  # (text, drug, expected kept (llt_name, match_type), hit, expected reasons)
        ("On Ozempic I get really bad sulfur burps.", "ozempic",
         [("Eructation", "metamap")], ("sulfur burps", "C9000001", "Eructation"), []),
        ("Ozempic gave me nausea.", "ozempic",          # exact MedDRA match wins the overlap
         [("Nausea", "exact")], ("nausea", "C0027497", "Nausea"), []),
        ("Wegovy left me with brain fog.", "wegovy",
         [], ("brain fog", "C9000002", "Brain fog"), ["metamap_unmapped"]),
        ("Mounjaro and no sulfur burps at all.", "mounjaro",   # our negation still applies
         [], ("sulfur burps", "C9000001", "Eructation"), ["negated:det"]),
        ("Ozempic is part of my plan.", "ozempic",      # "plan" matched via acronym "PLAN"
         [], ("plan", "C9000003", "PLAN"), ["lowercase_acronym"]),
    ]
    cases = [
        ("Week 3 on Ozempic: my headache isnt bad but the nausua is awful.",
         "ozempic", {"Nausea"}),
        ("I started Ozempic last week. My headache isnt bad.", "ozempic", set()),
        ("On Wegovy my nausea is bad and it is causing heartburn too.", "wegovy",
         {"Nausea", "Heartburn"}),
        ("Ozempic gave me no headache or nausea.", "ozempic", set()),
        ("I'm on Mounjaro now. It made me dizzy.", "mounjaro", {"Dizzy"}),
        ("Took Wegovy for a month. Then I switched to metformin. It gave me diarrhea.",
         "wegovy", set()),
        ("Week two on Ozempic without vomiting so far.", "ozempic", set()),
        ("Ozempic is fine. Phentermine gave me a headache.", "ozempic", set()),
        ("Started Ozempic. Feeling sick and tired of being sick and tired.", "ozempic", set()),
        ("Low calorie diets made me hungry. On Ozempic I never experienced this level of "
         "nausea.", "ozempic", {"Nausea"}),
        ("I think it's increasing my headaches. I didn't realize Ozempic was a shot.",
         "ozempic", {"Headache"}),
        ("Saxenda is great. This medication made me so tired, fatigue is real.",
         "saxenda", {"Fatigue"}),
        ("I don't have any constipation on Trulicity.", "trulicity", set()),
        ("I've been on Ozempic for weight loss and the headaches are constant.", "ozempic",
         {"Headache"}),
        ("Mounjaro plus my pain meds.", "mounjaro", set()),
        # Term-level filters
        ("On Ozempic my weight is stable but I had weight loss of 10 lbs.", "ozempic",
         {"Weight loss"}),                                   # bare "weight" dropped
        ("Mounjaro worked fine until my surgery.", "mounjaro", set()),       # excluded SOC
        ("Since Wegovy I added walking and my ADD is worse.", "wegovy", {"ADD"}),  # "added" dropped
    ]
    if scope == "word":  # word-level dependency + personal experience (optional 4th: reasons)
        cases += [
            ("On Ozempic you will get nausea.", "ozempic", set(), ["not_personal"]),
            ("My husband is on Ozempic and he has nausea.", "ozempic", {"Nausea"}),
            ("People on Wegovy get headaches.", "wegovy", set(), ["not_personal"]),
            ("I had a headache yesterday. I'm hoping to start Ozempic.", "ozempic", set(),
             ["not_drug_linked"]),                            # drug not active yet
            ("I started Ozempic last month. Now I have nausea every morning.", "ozempic",
             {"Nausea"}),                                     # carryover: drug is active
            ("He took the Ozempic at night and woke up with nausea.", "ozempic",
             {"Nausea"}),                                     # linked_clause via "and"
            ("The doctor who prescribed Ozempic gave me a headache.", "ozempic", set(),
             ["not_drug_dependent"]),                         # drug only in a relative clause
            # Uptake: did the person actually take the drug?
            ("Starting Ozempic soon and I'm so nervous about the nausea.", "ozempic", set(),
             ["no_uptake:future"]),
            ("I'm hoping to start Wegovy because my headache is constant.", "wegovy", set(),
             ["no_uptake:future"]),
            ("I never did Ozempic but I have nausea every day.", "ozempic", set(),
             ["no_uptake:negated"]),
            ("I stopped Wegovy last month and now I have headaches.", "wegovy",
             {"Headache"}),                                   # stopped still counts as taken
            ("Ozempic is approved for diabetes. I have a headache.", "ozempic", set(),
             ["no_uptake:none"]),
        ]
    ok = True
    for case, (doc, (kept, rej)) in zip(
            cases, run([(c[0], c[1]) for c in cases], nlp, finder, coref_ok, mode, scope)):
        text, drug, expected = case[:3]
        reasons = case[3] if len(case) > 3 else []
        got = {r["llt_name"] for r in kept}
        got_reasons = [r["reason"] for r in rej]
        good = got == expected and all(x in got_reasons for x in reasons)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {text}\n        kept={sorted(got)} "
              f"expected={sorted(expected)} "
              f"rejected={[(r['symptom_text'], r['reason']) for r in rej]} "
              f"links={[(r['symptom_text'], r.get('dep_relation'), r.get('experiencer')) for r in kept]}")

    print("-- MetaMapLite merge (synthetic hits)")
    hits = [[_fake_hit(t, *h)] for t, _, _, h, _ in mml_cases]
    for (text, drug, expected, _, reasons), (doc, (kept, rej)) in zip(
            mml_cases, run([(t, d) for t, d, *_ in mml_cases], nlp, finder, coref_ok, mode,
                           scope, mml_hits=hits)):
        got = sorted((r["llt_name"], r["match_type"]) for r in kept)
        got_reasons = [r["reason"] for r in rej]
        good = got == sorted(expected) and all(x in got_reasons for x in reasons)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {text}\n        kept={got} expected={expected} "
              f"rejected={[(r['symptom_text'], r['reason']) for r in rej]}")
    print("selftest", "passed" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="glp1_single_drug_5pct_subset.csv")
    ap.add_argument("--meddra-dir", default="meddra_29_1",
                    help="MedAscii folder (default: meddra_29_1, written by mdb_export.py)")
    ap.add_argument("--nrows", type=int)
    ap.add_argument("--output", default="symptoms_extracted.csv")
    ap.add_argument("--rejected-output", default="symptoms_rejected.csv")
    ap.add_argument("--link-scope", choices=["word", "post", "sentence"], default="word",
                    help="word (default): each symptom must depend on a drug mention in the parse "
                         "tree (or follow an active drug mention) and be a personal experience; "
                         "post: every sentence of a single-drug post is linked unless another "
                         "drug intervenes")
    ap.add_argument("--no-uptake", action="store_true",
                    help="word scope: do not require evidence that the person took the drug")
    ap.add_argument("--coref", choices=["both", "rules", "coreferee"], default="both")
    ap.add_argument("--model", default="en_core_web_lg")
    ap.add_argument("--min-llt-len", type=int, default=3)
    ap.add_argument("--debug", action="store_true", help="also write rejected candidates")
    ap.add_argument("--metamap", action="store_true",
                    help="also find symptoms with MetaMapLite (UMLS) and map them to MedDRA")
    ap.add_argument("--mml-dir", default=MML_DIR, help="MetaMapLite install folder")
    ap.add_argument("--rebuild-crosswalk", action="store_true",
                    help="rebuild MetaMap/cui_to_meddra.tsv (e.g. after a MedDRA update)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        nlp, coref_ok = build_nlp(args.model, use_coreferee=args.coref != "rules")
        sys.exit(0 if selftest(nlp, coref_ok, args.coref, args.link_scope) else 1)
    if not os.path.isdir(args.meddra_dir):
        ap.error(f"MedDRA folder {args.meddra_dir!r} not found. Export it from the MedDRA Desktop "
                 'Browser with: python mdb_export.py "MDB568English291/MedDRA 29.1 - English.dat" meddra_29_1')

    vocab = load_meddra(args.meddra_dir, args.min_llt_len)
    print(f"Loaded {len(vocab):,} current MedDRA LLTs", file=sys.stderr)
    df = pd.read_csv(args.input, nrows=args.nrows)
    df["body"] = df["body"].fillna("").astype(str)

    # MetaMapLite runs (in its own JVM) before spaCy loads, so the two don't share memory.
    crosswalk, mml_hits = None, None
    if args.metamap:
        from metamap_utils import load_crosswalk, run_metamaplite
        crosswalk = load_crosswalk(vocab, rebuild=args.rebuild_crosswalk)
        mml_hits = run_metamaplite(df["body"], mml_dir=args.mml_dir)

    nlp, coref_ok = build_nlp(args.model, use_coreferee=args.coref != "rules")
    finder = SymptomFinder(nlp, vocab, crosswalk=crosswalk)

    kept_rows, rej_rows = [], []
    for (idx, row), (doc, (kept, rej)) in zip(
            df.iterrows(),
            run(zip(df["body"], df["drug_name"]), nlp, finder, coref_ok, args.coref,
                args.link_scope, mml_hits, uptake=not args.no_uptake)):
        meta = {"row_id": idx, "id": row.get("id"), "drug_name": row["drug_name"],
                "body": row["body"]}
        kept_rows += [{**meta, **r} for r in kept]
        rej_rows += [{**meta, **r} for r in rej]

    cols = ["row_id", "id", "drug_name", "symptom_text", "match_type", "fuzzy_dist",
            "link_method",
            *(["drug_anchor", "dep_relation", "experiencer"] if args.link_scope == "word" else []),
            *(["uptake"] if args.link_scope == "word" and not args.no_uptake else []),
            *MEDDRA_COLS, *(UMLS_COLS if args.metamap else []), "sentence", "body"]
    pd.DataFrame(kept_rows, columns=cols).to_csv(args.output, index=False)
    if args.debug:
        pd.DataFrame(rej_rows, columns=cols + ["reason"]).to_csv(args.rejected_output, index=False)

    # Summary
    print(f"\nRows processed: {len(df)}")
    print(f"Posts with >=1 drug-linked symptom: {len({r['row_id'] for r in kept_rows})}")
    print(f"Symptoms kept: {len(kept_rows)}  |  candidates rejected: {len(rej_rows)}")
    print("Rejection reasons:", dict(Counter(r["reason"] for r in rej_rows)))
    print("Link methods (kept):", dict(Counter(r["link_method"] for r in kept_rows)))
    print("Match types (kept):", dict(Counter(r["match_type"] for r in kept_rows)))
    print("Top PTs:", Counter(r["pt_name"] for r in kept_rows).most_common(15))
    print(f"Wrote {args.output}" + (f" and {args.rejected_output}" if args.debug else ""))


if __name__ == "__main__":
    main()
