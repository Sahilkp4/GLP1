"""Shared MedDRA MedAscii loaders (used by stage 1)."""
import os
import sys


def _find_file(root, name):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() == name:
                return os.path.join(dirpath, f)
    return None


def _read_asc(path):
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                return [line.rstrip("\r\n").split("$") for line in fh if line.strip()]
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


HIER_FIELDS = ["hlt_code", "hlt_name", "hlgt_code", "hlgt_name", "soc_code", "soc_name", "soc_abbrev"]


def _load_pts(meddra_dir):
    """Return {pt_code: {pt_name, hlt_code, hlt_name, hlgt_code, hlgt_name, soc_code, soc_name,
    soc_abbrev}}, following each PT's primary SOC path in mdhier.asc."""
    pt_path = _find_file(meddra_dir, "pt.asc")
    if not pt_path:
        sys.exit(f"pt.asc not found under {meddra_dir}")
    hier_path = _find_file(meddra_dir, "mdhier.asc")
    soc_path = _find_file(meddra_dir, "soc.asc")

    socs = {}
    if soc_path:
        for r in _read_asc(soc_path):  # soc_code$soc_name$soc_abbrev$...
            socs[r[0]] = {"soc_name": r[1], "soc_abbrev": r[2] if len(r) > 2 else ""}
    pts = {}
    for r in _read_asc(pt_path):  # pt_code$pt_name$null$pt_soc_code$...
        soc_code = r[3] if len(r) > 3 else ""
        soc = socs.get(soc_code, {})
        pts[r[0]] = {"pt_name": r[1], **{f: "" for f in HIER_FIELDS},
                     "soc_code": soc_code, "soc_name": soc.get("soc_name", soc_code),
                     "soc_abbrev": soc.get("soc_abbrev", "")}
    if hier_path:  # full primary path (PT -> HLT -> HLGT -> SOC) from the hierarchy file
        for r in _read_asc(hier_path):
            # pt_code$hlt_code$hlgt_code$soc_code$pt_name$hlt_name$hlgt_name$soc_name$soc_abbrev$null$pt_soc_code$primary_soc_fg$
            if len(r) > 11 and r[11] == "Y" and r[0] in pts:
                pts[r[0]].update({"hlt_code": r[1], "hlt_name": r[5], "hlgt_code": r[2],
                                  "hlgt_name": r[6], "soc_code": r[3], "soc_name": r[7],
                                  "soc_abbrev": r[8]})
    return pts


def _llt_rows(meddra_dir):
    llt_path = _find_file(meddra_dir, "llt.asc")
    if not llt_path:
        sys.exit(f"llt.asc not found under {meddra_dir}")
    # llt_code$llt_name$pt_code$whoart$harts$costart$icd9$icd9cm$icd10$llt_currency$jart$
    for r in _read_asc(llt_path):
        yield r[0], r[1], r[2], (r[9] if len(r) > 9 else "Y")


def load_meddra(meddra_dir, min_len=3):
    """Return {lower(llt_name): {llt_code, llt_name, pt_code, pt_name, hlt_*, hlgt_*, soc_*}}
    for current LLTs, with the PT's primary-SOC hierarchy."""
    pts = _load_pts(meddra_dir)
    vocab = {}
    for code, name, pt_code, currency in _llt_rows(meddra_dir):
        if currency != "Y" or len(name) < min_len:
            continue
        pt = pts.get(pt_code, {})
        vocab.setdefault(name.lower(), {
            "llt_code": code, "llt_name": name, "pt_code": pt_code,
            "pt_name": pt.get("pt_name", ""), **{f: pt.get(f, "") for f in HIER_FIELDS},
        })
    return vocab

