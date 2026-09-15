"""External data sources: UniProt, RCSB PDB, AlphaFold DB, PubChem.

All calls are synchronous httpx requests with explicit timeouts. Failures raise
``SourceError`` with a message that can be shown to the user as-is.
"""

from __future__ import annotations

import copy
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .config import imports_dir

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
HEADERS = {"User-Agent": "Oritatami/0.2 (local protein structure playground)"}


class SourceError(RuntimeError):
    kind = "network"


def _ttl_cache(seconds: float, maxsize: int = 256):
    """Memoize successful lookups for a while; errors are not cached so a retry can succeed."""

    def deco(fn):
        store: dict[tuple, tuple[float, Any]] = {}
        lock = threading.Lock()

        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit and now - hit[0] < seconds:
                    return copy.deepcopy(hit[1])
            value = fn(*args, **kwargs)
            with lock:
                if len(store) >= maxsize:
                    store.pop(next(iter(store)))
                store[key] = (now, value)
            return copy.deepcopy(value)

        wrapper.__wrapped__ = fn
        wrapper.cache_clear = store.clear
        return wrapper

    return deco


def _get(url: str, *, params: dict[str, Any] | None = None, accept_404: bool = False) -> httpx.Response | None:
    try:
        with httpx.Client(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
            resp = client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise SourceError(f"通信に失敗しました: {url} ({exc.__class__.__name__}: {exc})") from exc
    if resp.status_code == 404 and accept_404:
        return None
    if resp.status_code >= 400:
        raise SourceError(f"{url} が HTTP {resp.status_code} を返しました")
    return resp


# ---------------------------------------------------------------- UniProt
ACCESSION_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")
UNIPROT_FEATURES = {
    "Active site", "Binding site", "Site", "Domain", "Region", "Motif", "Disulfide bond",
    "Modified residue", "Glycosylation", "Metal binding", "Transmembrane", "Signal", "Propeptide",
    "Mutagenesis", "Zinc finger", "DNA binding",
}


ORGANISM_WORDS = {
    "human": 9606, "ヒト": 9606, "mouse": 10090, "マウス": 10090, "rat": 10116, "yeast": 559292,
    "酵母": 559292, "ecoli": 83333, "e.coli": 83333, "大腸菌": 83333, "zebrafish": 7955, "fly": 7227,
    "drosophila": 7227, "arabidopsis": 3702, "chicken": 9031, "bovine": 9913, "cow": 9913,
}


def _uniprot_query(query: str) -> str:
    if ACCESSION_RE.match(query.upper()):
        return f"accession:{query.upper()}"
    if ":" in query or " AND " in query or " OR " in query:
        return query  # already UniProt query syntax
    words = query.split()
    organisms = [ORGANISM_WORDS[w.lower()] for w in words if w.lower() in ORGANISM_WORDS]
    terms = [w for w in words if w.lower() not in ORGANISM_WORDS]
    q = " ".join(terms) or query
    if organisms:
        q = f"({q}) AND organism_id:{organisms[0]}"
    return q


@_ttl_cache(600)
def uniprot_search(query: str, size: int = 15) -> list[dict[str, Any]]:
    """Reviewed (Swiss-Prot) entries first, then unreviewed ones, each in UniProt's relevance order."""
    query = query.strip()
    if not query:
        return []
    q = _uniprot_query(query)
    fields = "accession,id,protein_name,organism_name,length,gene_names,reviewed"
    results: list[dict[str, Any]] = []
    for extra in (" AND reviewed:true", " AND reviewed:false"):
        if len(results) >= size:
            break
        resp = _get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": f"({q}){extra}", "fields": fields, "format": "json", "size": size - len(results)},
        )
        assert resp is not None
        results.extend(resp.json().get("results", []))
    out = []
    for r in results:
        desc = r.get("proteinDescription", {})
        name = (desc.get("recommendedName") or {}).get("fullName", {}).get("value")
        if not name and desc.get("submissionNames"):
            name = desc["submissionNames"][0].get("fullName", {}).get("value")
        genes = [g.get("geneName", {}).get("value") for g in r.get("genes", []) if g.get("geneName")]
        out.append({
            "accession": r.get("primaryAccession"),
            "entry_name": r.get("uniProtkbId"),
            "name": name or r.get("uniProtkbId"),
            "organism": r.get("organism", {}).get("scientificName"),
            "length": r.get("sequence", {}).get("length"),
            "genes": [g for g in genes if g],
            "reviewed": str(r.get("entryType", "")).startswith("UniProtKB reviewed"),
        })
    return out


@_ttl_cache(3600)
def uniprot_entry(accession: str) -> dict[str, Any]:
    acc = accession.strip().upper()
    if not ACCESSION_RE.match(acc):
        raise SourceError(f"UniProt アクセッションの書式ではありません: {accession}")
    resp = _get(f"https://rest.uniprot.org/uniprotkb/{acc}.json", accept_404=True)
    if resp is None:
        raise SourceError(f"UniProt に {acc} が見つかりません")
    data = resp.json()
    desc = data.get("proteinDescription", {})
    name = (desc.get("recommendedName") or {}).get("fullName", {}).get("value")
    if not name and desc.get("submissionNames"):
        name = desc["submissionNames"][0].get("fullName", {}).get("value")
    function = []
    for c in data.get("comments", []):
        if c.get("commentType") in ("FUNCTION", "SUBUNIT", "CATALYTIC ACTIVITY", "COFACTOR"):
            for t in c.get("texts", []):
                if t.get("value"):
                    function.append(f"[{c['commentType']}] {t['value']}")
            if c.get("commentType") == "CATALYTIC ACTIVITY" and c.get("reaction", {}).get("name"):
                function.append(f"[REACTION] {c['reaction']['name']}")
            if c.get("commentType") == "COFACTOR":
                for cf in c.get("cofactors", []):
                    function.append(f"[COFACTOR] {cf.get('name')}")
    features = []
    for f in data.get("features", []):
        if f.get("type") not in UNIPROT_FEATURES:
            continue
        loc = f.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None:
            continue
        item = {
            "type": f.get("type"),
            "start": start,
            "end": end if end is not None else start,
            "description": f.get("description", ""),
        }
        if f.get("ligand", {}).get("name"):
            item["ligand"] = f["ligand"]["name"]
        if f.get("alternativeSequence", {}).get("alternativeSequences"):
            item["alternative"] = f["alternativeSequence"]["alternativeSequences"]
            item["original"] = f["alternativeSequence"].get("originalSequence")
        features.append(item)
    genes = [g.get("geneName", {}).get("value") for g in data.get("genes", []) if g.get("geneName")]
    return {
        "accession": data.get("primaryAccession", acc),
        "entry_name": data.get("uniProtkbId"),
        "name": name or data.get("uniProtkbId") or acc,
        "organism": data.get("organism", {}).get("scientificName"),
        "genes": [g for g in genes if g],
        "sequence": data.get("sequence", {}).get("value", ""),
        "function": function[:12],
        "features": features[:200],
    }


# ---------------------------------------------------------------- RCSB PDB
PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def _save_import(name: str, content: bytes) -> Path:
    path = imports_dir() / name
    path.write_bytes(content)
    return path


# An icosahedral virus is deposited as one wedge of the shell plus the symmetry operators
# that build the other 59. Downloading the plain entry therefore gives a fragment, not a
# particle; the assembly file is the expanded one. Measured: MS2 phage 0.36 MB → 21.6 MB,
# cowpea chlorotic mottle virus 0.45 MB → 27.1 MB (gemmi parses either in 0.1 s). Some
# entries expand much further — 6CGV reaches 693 MB — so the download is capped.
MAX_STRUCTURE_BYTES = 120_000_000
LARGE_STRUCTURE_BYTES = 40_000_000


def _download_structure(url: str, *, what: str) -> bytes | None:
    """Stream a structure file, refusing one too large to open."""
    try:
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0), headers=HEADERS,
                          follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code == 404:
                    return None
                if resp.status_code >= 400:
                    raise SourceError(f"{url} が HTTP {resp.status_code} を返しました")
                declared = int(resp.headers.get("content-length") or 0)
                if declared > MAX_STRUCTURE_BYTES:
                    raise SourceError(
                        f"{what} は {declared / 1e6:.0f} MB あり、開ける上限 "
                        f"({MAX_STRUCTURE_BYTES / 1e6:.0f} MB) を超えています")
                chunks, total = [], 0
                for chunk in resp.iter_bytes(1 << 20):
                    total += len(chunk)
                    if total > MAX_STRUCTURE_BYTES:
                        raise SourceError(
                            f"{what} が開ける上限 ({MAX_STRUCTURE_BYTES / 1e6:.0f} MB) を超えました")
                    chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise SourceError(f"通信に失敗しました: {url} ({exc.__class__.__name__}: {exc})") from exc
    return b"".join(chunks)


def fetch_pdb(pdb_id: str, assembly: bool = False) -> dict[str, Any]:
    pid = pdb_id.strip().upper()
    if not PDB_ID_RE.match(pid):
        raise SourceError(f"PDB ID の書式ではありません: {pdb_id}")
    note = None
    content = None
    if assembly:
        content = _download_structure(f"https://files.rcsb.org/download/{pid}-assembly1.cif",
                                      what=f"{pid} の生物学的単位")
        if content is None:
            note = f"{pid} には生物学的単位のファイルがないため、非対称単位を読み込みました"
    if content is None:
        content = _download_structure(f"https://files.rcsb.org/download/{pid}.cif",
                                      what=f"{pid} の構造")
    if content is None:
        raise SourceError(f"PDB に {pid} が見つかりません")
    suffix = "_assembly1" if assembly and note is None else ""
    path = _save_import(f"pdb_{pid}{suffix}.cif", content)
    title = pid
    info = _get(f"https://data.rcsb.org/rest/v1/core/entry/{pid}", accept_404=True)
    if info is not None:
        title = info.json().get("struct", {}).get("title") or pid
    if assembly and note is None:
        title = f"{title} (生物学的単位)"
    return {"path": path, "title": title, "note": note, "bytes": len(content),
            "source": {"db": "PDB", "id": pid, "assembly": assembly and note is None}}


def pdb_search(text: str, rows: int = 12) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    payload = {
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": text}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    try:
        with httpx.Client(timeout=TIMEOUT, headers=HEADERS) as client:
            resp = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=payload)
    except httpx.HTTPError as exc:
        raise SourceError(f"PDB 検索に失敗しました ({exc})") from exc
    if resp.status_code == 204:
        return []
    if resp.status_code >= 400:
        raise SourceError(f"PDB 検索が HTTP {resp.status_code} を返しました")
    ids = [r["identifier"] for r in resp.json().get("result_set", [])]
    titles = _pdb_titles(ids)
    return [{"id": pid, "title": titles.get(pid, "")} for pid in ids]


def _pdb_titles(ids: list[str]) -> dict[str, str]:
    """Titles for many entries in one GraphQL request (instead of one REST call per entry)."""
    if not ids:
        return {}
    query = "query($ids: [String!]!) { entries(entry_ids: $ids) { rcsb_id struct { title } } }"
    try:
        with httpx.Client(timeout=TIMEOUT, headers=HEADERS) as client:
            resp = client.post("https://data.rcsb.org/graphql", json={"query": query, "variables": {"ids": ids}})
        resp.raise_for_status()
        entries = (resp.json().get("data") or {}).get("entries") or []
        return {e["rcsb_id"]: ((e.get("struct") or {}).get("title") or "") for e in entries if e}
    except (httpx.HTTPError, ValueError, KeyError):
        # Titles are cosmetic; fall back to per-entry requests so search still works.
        out = {}
        for pid in ids:
            try:
                info = _get(f"https://data.rcsb.org/rest/v1/core/entry/{pid}", accept_404=True)
                out[pid] = info.json().get("struct", {}).get("title", "") if info is not None else ""
            except SourceError:
                out[pid] = ""
        return out


@_ttl_cache(3600)
def ccd_info(code: str) -> dict[str, Any] | None:
    code = code.strip().upper()
    resp = _get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", accept_404=True)
    if resp is None:
        return None
    d = resp.json()
    desc = d.get("rcsb_chem_comp_descriptor", {}) or {}
    return {
        "ccd": code,
        "name": d.get("chem_comp", {}).get("name"),
        "formula": d.get("chem_comp", {}).get("formula"),
        "smiles": desc.get("SMILES_stereo") or desc.get("SMILES"),
    }


# ---------------------------------------------------------------- AlphaFold DB
def fetch_afdb(accession: str) -> dict[str, Any]:
    acc = accession.strip().upper()
    if not ACCESSION_RE.match(acc):
        raise SourceError(f"UniProt アクセッションの書式ではありません: {accession}")
    resp = _get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", accept_404=True)
    if resp is None or not resp.json():
        raise SourceError(f"AlphaFold DB に {acc} の予測がありません")
    entries = resp.json()
    # Prefer the canonical isoform entry
    entry = next((e for e in entries if e.get("uniprotAccession") == acc), entries[0])
    cif_url = entry.get("cifUrl") or entry.get("pdbUrl")
    if not cif_url:
        raise SourceError(f"AlphaFold DB の応答に構造ファイルの URL がありません ({acc})")
    file_resp = _get(cif_url)
    assert file_resp is not None
    ext = ".cif" if cif_url.endswith(".cif") else ".pdb"
    path = _save_import(f"afdb_{acc}{ext}", file_resp.content)
    return {
        "path": path,
        "title": entry.get("uniprotDescription") or acc,
        "source": {"db": "AFDB", "id": acc, "version": entry.get("latestVersion")},
        "plddt_in_bfactor": True,
    }


# ---------------------------------------------------------------- PubChem
_PUBCHEM_PROPS = "SMILES,ConnectivitySMILES,MolecularWeight,MolecularFormula,IUPACName,Title"


@_ttl_cache(3600)
def pubchem_lookup(name: str) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise SourceError("化合物名が空です")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name, safe='')}/property/{_PUBCHEM_PROPS}/JSON"
    resp = _get(url, accept_404=True)
    if resp is None:
        raise SourceError(f"PubChem に「{name}」が見つかりません")
    props = resp.json().get("PropertyTable", {}).get("Properties", [])
    if not props:
        raise SourceError(f"PubChem に「{name}」が見つかりません")
    p = props[0]
    smiles = p.get("SMILES") or p.get("IsomericSMILES") or p.get("ConnectivitySMILES") or p.get("CanonicalSMILES")
    if not smiles:
        raise SourceError(f"PubChem の応答に SMILES がありません ({name})")
    return {
        "name": p.get("Title") or name,
        "cid": p.get("CID"),
        "smiles": smiles,
        "formula": p.get("MolecularFormula"),
        "molecular_weight": p.get("MolecularWeight"),
        "iupac": p.get("IUPACName"),
    }


def import_uploaded(filename: str, content: bytes) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[-80:] or "upload.cif"
    if not safe.lower().endswith((".cif", ".mmcif", ".pdb", ".ent")):
        raise SourceError("対応している構造ファイルは .cif / .mmcif / .pdb です")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _save_import(f"upload_{stamp}_{safe}", content)
