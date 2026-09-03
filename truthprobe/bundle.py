# -*- coding: utf-8 -*-
"""
truthprobe.bundle

The artifacts, with their provenance within.

TWO RULES OF THE REPRODUCIBILITY CONTRACT, IMPOSED HERE:
  (i)   un file esportato porta la sua identita' completa dentro il file e nel
        nome: modello, blocco, protocollo, versione della libreria.
  (iii) un artefatto la cui provenienza non si stabilisce dal contenuto viene
        messo in quarantena, non riparato.

Il secondo punto e' il motivo per cui load() rifiuta i file senza protocollo
invece di indovinarne uno: un bundle prodotto da uno strumento vecchio non e'
sbagliato, e' semplicemente di provenienza ignota, e va dichiarato come tale
con adopt_legacy() invece che assunto.

require_comparable() e' la funzione che rende impossibile l'errore trovato in
questa serie: due bundle con suffissi diversi non si confrontano piu' in
silenzio, si fermano con un messaggio che dice quali campi divergono.
"""

import hashlib
import os
from datetime import datetime, timezone

import torch

from . import __version__
from .protocol import Protocol


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def save(path, protocol, model, payload, analysis=None, verbose=True):
    """Write the bundle with their provenance within.

    payload   tensors and the tool's lists (axes, t_global, cos_peak,
              transfer, cats, and whatever is needed)
    analysis : parameters that alter the results but are specific to this 
               measurement run (block, component, permutations). These 
               end up in the metadata, not in the protocol.
    """
    meta = dict(
        truthprobe_version=__version__,
        model=model,
        protocol=protocol.to_dict(),
        analysis=dict(analysis or {}),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    obj = dict(payload)
    obj["meta"] = meta
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(obj, path)
    if verbose:
        print("  [salvato] %s" % path)
        print("            protocol %s   truthprobe %s" % (protocol.label(), __version__))
    return path


def load(path, allow_legacy=False):
    """Reads a bundle and returns (payload, protocol, meta).

    If the bundle does not carry a protocol, it halts: the provenance is unknown.
    By setting allow_legacy=True, it is accepted by assuming the historical 
    dictionary format, but this is explicitly printed on screen because it 
    is a guess and not factual data read from the file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta = obj.get("meta", {}) or {}
    p = Protocol.from_dict(meta.get("protocol"))
    if p is None:
        if not allow_legacy:
            raise ValueError(
                "%s does not carry a protocol: unknown provenance.\n"
                " It was produced by a tool predating this library version. "
                "Regenerate it, or load it with allow_legacy=True if you know "
                "which convention was used to build it." % os.path.basename(path))
        from .protocol import LEGACY_DICT
        p = LEGACY_DICT
        print(" [assumption] %s: protocol missing, assuming the historical "
              "dictionary format (empty suffix, join raw). It is not data read "
              "from the file." % os.path.basename(path))
    return obj, p, meta


def require_comparable(bundles, names=None, what="comparison"):
    """Verifies that a list of bundles is comparable BEFORE computing.

    bundles: list of (payload, protocol, meta) as returned by load().
    It halts execution if protocols diverge, detailing which fields mismatch.
    """
    if len(bundles) < 2:
        return
    names = names or ["bundle %d" % i for i in range(len(bundles))]
    p0 = bundles[0][1]
    for k in range(1, len(bundles)):
        pk = bundles[k][1]
        if not p0.compatible_with(pk):
            d = p0.diff(pk)
            righe = "\n".join("    %-12s %-24r %r" % (c, v[0], v[1]) for c, v in d.items())
            raise ValueError(
                "%s between incomparable artifacts:\n"
                "  %s  versus  %s\n%s\n"
                "  They do not measure the same object. Rebuild one using the same "
                "protocol." % (what, names[0], names[k], righe))
    # avviso non bloccante: stessa convenzione ma dimensioni diverse
    for k in range(1, len(bundles)):
        pk = bundles[k][1]
        for c in ("k_relations", "pairs_per_relation", "seed"):
            a, b = getattr(p0, c), getattr(pk, c)
            if a is not None and b is not None and a != b:
                print(" [warning] %s differs: %s versus %s. The comparison remains "
              "valid on shared categories, but the estimation has "
              "different sample sizes." % (c, a, b))


def align_categories(bundles):
    """Reorders multiple bundles based on shared categories, returning the indices.

    Necessary because category ordering can differ between tools: a cell-by-cell 
    comparison without reordering yields massive discrepancies that appear to be 
    content differences, but are not. It happened in this series: a mismatch 
    of 0.93 plummeted to 1.25e-06 after reordering.
    """
    liste = [[str(c) for c in b[0]["cats"]] for b in bundles]
    comuni = set(liste[0])
    for l in liste[1:]:
        comuni &= set(l)
    comuni = [c for c in liste[0] if c in comuni]      # ordine del primo
    idx = [[l.index(c) for c in comuni] for l in liste]
    return comuni, idx


def fingerprint(payload, key="cos_peak", tol=5e-4):
    """The fingerprint of a bundle: its cosine matrix uniquely identifies the 
    content regardless of the filename. In this series, it resolved an 
    identity swap between bundles: names can lie, axes do not."""
    M = payload.get(key)
    if M is None:
        return None
    M = torch.as_tensor(M).float()
    return dict(shape=tuple(M.shape), sum=float(M.sum()),
                offdiag_median=float(M[~torch.eye(M.shape[0], dtype=torch.bool)].median()),
                tol=tol)


def same_content(pa, pb, key="cos_peak", tol=5e-4):
    """Do two bundles have the same content? Compares the matrices cell-by-cell 
    after aligning categories."""
    ca = [str(c) for c in pa["cats"]]
    cb = [str(c) for c in pb["cats"]]
    if set(ca) != set(cb):
        return False, float("inf")
    perm = [ca.index(c) for c in cb]
    A = torch.as_tensor(pa[key]).float()[perm][:, perm]
    B = torch.as_tensor(pb[key]).float()
    d = float((A - B).abs().max())
    return d < tol, d
