# -*- coding: utf-8 -*-
"""
truthprobe.bundle

Gli artefatti, con la loro provenienza dentro.

DUE REGOLE DEL CONTRATTO DI RIPRODUCIBILITA', QUI IMPOSTE DAL CODICE
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
    """Scrive un bundle con la sua provenienza.

    payload   i tensori e le liste dello strumento (axes, t_global, cos_peak,
              transfer, cats, e quel che serve)
    analysis  i parametri che cambiano il risultato ma sono propri di questa
              misura (blocco, componente, permutazioni). Finiscono nei
              metadati, non nel protocollo.
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
        print("            protocollo %s   truthprobe %s" % (protocol.label(), __version__))
    return path


def load(path, allow_legacy=False):
    """Legge un bundle e restituisce (payload, protocol, meta).

    Se il bundle non porta un protocollo, si ferma: e' di provenienza ignota.
    Con allow_legacy si accetta assumendo il protocollo storico dei dizionari,
    ma la cosa viene dichiarata a schermo, perche' e' un'assunzione e non un
    dato letto dal file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta = obj.get("meta", {}) or {}
    p = Protocol.from_dict(meta.get("protocol"))
    if p is None:
        if not allow_legacy:
            raise ValueError(
                "%s non porta un protocollo: provenienza ignota.\n"
                "  E' stato prodotto da uno strumento precedente alla libreria. "
                "Rigeneralo, oppure caricalo con allow_legacy=True se sai con "
                "quale convenzione e' stato costruito." % os.path.basename(path))
        from .protocol import LEGACY_DICT
        p = LEGACY_DICT
        print("  [assunzione] %s: protocollo non presente, assumo quello storico "
              "dei dizionari (suffisso vuoto, join raw). Non e' un dato letto "
              "dal file." % os.path.basename(path))
    return obj, p, meta


def require_comparable(bundles, names=None, what="confronto"):
    """Verifica che una lista di bundle sia confrontabile PRIMA di calcolare.

    bundles: lista di (payload, protocol, meta) come tornati da load().
    Si ferma se i protocolli divergono, e dice su quali campi.
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
                "%s fra artefatti non confrontabili:\n"
                "  %s  contro  %s\n%s\n"
                "  Non misurano lo stesso oggetto. Rigenerane uno con lo stesso "
                "protocollo." % (what, names[0], names[k], righe))
    # avviso non bloccante: stessa convenzione ma dimensioni diverse
    for k in range(1, len(bundles)):
        pk = bundles[k][1]
        for c in ("k_relations", "pairs_per_relation", "seed"):
            a, b = getattr(p0, c), getattr(pk, c)
            if a is not None and b is not None and a != b:
                print("  [avviso] %s differisce: %s contro %s. Il confronto resta "
                      "lecito sulle categorie condivise, ma la stima ha "
                      "numerosita' diversa." % (c, a, b))


def align_categories(bundles):
    """Riordina piu' bundle sulle categorie condivise, restituendo gli indici.

    Serve perche' l'ordine delle categorie puo' differire fra strumenti: un
    confronto cella per cella senza riordino da' scarti enormi che sembrano
    differenze di contenuto e non lo sono. E' successo in questa serie: uno
    scarto di 0.93 che dopo il riordino era 1.25e-06.
    """
    liste = [[str(c) for c in b[0]["cats"]] for b in bundles]
    comuni = set(liste[0])
    for l in liste[1:]:
        comuni &= set(l)
    comuni = [c for c in liste[0] if c in comuni]      # ordine del primo
    idx = [[l.index(c) for c in comuni] for l in liste]
    return comuni, idx


def fingerprint(payload, key="cos_peak", tol=5e-4):
    """L'impronta di un bundle: la sua matrice dei coseni identifica il
    contenuto a prescindere dal nome del file. In questa serie e' servita a
    risolvere uno scambio di identita' fra bundle: i nomi possono mentire, gli
    assi no."""
    M = payload.get(key)
    if M is None:
        return None
    M = torch.as_tensor(M).float()
    return dict(shape=tuple(M.shape), sum=float(M.sum()),
                offdiag_median=float(M[~torch.eye(M.shape[0], dtype=torch.bool)].median()),
                tol=tol)


def same_content(pa, pb, key="cos_peak", tol=5e-4):
    """Due bundle hanno lo stesso contenuto? Confronta le matrici cella per
    cella dopo aver allineato le categorie."""
    ca = [str(c) for c in pa["cats"]]
    cb = [str(c) for c in pb["cats"]]
    if set(ca) != set(cb):
        return False, float("inf")
    perm = [ca.index(c) for c in cb]
    A = torch.as_tensor(pa[key]).float()[perm][:, perm]
    B = torch.as_tensor(pb[key]).float()
    d = float((A - B).abs().max())
    return d < tol, d
