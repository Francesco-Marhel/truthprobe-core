# -*- coding: utf-8 -*-
"""
truthprobe.data

Il caricamento delle coppie. Ogni frase passa da Protocol.sentence, che e'
l'unico posto in tutta la libreria dove prompt e target vengono concatenati.

E' il modulo che chiude il buco trovato in questa serie. Prima ogni strumento
costruiva le frasi per conto suo, e due di loro sono divergiuti per mesi senza
che nessuno se ne accorgesse. Qui la costruzione non e' accessibile: si chiede
un PairSet a partire da un Protocol, e il PairSet porta il Protocol con se'.

DUE OGGETTI

  Pair      una coppia minimale: prompt, target vero, target falso, e la
            categoria se il materiale ne ha una.
  PairSet   un insieme di coppie con il suo Protocol, gli indici delle frasi
            e la mappa categoria -> coppie. E' quello che gli strumenti
            ricevono, e da cui non si puo' ricostruire una frase diversamente.

DUE MODI DI COSTRUIRE IL FALSO, DICHIARATI
  esplicito  il dataset fornisce target_false (CounterFact, o un file con
             false_targets). E' il caso normale.
  scambio    il falso viene dal target di un altro item della stessa
             relazione. Serve ai file di fatti che non portano il falso, ed e'
             deterministico: si scorrono gli item della relazione in ordine di
             comparsa, a partire dal successivo e ciclicamente, e si prende il
             primo target diverso. Un item la cui relazione non offre
             alternative viene saltato e il motivo viene registrato.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .protocol import Protocol, CANONICAL


@dataclass
class Pair:
    prompt: str
    target_true: str
    target_false: str
    category: Optional[str] = None
    ident: Optional[str] = None
    origin: str = "esplicito"          # "esplicito" oppure "scambio"


@dataclass
class PairSet:
    """Coppie pronte, con il protocollo che le ha costruite.

    items    le frasi, in ordine: per ogni coppia prima il vero poi il falso
    pidx     gli indici (vero, falso) dentro items
    pairs    gli oggetti Pair, allineati a pidx
    protocol il Protocol usato, che viaggia con i dati e finisce nei bundle
    skipped  gli item scartati, con il motivo, per non perderli in silenzio
    """
    items: List[str]
    pidx: List[Tuple[int, int]]
    pairs: List[Pair]
    protocol: Protocol
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def __len__(self):
        return len(self.pidx)

    @property
    def categories(self):
        vis = []
        for p in self.pairs:
            if p.category and p.category not in vis:
                vis.append(p.category)
        return vis

    def by_category(self):
        """categoria -> lista di (indice_vero, indice_falso)"""
        out = {}
        for k, p in enumerate(self.pairs):
            if p.category:
                out.setdefault(p.category, []).append(self.pidx[k])
        return out

    def index_of_category(self):
        """categoria -> posizioni nella lista delle coppie"""
        out = {}
        for k, p in enumerate(self.pairs):
            if p.category:
                out.setdefault(p.category, []).append(k)
        return out

    def subset(self, keep):
        """Nuovo PairSet con le sole coppie indicate. Gli indici vengono
        rinumerati, quindi resta coerente."""
        items, pidx, pairs = [], [], []
        for k in keep:
            p = self.pairs[k]
            it = len(items); items.append(self.protocol.sentence(p.prompt, p.target_true))
            iff = len(items); items.append(self.protocol.sentence(p.prompt, p.target_false))
            pidx.append((it, iff)); pairs.append(p)
        return PairSet(items, pidx, pairs, self.protocol, list(self.skipped))

    def describe(self):
        r = ["%d coppie = %d frasi" % (len(self.pidx), len(self.items))]
        c = self.categories
        if c:
            r.append("%d categorie" % len(c))
        sw = sum(1 for p in self.pairs if p.origin == "scambio")
        if sw:
            r.append("%d falsi per scambio dentro relazione" % sw)
        if self.skipped:
            r.append("%d saltati" % len(self.skipped))
        r.append("suffisso %r, join %r" % (self.protocol.suffix, self.protocol.join))
        return ", ".join(r)


# =====================================================================
#  costruzione: l'unico punto in cui si formano le frasi
# =====================================================================
def build(pairs, protocol=CANONICAL, skipped=None):
    items, pidx = [], []
    for p in pairs:
        it = len(items); items.append(protocol.sentence(p.prompt, p.target_true))
        iff = len(items); items.append(protocol.sentence(p.prompt, p.target_false))
        pidx.append((it, iff))
    return PairSet(items, pidx, list(pairs), protocol, list(skipped or []))


# =====================================================================
#  sorgenti
# =====================================================================
def _open_dataset(protocol, local_file=None):
    if local_file:
        from datasets import load_dataset
        ext = os.path.splitext(local_file)[1].lower().lstrip(".")
        fmt = {"parquet": "parquet", "json": "json", "jsonl": "json",
               "csv": "csv"}.get(ext)
        if fmt is None:
            raise ValueError("estensione locale non supportata: %s" % ext)
        return load_dataset(fmt, data_files=local_file, split="train")
    from datasets import load_dataset
    return load_dataset(protocol.dataset, split="train", revision=protocol.revision)


def _raw_rows(protocol, local_file=None, verbose=True):
    ds = _open_dataset(protocol, local_file)
    if verbose:
        print("  [provenienza] %s @ %s  (%d righe)"
              % (protocol.dataset, (protocol.revision or "latest")[:12], len(ds)))
    out = []
    for ex in ds:
        rid = str(ex.get("relation_id", "")).strip()
        prompt = str(ex.get("prompt", "")).strip()
        tt = str(ex.get("target_true", ""))
        tf = str(ex.get("target_false", ""))
        if not prompt or not tt.strip() or not tf.strip():
            continue
        if tt.strip().lower() == tf.strip().lower():
            continue
        out.append((rid, prompt, tt, tf))
    return out


def counterfact_flat(protocol=CANONICAL, max_pairs=250, local_file=None, verbose=True):
    """Coppie pescate a caso su tutto CounterFact, senza raggruppare per
    relazione. E' il campionamento del protocollo canonico di truth_probe."""
    rows = _raw_rows(protocol, local_file, verbose)
    idx = list(range(len(rows)))
    random.Random(protocol.seed).shuffle(idx)
    pairs, seen = [], set()
    for i in idx:
        rid, prompt, tt, tf = rows[i]
        key = (prompt, tt.strip(), tf.strip())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(Pair(prompt, tt, tf, category=rid or None))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError("nessuna coppia costruita: lo schema del dataset e' cambiato?")
    return build(pairs, protocol)


def counterfact_by_relation(protocol=CANONICAL, k=33, n_per=60,
                            whitelist=None, local_file=None, verbose=True):
    """Coppie raggruppate per relazione Wikidata: le k relazioni con piu'
    coppie uniche, n_per ciascuna. E' il campionamento della famiglia
    dizionari. La chiave di unicita' e' (prompt, target vero), come nel
    codice canonico."""
    rows = _raw_rows(protocol, local_file, verbose)
    by_rel = {}
    for rid, prompt, tt, tf in rows:
        if not rid:
            continue
        by_rel.setdefault(rid, {})[(prompt, tt.strip())] = (prompt, tt, tf)

    if whitelist:
        top = [r for r in whitelist if r in by_rel]
        mancanti = [r for r in whitelist if r not in by_rel]
        if mancanti and verbose:
            print("  [avviso] relazioni richieste e assenti: %s" % mancanti)
    else:
        top = sorted(by_rel, key=lambda r: len(by_rel[r]), reverse=True)[:k]

    pairs = []
    for rid in top:
        rws = list(by_rel[rid].values())
        random.Random(protocol.seed).shuffle(rws)
        rws = rws[:n_per]
        if verbose:
            print("    %-7s %3d coppie" % (rid, len(rws)))
        for prompt, tt, tf in rws:
            pairs.append(Pair(prompt, tt, tf, category=rid))
    p = protocol.with_(k_relations=len(top), pairs_per_relation=n_per)
    return build(pairs, p)


def from_json(path, protocol=CANONICAL, verbose=True):
    """Un file di fatti nello schema del progetto: lista di oggetti con
    prompts, targets, e facoltativi false_targets, relation_id, id.

    Se false_targets manca, il falso viene per scambio dentro la relazione,
    con la regola dichiarata in cima al modulo."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ValueError("il file non contiene una lista di item: %s" % path)

    rows = [el for el in data if isinstance(el, dict)]
    scartati = [("(elemento non valido)", "non e' un oggetto")] * (len(data) - len(rows))

    def first(it, key):
        v = it.get(key)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    return x
            return ""
        return v if isinstance(v, str) else ""

    by_rel = {}
    for pos, it in enumerate(rows):
        rid = it.get("relation_id")
        if rid:
            by_rel.setdefault(rid, []).append(pos)

    pairs = []
    for pos, it in enumerate(rows):
        ident = str(it.get("id", "?"))
        prompt, tt = first(it, "prompts"), first(it, "targets")
        if not prompt or not tt:
            scartati.append((ident, "prompt o target vero mancante"))
            continue
        tf = first(it, "false_targets")
        origin = "esplicito"
        if not tf:
            rid = it.get("relation_id")
            grp = by_rel.get(rid, []) if rid else []
            if len(grp) < 2:
                scartati.append((ident, "nessun false_target e nessuna alternativa"))
                continue
            k = grp.index(pos)
            for step in range(1, len(grp)):
                cand = first(rows[grp[(k + step) % len(grp)]], "targets")
                if cand and cand.strip().lower() != tt.strip().lower():
                    tf, origin = cand, "scambio"
                    break
            if not tf:
                scartati.append((ident, "nessun target diverso nella relazione"))
                continue
        pairs.append(Pair(prompt, tt, tf, category=it.get("relation_id"),
                          ident=ident, origin=origin))

    ps = build(pairs, protocol, scartati)
    if verbose:
        print("  [file] %s: %s" % (os.path.basename(path), ps.describe()))
    return ps
