# -*- coding: utf-8 -*-
"""
truthprobe.data

Pair loading mechanism. Every sentence passes through Protocol.sentence, which is 
the single, centralized location in the entire library where prompts and targets 
are concatenated.

This module closes the loophole discovered in this series. Previously, each 
analytical tool constructed sentences independently, causing two of them to 
diverge for months without anyone noticing. Here, direct text construction is 
not exposed: an external tool must request a PairSet using a Protocol, 
and the PairSet carries that Protocol along with it.


TWO OBJECTS

  Pair       A minimal pair: prompt, true target, false target, and the category 
             if the underlying material provides one.
  PairSet    A collection of pairs packaged with their Protocol, the sentence indices, 
             and the category -> pairs mapping. This is what down-stream tools 
             receive, and it guarantees that a sentence cannot be reconstructed 
             in an inconsistent format.


TWO MODES FOR INFERRING FALSE TARGETS, EXPLICITLY DECLARED

  explicit   The dataset natively provides a 'target_false' (e.g., CounterFact, 
             or a custom file containing false_targets). This is the default case.
  swap       The false target is pulled from the true target of another item belonging 
             to the exact same relation. This is necessary for fact files that lack 
             explicit false targets. It is strictly deterministic: the items within 
             the relation are scanned in their order of appearance, starting from 
             the subsequent item and wrapping around cyclically, selecting the first 
             distinct target encountered. An item whose relation offers no valid 
             alternative target is skipped, and the reason is logged.
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
    origin: str = "explicit"          # "explicit" or "swap"

@dataclass
class PairSet:
        """Ready pairs, packaged with the protocol that generated them.

    items     the sentences in order: for each pair, the true one comes first, then the false one
    pidx      the indices (true, false) pointing into items
    pairs     the Pair objects, aligned with pidx
    protocol  the Protocol instance used, which travels with the data and ends up in the bundles
    skipped   the discarded items along with their reason, to avoid losing them silently
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
        """category -> list of (true_index, false_index)"""
        out = {}
        for k, p in enumerate(self.pairs):
            if p.category:
                out.setdefault(p.category, []).append(self.pidx[k])
        return out

    def index_of_category(self):
        """categori -> posizioni nella lista delle coppie"""
        out = {}
        for k, p in enumerate(self.pairs):
            if p.category:
                out.setdefault(p.category, []).append(k)
        return out

    def subset(self, keep):
        """New PairSet with only the indicated pairs. The indices are
        renumbered, so it remains coherent."""
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
            raise ValueError("local extension not supported: %s" % ext)
        return load_dataset(fmt, data_files=local_file, split="train")
    from datasets import load_dataset
    return load_dataset(protocol.dataset, split="train", revision=protocol.revision)


def _raw_rows(protocol, local_file=None, verbose=True):
    ds = _open_dataset(protocol, local_file)
    if verbose:
        print("  [provenance] %s @ %s  (%d lines)"
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
    """Randomly sampled pairs across all CounterFact, without grouping by relation. 
    This represents the sampling of the canonical truth_probe protocol.
    """
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
        raise RuntimeError("No pairs generated: has the dataset schema changed?")
    return build(pairs, protocol)


def counterfact_by_relation(protocol=CANONICAL, k=33, n_per=60,
                            whitelist=None, local_file=None, verbose=True):
  """Pairs grouped by Wikidata relation: the K relations with the highest 
    number of unique pairs, selecting n_per pairs for each. This represents the sampling 
    of the dictionary family. The uniqueness key is (prompt, true_target), matching 
    the canonical codebase.
    """
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
            print("  [warning] required relations missing: %s" % mancanti)
    else:
        top = sorted(by_rel, key=lambda r: len(by_rel[r]), reverse=True)[:k]

    pairs = []
    for rid in top:
        rws = list(by_rel[rid].values())
        random.Random(protocol.seed).shuffle(rws)
        rws = rws[:n_per]
        if verbose:
            print("    %-7s %3d pairs" % (rid, len(rws)))
        for prompt, tt, tf in rws:
            pairs.append(Pair(prompt, tt, tf, category=rid))
    p = protocol.with_(k_relations=len(top), pairs_per_relation=n_per)
    return build(pairs, p)


def from_json(path, protocol=CANONICAL, verbose=True):
    """A fact file matching the project schema: a list of objects containing 
    'prompts', 'targets', and optional 'false_targets', 'relation_id', 'id'.

    If 'false_targets' is missing, the false target is inferred by permutation 
    within the relation, following the rule declared at the top of the module.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ValueError("the file does not contain a list of items: %s" % path)

    rows = [el for el in data if isinstance(el, dict)]
    scartati = [("(invalid element)", "it is not an object")] * (len(data) - len(rows))

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
            scartati.append((ident, "missing prompt or true target"))
            continue
        tf = first(it, "false_targets")
        origin = "explicit"
        if not tf:
            rid = it.get("relation_id")
            grp = by_rel.get(rid, []) if rid else []
            if len(grp) < 2:
                scartati.append((ident, "no false_target and no alternative"))
                continue
            k = grp.index(pos)
            for step in range(1, len(grp)):
                cand = first(rows[grp[(k + step) % len(grp)]], "targets")
                if cand and cand.strip().lower() != tt.strip().lower():
                    tf, origin = cand, "swap"
                    break
            if not tf:
                scartati.append((ident, "no distinct target within the relation"))
                continue
        pairs.append(Pair(prompt, tt, tf, category=it.get("relation_id"),
                          ident=ident, origin=origin))

    ps = build(pairs, protocol, scartati)
    if verbose:
        print("  [file] %s: %s" % (os.path.basename(path), ps.describe()))
    return ps
