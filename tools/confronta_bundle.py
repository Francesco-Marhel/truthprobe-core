# -*- coding: utf-8 -*-
"""
confronta_bundle.py  --  are two bundles the same measurement?

Converted to a thin wrapper over the truthprobe core. It used to carry its own
gauge, its own Mantel and its own permutation loop; all three are now library
functions, so what is left is the comparison itself.

TWO CHECKS THAT MUST STAY SEPARATE
  PROTOCOL   identifies the QUESTION. Two bundles built with different sentence
             conventions are not the same measurement done badly, they are two
             different measurements, and comparing them is a category error.
             This is checked FIRST and reported, never silently mixed in.
  CONTENT    identifies the ARTEFACT. Two bundles with the same content share a
             cosine matrix even under different filenames or category orders.
             This is the fingerprint that once revealed a bundle filed under
             the wrong name, and that the categories were the same set in a
             different order.

Keeping them apart makes the diagnosis immediate: different protocol means a
different convention, same protocol with a different fingerprint means
something changed in the data or the code.

WHAT IS REPORTED
  Mantel on the SIGNED cells, with the consensus gauge applied to each bundle
  independently, and Mantel on the ABSOLUTE values, which needs no gauge at
  all. If the signed one is low while the absolute one is high, only the
  orientation differs and the structure is shared.
  The triple Mantel on cycle products, which is gauge-FREE by construction and
  is the number to use when the two gauges are not comparable.
  Cosines between the global axes and between each pair of category axes.
  Within-category AUC from the transfer diagonal, and the concordance of their
  ORDERING, which is what the knowledge gate actually uses: the gate ranks
  categories, it does not read absolute levels.

No verdicts are printed.

    python confronta_bundle.py --a first.pt --b second.pt
"""

import argparse
import json
import os
import sys

import torch

from truthprobe import Protocol, __version__
from truthprobe.geometry import unit
from truthprobe.stats import (consensus_gauge, apply_gauge, mantel,
                              triple_mantel, eigengap, frustration)


def load(path):
    if not os.path.isfile(path):
        sys.exit("file not found: %s" % path)
    return torch.load(path, map_location="cpu", weights_only=False)


def gauged(C):
    s, m, thin = consensus_gauge(C)
    return apply_gauge(C, s), s, m, thin


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--perms", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="compare even if the protocols differ. The numbers are "
                         "then a comparison BETWEEN CONVENTIONS, which is a "
                         "legitimate question but a different one.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    ca, cb = list(A["cats"]), list(B["cats"])
    if set(ca) != set(cb):
        sys.exit("different category sets: only in A %s, only in B %s"
                 % (sorted(set(ca) - set(cb)), sorted(set(cb) - set(ca))))
    perm = [ca.index(c) for c in cb]              # reorder A into B's order
    K = len(cb)
    ma = A.get("meta", {}) or {}
    mb = B.get("meta", {}) or {}

    print()
    print("[library] truthprobe %s" % __version__)
    print()
    print("%-14s %-30s %s" % ("", os.path.basename(a.a)[:30], os.path.basename(a.b)[:30]))
    for k in ("model", "peak_block", "view", "component", "k_relations",
              "pairs_per_relation", "seed", "truthprobe_version"):
        va, vb = ma.get(k, "?"), mb.get(k, "?")
        print("%-14s %-30s %-30s%s" % (k[:14], repr(va)[:30], repr(vb)[:30],
                                       "" if va == vb else "   <-- DIFFERENT"))

    # ---------------- protocol: the QUESTION ----------------
    pa = Protocol.from_dict(ma.get("protocol"))
    pb = Protocol.from_dict(mb.get("protocol"))
    print()
    print("PROTOCOL")
    if pa is None or pb is None:
        print("  at least one bundle carries no protocol: it predates the")
        print("  convention being made explicit. Comparability cannot be verified.")
        if ca != cb:
            print("  (category order differs and has been aligned)")
    elif pa.compatible_with(pb):
        print("  compatible: suffix %r, join %r, pool %r" % (pa.suffix, pa.join, pa.pool))
    else:
        d = pa.diff(pb)
        print("  INCOMPATIBLE, the two do not measure the same object:")
        for k, (x, y) in d.items():
            print("    %-12s %r  against  %r" % (k, x, y))
        print("  Measured on Gemma-2-2b, changing only the suffix leaves the")
        print("  arrangement at Mantel +0.775 and moves the axes to cosine +0.52.")
        if not a.force:
            sys.exit("  Stopping. Use --force to compare conventions on purpose.")
        print("  --force given: what follows is a comparison BETWEEN conventions.")

    CA = A["cos_peak"].float()[perm][:, perm]
    CB = B["cos_peak"].float()
    GA, sa, mga, thin_a = gauged(CA)
    GB, sb, mgb, thin_b = gauged(CB)

    # ---------------- arrangement: the CONTENT ----------------
    r_s = mantel(GA, GB, perms=a.perms, seed=a.seed)
    r_a = mantel(CA.abs(), CB.abs(), perms=a.perms, seed=a.seed + 1)
    r_t = triple_mantel(CA, CB, perms=min(a.perms, 5000), seed=a.seed + 2)
    print()
    print("ARRANGEMENT")
    print("  Mantel, signed cells (consensus gauge per bundle) : %+.3f   p %.4f"
          % (r_s["r"], r_s["p"]))
    print("  Mantel, absolute values (no gauge involved)       : %+.3f   p %.4f"
          % (r_a["r"], r_a["p"]))
    print("  Triple Mantel on cycle products (gauge-FREE)      : %+.3f   p %.4f"
          % (r_t["r"], r_t["p"]))
    flip = [cb[i] for i in range(K) if sa[i] != sb[i]]
    print("  categories gauged with opposite sign : %s" % (flip or "none"))
    print("  unsigned in A: %s" % ([cb[i] for i in thin_a] or "none"))
    print("  unsigned in B: %s" % ([cb[i] for i in thin_b] or "none"))
    if r_s["floor"]:
        print("  [warning] with K=%d the permutation p has a floor of %.4f"
              % (K, r_s["floor"]))
    ea, eb = eigengap(CA), eigengap(CB)
    print("  relative eigengap: A %.3f   B %.3f   (small means the whole gauge "
          "is unstable, not just a few categories)" % (ea["rel"], eb["rel"]))
    fa, fb = frustration(CA, perms=500), frustration(CB, perms=500)
    print("  frustrated triangles: A %.3f (null %.3f)   B %.3f (null %.3f)"
          % (fa["frac"], fa["null_mean"], fb["frac"], fb["null_mean"]))

    # ---------------- axes ----------------
    axA = A["axes"].float()[perm]
    axB = B["axes"].float()
    axA = axA / axA.norm(dim=1, keepdim=True).clamp_min(1e-12)
    axB = axB / axB.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_cat = (axA * axB).sum(1)
    tgA, tgB = unit(A["t_global"].float()), unit(B["t_global"].float())
    order = sorted(range(K), key=lambda i: abs(float(cos_cat[i])))
    print()
    print("AXES")
    print("  cosine between the global axes : %+.3f" % float(tgA @ tgB))
    print("  per category: median %+.3f   |cos| median %.3f   min %+.3f   max %+.3f"
          % (float(cos_cat.median()), float(cos_cat.abs().median()),
             float(cos_cat.min()), float(cos_cat.max())))
    print("  most different : %s"
          % ", ".join("%s %+.2f" % (cb[i], cos_cat[i]) for i in order[:3]))
    print("  most similar   : %s"
          % ", ".join("%s %+.2f" % (cb[i], cos_cat[i]) for i in order[-3:]))

    # ---------------- knowledge proxy ----------------
    res_tr = None
    if A.get("transfer") is not None and B.get("transfer") is not None:
        TA = A["transfer"].float()[perm][:, perm]
        TB = B["transfer"].float()
        wa = torch.tensor([TA[i, i] for i in range(K)])
        wb = torch.tensor([TB[i, i] for i in range(K)])
        conc = sum(1 for i in range(K) for j in range(i + 1, K)
                   if (wa[i] - wa[j]) * (wb[i] - wb[j]) > 0)
        tot = K * (K - 1) // 2
        print()
        print("WITHIN-CATEGORY AUC (transfer diagonal)")
        print("  median A %.3f   median B %.3f   median difference %+.3f"
              % (float(wa.median()), float(wb.median()), float((wa - wb).median())))
        print("  above 0.6: A %d/%d   B %d/%d"
              % (int((wa >= .6).sum()), K, int((wb >= .6).sum()), K))
        print("  ORDERING concordance : %d/%d pairs = %.2f" % (conc, tot, conc / tot))
        print("  (the gate ranks categories, it does not read absolute levels:")
        print("   if the ordering holds, the gate holds even if the levels move)")
        res_tr = dict(median_a=float(wa.median()), median_b=float(wb.median()),
                      above_a=int((wa >= .6).sum()), above_b=int((wb >= .6).sum()),
                      concordance=conc / tot)

    out = a.out or "confronto_bundle.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__,
                       a=os.path.abspath(a.a), b=os.path.abspath(a.b), cats=cb,
                       protocol_a=(pa.to_dict() if pa else None),
                       protocol_b=(pb.to_dict() if pb else None),
                       mantel_signed=r_s, mantel_abs=r_a, mantel_triple=r_t,
                       gauge_flips=flip,
                       cos_global=float(tgA @ tgB),
                       cos_per_category={cb[i]: float(cos_cat[i]) for i in range(K)},
                       eigengap_a=ea, eigengap_b=eb, transfer=res_tr),
                  fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % out)
    print()


if __name__ == "__main__":
    main()
