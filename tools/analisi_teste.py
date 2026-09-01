# -*- coding: utf-8 -*-
"""
analisi_teste.py  --  what each attention head writes, against the residual.

Reads the bundles produced by teste_dizionario.py. Loads no model: works on the
dictionaries alone, on CPU, in seconds.

WHAT IS BEING MEASURED, AND WHAT IS NOT
The per-head axes are fitted on what the head WRITES into the residual stream,
that is the OV circuit. They say what the head deposits. They say nothing about
Q and K, which decide WHERE the head reads: the attention pattern is a different
object and needs different measurements. In the current metaphor this measures
the current, not the valve.

SIX QUESTIONS

1. COVERAGE      is the residual global axis reconstructible from the directions
                 the heads write? The fraction of its squared norm inside their
                 span, with the chance value n_heads/d printed beside it. Low
                 coverage does not mean the heads carry no truth: it means they
                 do not carry THAT direction.

2. GEOMETRY      cosines between the heads' global axes. Whether the heads write
                 in agreement, in opposition, or independently of each other.

3. ARRANGEMENT   each head's category matrix against the residual one, under the
                 consensus gauge applied to each view independently, with a
                 category-relabelling permutation null.

4. HEAD TO HEAD  the same, between every pair of heads. This is where a pair
                 that shares a map but writes it with opposite sign shows up,
                 which is propagate-versus-overwrite seen inside attention.

5. PER CATEGORY  coverage of each residual category axis, and which categories
                 the heads cover worst.

6. SUBSPACES     principal angles between the span of the head axes and the span
                 of the residual category axes, plus the effective rank of each.
                 This answers a question the earlier version could only estimate
                 indirectly: how many dimensions the two maps actually share,
                 and how many each of them really occupies.

THREE DECLARED CAUTIONS
  Signed cosines between category axes depend on the gauge; only cycle products
  are invariant. Each view is gauged on its own, as prescribed for comparisons,
  and the thin margins are printed because there the sign is a coin.
  With K categories the permutation test has a floor of one over K factorial:
  below K=6 significance is unreachable whatever the data say.
  Coverage is a property of ONE direction, never of the signal as a whole.

No verdicts are printed.

    python analisi_teste.py --dir teste_dot
    python analisi_teste.py --dir teste_dot --perms 9999
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

from truthprobe import __version__
from truthprobe.geometry import unit, subspace_fraction
from truthprobe.stats import consensus_gauge, apply_gauge, mantel, eigengap
from truthprobe.subspace import principal_angles, subspace_overlap, effective_rank


def gauged(C):
    """The signed matrix under the consensus gauge, plus margins and the list of
    categories whose sign is not identifiable and must be reported unsigned."""
    s, m, thin = consensus_gauge(C)
    return apply_gauge(C, s), m, thin


def load_dir(d):
    res = sorted(glob.glob(os.path.join(d, "*_resid.pt")))
    heads = sorted(glob.glob(os.path.join(d, "*_head*.pt")))
    if not res or not heads:
        sys.exit("the folder needs one *_resid.pt and at least one *_head*.pt")
    R = torch.load(res[0], map_location="cpu", weights_only=False)
    out = []
    for p in heads:
        b = torch.load(p, map_location="cpu", weights_only=False)
        if list(b["cats"]) != list(R["cats"]):
            sys.exit("categories of %s differ from the residual bundle"
                     % os.path.basename(p))
        out.append((int(re.search(r"head(\d+)", os.path.basename(p)).group(1)), b))
    out.sort(key=lambda t: t[0])
    return os.path.basename(res[0]), R, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", required=True, help="folder with the bundles")
    ap.add_argument("--perms", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    name, R, heads = load_dir(a.dir)
    cats = list(R["cats"])
    K, nH = len(cats), len(heads)
    tg_res = unit(R["t_global"].float())
    d = tg_res.shape[0]
    C_res, m_res, thin_res = gauged(R["cos_peak"].float())
    meta = R.get("meta", {}) or {}

    print()
    print("[library] truthprobe %s" % __version__)
    print("[bundle]  %s" % name)
    print("[model]   %s   block %s   K=%d  d=%d  heads=%d"
          % (meta.get("model", "?"), meta.get("peak_block", "?"), K, d, nH))
    proto = meta.get("protocol")
    if proto:
        print("[protocol] suffix %r, join %r" % (proto.get("suffix"), proto.get("join")))
    print("[note] the per-head axes measure the OV circuit, what the head WRITES.")
    print("       Q and K, where the head reads, are not measured here.")

    # ---------------- 1. coverage ----------------
    A = torch.stack([unit(b["t_global"].float()) for _, b in heads], 0)   # [nH, d]
    cov = float(subspace_fraction(tg_res.unsqueeze(0), A)[0])
    coef = torch.linalg.lstsq(A.T, tg_res.unsqueeze(1)).solution.squeeze(1)
    print()
    print("1. COVERAGE   the residual global axis inside the span of the head axes")
    print("   fraction captured  : %.3f" % cov)
    print("   chance value       : %.4f   (n_heads / d = %d / %d)" % (nH / d, nH, d))
    print("   ratio over chance  : %.0fx" % (cov / (nH / d)))
    print("   weight of each head in the reconstruction (least squares):")
    print("     " + "  ".join("h%d %+.2f" % (h, coef[i]) for i, (h, _) in enumerate(heads)))

    # ---------------- 2. geometry between heads ----------------
    G = A @ A.T
    print()
    print("2. GEOMETRY   cosines between the heads' global axes")
    print("      " + " ".join("%6d" % h for h, _ in heads))
    for i, (h, _) in enumerate(heads):
        print("   h%-2d " % h + " ".join("%+6.2f" % G[i, j] for j in range(nH)))
    off = [(float(G[i, j]), heads[i][0], heads[j][0])
           for i in range(nH) for j in range(i + 1, nH)]
    off.sort()
    print("   most anti-aligned pair : h%d and h%d  %+.3f" % (off[0][1], off[0][2], off[0][0]))
    print("   most aligned pair      : h%d and h%d  %+.3f" % (off[-1][1], off[-1][2], off[-1][0]))
    print("   off-diagonal median    : %+.3f"
          % float(torch.tensor([o[0] for o in off]).median()))

    # ---------------- 3. arrangement against the residual ----------------
    print()
    print("3. ARRANGEMENT   each head's category matrix against the residual one")
    print("   (consensus gauge applied to each view independently)")
    print("   %5s %10s %10s %12s %10s" % ("head", "cos resid", "Mantel r", "p", "gauge med"))
    print("   " + "-" * 52)
    Cg, rows = [], []
    for i, (h, b) in enumerate(heads):
        C_h, m_h, thin_h = gauged(b["cos_peak"].float())
        Cg.append(C_h)
        r = mantel(C_h, C_res, perms=a.perms, seed=a.seed + h)
        cosr = float(A[i] @ tg_res)
        print("   %5d %+10.3f %+10.3f %12.4f %10.3f"
              % (h, cosr, r["r"], r["p"], float(m_h.median())))
        rows.append(dict(head=h, cos_global_vs_resid=cosr, mantel_r=r["r"],
                         mantel_p=r["p"], gauge_margin_median=float(m_h.median()),
                         unsigned=[cats[k] for k in thin_h],
                         lstsq_weight=float(coef[i])))
    print()
    print("   unsigned categories in the residual: %s"
          % ([cats[k] for k in thin_res] or "none"))
    eg = eigengap(C_res)
    print("   residual eigengap: lam1 %.3f  lam2 %.3f  relative %.3f"
          % (eg["lam1"], eg["lam2"], eg["rel"]))
    print("   (a small relative gap means the gauge is unstable as a whole, not")
    print("    just on a few categories)")
    if K < 6:
        print("   [warning] with K=%d the permutation p has a floor of 1/%d!" % (K, K))

    # ---------------- 4. head against head ----------------
    print()
    print("4. HEAD TO HEAD   every head's arrangement against every other")
    print("   (r above the diagonal, p below)")
    M = torch.eye(nH)
    P = torch.ones(nH, nH)
    for i in range(nH):
        for j in range(i + 1, nH):
            r = mantel(Cg[i], Cg[j], perms=a.perms, seed=a.seed + 100 * i + j)
            M[i, j] = M[j, i] = r["r"]
            P[i, j] = P[j, i] = r["p"]
    print("      " + " ".join("%6d" % h for h, _ in heads))
    for i, (h, _) in enumerate(heads):
        cells = ["%6s" % "-" if j == i else
                 ("%+6.2f" % M[i, j] if j > i else "%6.3f" % P[i, j])
                 for j in range(nH)]
        print("   h%-2d " % h + " ".join(cells))
    pairs = [(float(M[i, j]), i, j) for i in range(nH) for j in range(i + 1, nH)]
    pairs.sort()
    hid = [h for h, _ in heads]
    bi, bj = pairs[-1][1], pairs[-1][2]
    print("   most similar arrangement : h%d and h%d  r %+.3f  p %.4f  (cos of axes %+.2f)"
          % (hid[bi], hid[bj], pairs[-1][0], float(P[bi, bj]), float(G[bi, bj])))
    print("   most opposite arrangement: h%d and h%d  r %+.3f"
          % (hid[pairs[0][1]], hid[pairs[0][2]], pairs[0][0]))
    print("   (a pair with HIGH r and NEGATIVE cosine shares the map and writes it")
    print("    with opposite sign: propagate versus overwrite, inside attention)")

    # ---------------- 5. coverage per category ----------------
    AX = torch.stack([b["axes"].float() for _, b in heads], 0)            # [nH, K, d]
    AX = AX / AX.norm(dim=2, keepdim=True).clamp_min(1e-12)
    ax_res = R["axes"].float()
    ax_res = ax_res / ax_res.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cov_g = subspace_fraction(ax_res, A)
    flat = AX.reshape(nH * K, d)
    cov_c = subspace_fraction(ax_res, flat)
    print()
    print("5. PER CATEGORY   each residual category axis inside the head spans")
    print("   inside the %d head global axes : median %.3f  (chance %.4f)"
          % (nH, float(cov_g.median()), nH / d))
    print("   inside the %d head category axes: median %.3f  (chance %.4f)"
          % (nH * K, float(cov_c.median()), min(nH * K, d) / d))
    order = sorted(range(K), key=lambda k: float(cov_c[k]))
    print("   least covered: %s"
          % ", ".join("%s %.3f" % (cats[k], cov_c[k]) for k in order[:3]))
    print("   most covered : %s"
          % ", ".join("%s %.3f" % (cats[k], cov_c[k]) for k in order[-3:]))

    # ---------------- 6. subspaces ----------------
    print()
    print("6. SUBSPACES   how many dimensions the two maps share, and how many")
    print("   each of them really occupies")
    er_res = effective_rank(ax_res)
    er_head = effective_rank(A)
    er_hcat = effective_rank(flat)
    print("   effective rank, residual category axes : %.2f out of %d"
          % (er_res["effective_rank"], K))
    print("   effective rank, head global axes       : %.2f out of %d"
          % (er_head["effective_rank"], nH))
    print("   effective rank, head category axes     : %.2f out of %d"
          % (er_hcat["effective_rank"], nH * K))
    pa = principal_angles(A, ax_res)
    ang = pa["angles"]
    print("   principal angles, head global axes against residual category axes:")
    print("     " + "  ".join("%.1f" % float(x) for x in ang))
    print("     shared below 45 degrees: %d   two random subspaces would share %d"
          % (pa["shared_at_45"], pa["random_shared_at_45"]))
    ov = subspace_overlap(A, ax_res)
    print("   overlap %.3f   (chance %.3f)" % (ov["overlap"], ov["chance"]))

    out = a.out or os.path.join(a.dir, "analisi_teste.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__,
                       model=meta.get("model"), peak_block=meta.get("peak_block"),
                       protocol=proto, cats=cats, n_heads=nH, d=d,
                       coverage=cov, coverage_chance=nH / d,
                       head_cosines=[[float(x) for x in r] for r in G],
                       head_mantel_r=[[float(x) for x in r] for r in M],
                       head_mantel_p=[[float(x) for x in r] for r in P],
                       coverage_per_category=dict(
                           span_global=[float(x) for x in cov_g],
                           span_categories=[float(x) for x in cov_c]),
                       effective_rank=dict(residual=er_res["effective_rank"],
                                           head_global=er_head["effective_rank"],
                                           head_categories=er_hcat["effective_rank"]),
                       principal_angles=[float(x) for x in ang],
                       shared_at_45=pa["shared_at_45"],
                       random_shared_at_45=pa["random_shared_at_45"],
                       subspace_overlap=ov["overlap"],
                       heads=rows), fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % out)
    print()


if __name__ == "__main__":
    main()
