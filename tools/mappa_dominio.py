# -*- coding: utf-8 -*-
"""
mappa_dominio.py  --  does the dictionary recognise this contrast, and where
does the signal live?

Merges the former mappa_dominio.py and span_svd.py. They shared six identical
helpers, all now in the library, and both did the same operation: project the
intra-pair difference vectors onto the dictionary. Two readings of one
projection are better as one tool than as two.

THE IDEA
Instead of asking the dictionary who is right, ask it whether it recognises the
ground. Every difference vector splits exactly into the part inside the span of
the K category axes and the residual outside it. What is inside, what is
outside, and whether either separates true from false, are three different
questions.

WHAT IT REPORTS

  RECOGNITION (per item, no label needed for the first two)
    inside    fraction of the difference's squared norm inside the span of the
              K axes. Says whether you are in the dictionary's domain. The
              chance value is K/d and is printed beside it: 33 directions in
              2304 capture about 1.4% of a random vector, so anything above a
              few percent is already structure.
    shape     how much the projection profile resembles a row of the peak
              cosine matrix. Says whether you are inside the STRUCTURE, not
              merely inside the space: a difference can sit in the span and
              match no category.
    global    signed cosine with the global axis. The truth verdict.

  CHANNELS (held out over pairs, with a permutation null)
    inside    the best direction expressible by the dictionary
    outside   the dominant direction of what the dictionary does NOT see
    Each is reported against its theoretical ceiling: a direction inside the
    span cannot exceed sqrt(inside), one outside cannot exceed sqrt(1-inside).
    A gain must be read against its maximum, never against zero.

  TRANSFER (optional)
    the same directions, fitted here and read elsewhere WITHOUT refitting, with
    a null that keeps the direction fixed and permutes the arrival labels.

A DECLARED LIMIT ON THE NEGATIVE
On planted ground truth, a weak signal outside the span is recovered by the
supervised probe at 0.70 with 66 pairs, 0.85 with 150, 0.90 with 400. At a few
dozen pairs the correct statement is that no direction outside the span
separates, NOT that nothing is there.

No verdicts are printed.

    python mappa_dominio.py --bundle dict.pt --facts fatti.json \\
           --model google/gemma-2-2b --peak 11
    python mappa_dominio.py --bundle dict.pt --facts fatti.json --model ... \\
           --peak 11 --channels --probe --transfer altri.json
"""

import argparse
import json
import math
import os
import random
import sys

import torch

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import from_json
from truthprobe.geometry import unit, subspace_fraction
from truthprobe.hooks import describe, BlockCapture
from truthprobe.stats import pearson, kfold_pairs, se_binomial


# =====================================================================
#  extraction
# =====================================================================
@torch.no_grad()
def differences(model, tok, ps, block, component, arch, device, batch):
    """The intra-pair difference vectors, read on the requested stream.

    With component 'resid' the state comes from hidden_states; otherwise the
    additive contribution of that block is captured by a hook in the SAME
    forward pass, so reading a component costs nothing extra.

    Reading the residual on axes fitted on a component, or the reverse, is
    measurable but is not the same object: the two live in the same space and
    not in the same informative subspace. The tool refuses to guess and takes
    the stream from the bundle unless told otherwise."""
    level = block + 1
    out_v = []
    cap = None if component == "resid" else BlockCapture(model, arch, block)
    if cap:
        cap.__enter__()
    try:
        for s in range(0, len(ps.items), batch):
            enc = tok(ps.items[s:s + batch], return_tensors="pt",
                      padding=True).to(device)
            o = model(**enc, output_hidden_states=(component == "resid"))
            if component == "resid":
                out_v.append(o.hidden_states[level][:, -1, :].float().cpu())
            else:
                out_v.append(cap.attn() if component == "attn" else cap.ffn())
            print("\r  [extract] %d/%d" % (min(s + batch, len(ps.items)),
                                           len(ps.items)), end="", flush=True)
        print()
    finally:
        if cap:
            cap.__exit__(None, None, None)
    H = torch.cat(out_v, 0)
    return torch.stack([H[i] - H[j] for i, j in ps.pidx], 0)


# =====================================================================
#  channels: the best direction inside and outside the span
# =====================================================================
def top_dir(X):
    """Dominant direction of the rows of X. By the label-free lemma its span
    does not depend on the within-pair signs."""
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return unit(Vh[0])


def orient(w, D, idx, signs):
    s = (D[idx] @ w) * torch.tensor([signs[i] for i in idx], dtype=D.dtype)
    return w if float(s.sum()) >= 0 else -w


def paired_acc(scores, signs):
    good = sum(1 for s, y in zip(scores.tolist(), signs) if s * y > 0)
    tie = sum(1 for s in scores.tolist() if s == 0.0)
    return (good + 0.5 * tie) / len(signs)


def fit_probe(X, signs, l2=1e-2, steps=400, lr=0.1):
    """Supervised direction: bias-free logistic on the rows of X. Used only to
    ask whether a signal exists that the SVD misses because the dominant
    variance points elsewhere."""
    y = torch.tensor(signs, dtype=X.dtype)
    w = torch.zeros(X.shape[1], dtype=X.dtype, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.softplus(-((X @ w) * y)).mean() + l2 * w.pow(2).sum()
        loss.backward()
        opt.step()
    return unit(w.detach())


def cv_channels(D, Q, folds, seed, signs, probe=False, m_out=20):
    """Held-out accuracy of the best direction inside and outside the span.

    The subspace basis Q is fixed (it comes from the dictionary), but the
    direction within each channel is refitted on training folds only, and for
    the outside channel the reduction basis is estimated on the training folds
    too, otherwise the test pairs would leak into the basis."""
    sg = torch.tensor(signs, dtype=D.dtype).unsqueeze(1)
    Ds = D * sg
    C = Ds @ Q
    R = Ds - C @ Q.T
    acc = {k: [] for k in ("inside", "outside", "p_inside", "p_outside")}
    for tr, te in kfold_pairs(D.shape[0], folds, seed):
        w_in = orient(unit(Q @ top_dir(C[tr])), D, tr, signs)
        w_out = orient(top_dir(R[tr]), D, tr, signs)
        acc["inside"].append(paired_acc(D[te] @ w_in, [signs[i] for i in te]))
        acc["outside"].append(paired_acc(D[te] @ w_out, [signs[i] for i in te]))
        if probe:
            s_tr = [signs[i] for i in tr]
            w_pi = unit(Q @ fit_probe(C[tr], s_tr))
            _, _, Vh = torch.linalg.svd(R[tr], full_matrices=False)
            B = Vh[:min(m_out, Vh.shape[0])].T
            w_po = unit(B @ fit_probe(R[tr] @ B, s_tr))
            acc["p_inside"].append(paired_acc(D[te] @ w_pi, [signs[i] for i in te]))
            acc["p_outside"].append(paired_acc(D[te] @ w_po, [signs[i] for i in te]))
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}


def with_null(D, Q, folds, seed, perms, probe, m_out):
    n = D.shape[0]
    obs = cv_channels(D, Q, folds, seed, [1] * n, probe, m_out)
    rng = random.Random(seed)
    null = {k: [] for k in obs}
    for b in range(perms):
        sg = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        r = cv_channels(D, Q, folds, seed, sg, probe, m_out)
        for k in obs:
            null[k].append(r[k])
        print("\r  [null] %d/%d" % (b + 1, perms), end="", flush=True)
    if perms:
        print()
    out = {}
    for k, o in obs.items():
        if o != o:
            continue
        v = sorted(x for x in null[k] if x == x)
        q = v[int(0.95 * (len(v) - 1))] if v else float("nan")
        p = (1 + sum(1 for x in v if x >= o)) / (len(v) + 1) if v else float("nan")
        out[k] = dict(obs=o, null_mean=(sum(v) / len(v) if v else float("nan")),
                      null_p95=q, p=p)
    return out


# =====================================================================
#  main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--peak", type=int, required=True, help="BLOCK")
    ap.add_argument("--component", default=None, choices=["resid", "attn", "ffn"],
                    help="which stream to read. Default: the one recorded in the "
                         "bundle, so the reading always matches the fit.")
    ap.add_argument("--fit-here", action="store_true",
                    help="fit a fresh global axis ON THIS FILE instead of using the "
                         "bundle's, held out over pairs with a permutation null, and "
                         "transfer it. Answers a question the bundle cannot pose: "
                         "does a truth direction exist on THIS material at all. The "
                         "dictionary is still used for the span, so the recognition "
                         "columns keep their meaning.")
    ap.add_argument("--channels", action="store_true",
                    help="also compute the best direction inside and outside the span")
    ap.add_argument("--probe", action="store_true",
                    help="add the supervised direction, to find signal the SVD misses")
    ap.add_argument("--transfer", nargs="*", default=[],
                    help="files to read the fitted directions on, without refitting")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--m-out", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    for p in (a.bundle, a.facts):
        if not os.path.isfile(p):
            sys.exit("file not found: %s" % p)

    b = torch.load(a.bundle, map_location="cpu", weights_only=False)
    axes = b["axes"].float()
    axes = axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    tg = unit(b["t_global"].float())
    C_peak = b["cos_peak"].float()
    cats = list(b["cats"])
    K, d = axes.shape
    meta = b.get("meta", {}) or {}
    proto = Protocol.from_dict(meta.get("protocol")) or CANONICAL
    comp = a.component or meta.get("component", meta.get("view", "resid"))
    if comp.startswith("head"):
        comp = "attn"

    print()
    print("[library] truthprobe %s" % __version__)
    print("[bundle]  %s   K=%d  d=%d   stream %r"
          % (os.path.basename(a.bundle), K, d, comp))
    print("[protocol] suffix %r, join %r" % (proto.suffix, proto.join))
    ps = from_json(a.facts, proto)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=(torch.bfloat16 if a.dtype == "bfloat16" else torch.float32),
        use_safetensors=True, trust_remote_code=False).to(device)
    model.eval()
    arch = describe(model)
    if model.config.hidden_size != d:
        sys.exit("dimension mismatch: bundle %d, model %d" % (d, model.config.hidden_size))
    print("[model] %s on %s (%s)   block %d" % (a.model, device, a.dtype, a.peak))

    D = differences(model, tok, ps, a.peak, comp, arch, device, a.batch)
    Q, _ = torch.linalg.qr(axes.T)                      # orthonormal basis of the span

    # ---------------- recognition ----------------
    nrm = D.norm(dim=1).clamp_min(1e-12)
    U = D / nrm.unsqueeze(1)
    inside = subspace_fraction(D, axes)
    prof = U @ axes.T                                   # [n, K]
    shape = torch.stack([torch.tensor([pearson(prof[i], C_peak[k]) for k in range(K)])
                         for i in range(D.shape[0])], 0)
    best = shape.abs().max(dim=1)
    glob = U @ tg

    med = lambda t: float(t.median())
    print()
    print("=== RECOGNITION ===")
    print("  items                    : %d" % D.shape[0])
    print("  inside the span (median) : %.3f   chance %.4f (K/d)   ratio %.0fx"
          % (med(inside), K / d, med(inside) / (K / d)))
    print("  shape, |max corr| median : %.3f" % med(best.values))
    print("  cosine with global axis  : %+.3f" % med(glob))
    print("  difference norm (median) : %.1f" % med(nrm))
    pos = int((glob > 0).sum())
    print("  global sign positive     : %d/%d  %.1f%%  (se %.3f)"
          % (pos, len(glob), 100 * pos / len(glob),
             se_binomial(pos / len(glob), len(glob))))
    print("  theoretical ceiling: inside %.3f   outside %.3f"
          % (math.sqrt(med(inside)), math.sqrt(1 - med(inside))))

    res = dict(n=D.shape[0], inside=med(inside), shape=med(best.values),
               global_cos=med(glob), norm=med(nrm),
               sign_positive=pos / len(glob), chance_inside=K / d)

    # ---------------- channels ----------------
    if a.channels:
        print()
        print("=== CHANNELS (held out, %d permutations) ===" % a.perms)
        ch = with_null(D, Q, a.folds, a.seed, a.perms, a.probe, a.m_out)
        print("  %-10s %9s %9s %9s %9s %9s"
              % ("channel", "held-out", "null mean", "null 95", "margin", "p"))
        print("  " + "-" * 60)
        names = [("inside", "inside"), ("outside", "outside")]
        if a.probe:
            names += [("p_inside", "P inside"), ("p_outside", "P outside")]
        for k, lab in names:
            if k not in ch:
                continue
            r = ch[k]
            print("  %-10s %9.3f %9.3f %9.3f %+9.3f %9.4f"
                  % (lab, r["obs"], r["null_mean"], r["null_p95"],
                     r["obs"] - r["null_p95"], r["p"]))
        res["channels"] = ch
        if a.probe:
            print("  (P = supervised. It finds signal the SVD misses when the")
            print("   dominant variance points elsewhere; at a few dozen pairs its")
            print("   absence is non-detection, not evidence of absence.)")

    # ---------------- an axis fitted here ----------------
    w_here = None
    if a.fit_here:
        n = D.shape[0]
        ones = [1] * n

        def cv_free(signs):
            acc = []
            for tr, te in kfold_pairs(n, a.folds, a.seed):
                w = orient(top_dir(D[tr] * torch.tensor(
                    [signs[i] for i in tr], dtype=D.dtype).unsqueeze(1)),
                    D, tr, signs)
                acc.append(paired_acc(D[te] @ w, [signs[i] for i in te]))
            return sum(acc) / len(acc)

        obs = cv_free(ones)
        rg = random.Random(a.seed)
        nl = []
        for b_ in range(a.perms):
            nl.append(cv_free([1 if rg.random() < 0.5 else -1 for _ in range(n)]))
            print("\r  [null] %d/%d" % (b_ + 1, a.perms), end="", flush=True)
        if a.perms:
            print()
        nl.sort()
        q95 = nl[int(0.95 * (len(nl) - 1))]
        pv = (1 + sum(1 for x in nl if x >= obs)) / (len(nl) + 1)
        w_here = orient(top_dir(D), D, list(range(n)), ones)
        print()
        print("=== AXIS FITTED HERE (held out, %d permutations) ===" % a.perms)
        print("  paired accuracy %.3f   null mean %.3f   null 95 %.3f   margin %+.3f   p %.4f"
              % (obs, sum(nl) / len(nl), q95, obs - q95, pv))
        print("  cosine with the bundle's global axis: %+.3f" % float(w_here @ tg))
        print("  (a margin at or below zero means no truth direction is recoverable")
        print("   from this material, whatever the dictionary says)")
        res["fit_here"] = dict(obs=obs, null_mean=sum(nl) / len(nl), null_p95=q95,
                               p=pv, cos_with_bundle=float(w_here @ tg))

    # ---------------- transfer ----------------
    if a.transfer:
        n = D.shape[0]
        ones = [1] * n
        C_all = D @ Q
        w_in = orient(unit(Q @ top_dir(C_all)), D, list(range(n)), ones)
        w_out = orient(top_dir(D - C_all @ Q.T), D, list(range(n)), ones)
        print()
        print("=== TRANSFER  (directions fitted on %s, NOT refitted) ==="
              % os.path.basename(a.facts))
        cols = ("file", "n", "inside", "outside") + (("free",) if w_here is not None else ())
        print("  %-34s %5s %9s %9s%s" % (cols[0], cols[1], cols[2], cols[3],
                                         "%9s" % "free" if w_here is not None else ""))
        print("  " + "-" * (62 + (9 if w_here is not None else 0)))
        rows = []
        for path in a.transfer:
            if not os.path.isfile(path):
                print("  %-34s   file not found" % os.path.basename(path)[:34])
                continue
            pt = from_json(path, proto, verbose=False)
            Dt = differences(model, tok, pt, a.peak, comp, arch, device, a.batch)
            o = [1] * Dt.shape[0]
            ai, ao = paired_acc(Dt @ w_in, o), paired_acc(Dt @ w_out, o)
            rg = random.Random(a.seed + 7)
            si, so = Dt @ w_in, Dt @ w_out
            ni, no = [], []
            for _ in range(a.perms):
                sg = [1 if rg.random() < 0.5 else -1 for _ in range(Dt.shape[0])]
                ni.append(paired_acc(si, sg))
                no.append(paired_acc(so, sg))
            f = lambda obs, nl: ((1 + sum(1 for x in nl if x >= obs)) / (len(nl) + 1),
                                 sorted(nl)[int(0.95 * (len(nl) - 1))])
            pi, qi = f(ai, ni)
            po, qo = f(ao, no)
            af = paired_acc(Dt @ w_here, o) if w_here is not None else None
            print("  %-34s %5d %9.3f %9.3f%s"
                  % (os.path.basename(path)[:34], Dt.shape[0], ai, ao,
                     "%9.3f" % af if af is not None else ""))
            print("  %-34s %5s   null95 %.3f p %.4f | null95 %.3f p %.4f"
                  % ("", "", qi, pi, qo, po))
            rows.append(dict(file=path, n=Dt.shape[0], inside=ai, outside=ao,
                             inside_p=pi, outside_p=po, free=af))
        res["transfer"] = rows

    out = a.out or ("mappa_%s_%s__%s.json"
                    % (a.label or os.path.splitext(os.path.basename(a.facts))[0],
                       comp, a.model.replace("/", "_")))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__, bundle=os.path.abspath(a.bundle),
                       facts=os.path.abspath(a.facts), model=a.model,
                       block=a.peak, component=comp, protocol=proto.to_dict(),
                       cats=cats, K=K, d=d, results=res),
                  fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % out)
    print()


if __name__ == "__main__":
    main()
