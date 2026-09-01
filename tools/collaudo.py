# -*- coding: utf-8 -*-
"""
collaudo.py  --  commissioning the library against the published numbers.

Two modes, one file.

  DENSE   recompute, using ONLY library functions, quantities whose value is
          already published, and print them next to the expected figure. This
          does not reproduce the canonical tools: it checks that the library
          lands on the same numbers by an independent path. If it does, every
          tool built on it inherits that.

  MOE     the whole Part-2 loop on a mixture of experts: explicit wiring,
          identity gate, dictionary, per-head decomposition, router capture.
          Nothing here was written for a specific MoE; the architecture is
          described from outside in three lines.

WHY A TOLERANCE, AND WHY IT IS WIDE
The expected figures come from runs with a different pair sample and, for the
dictionary family, a different sentence convention. They are not meant to match
to the third decimal: the point is whether the library lands in the same place,
not whether it reproduces a specific run bit for bit. A figure inside the band
means the machinery agrees; one far outside means something structural differs
and deserves a look. The band is printed with every line.

    python collaudo.py --model Qwen/Qwen2.5-1.5B --peak 15
    python collaudo.py --model google/gemma-2-2b --peak 11
    python collaudo.py --model allenai/OLMoE-1B-7B-0924 --peak 8 --moe
"""

import argparse
import json
import os
import sys
import time

import torch

from truthprobe import Protocol, CANONICAL, __version__
from truthprobe.data import counterfact_flat, counterfact_by_relation
from truthprobe.geometry import unit, fit_axis, cosine_matrix, subspace_fraction
from truthprobe.hooks import (Wiring, describe, BlockCapture,
                              identity_gate, head_gate)
from truthprobe.stats import (auc_score, kfold_pairs, paired_accuracy,
                              project_and_score, consensus_gauge, eigengap,
                              frame_gap, se_binomial)
from truthprobe.subspace import effective_rank, principal_angles


# Published values. Only what is actually known, with the source of each.
EXPECTED = {
    "Qwen/Qwen2.5-1.5B": dict(peak=15, axis_auc=0.793, probe_auc=0.854,
                              auc_attn=0.849, auc_ffn=0.736,
                              pre_attn=1.31, pre_ffn=-0.41,
                              post_ffn_b=0.57, post_ffn_b1=-0.82,
                              note="anatomy.py at block 15 (residual 0.793, "
                                   "attention 0.849, FFN 0.736); Table 9 clean "
                                   "frame 15: attention +1.31, FFN -0.41"),
    "google/gemma-2-2b": dict(peak=11, axis_auc=0.776, probe_auc=0.875,
                              note="Part I, and the gap that instruction tuning closes"),
    "google/gemma-2-2b-it": dict(peak=11, axis_auc=0.876, probe_auc=0.884,
                                 note="instruct: the gap falls to under a hundredth"),
    "meta-llama/Llama-3.2-3B": dict(peak=9, pre_attn=1.63, pre_ffn=-1.43,
                                    post_ffn_b=0.49, post_ffn_b1=-1.25, note="Part I"),
    "Qwen/Qwen2.5-3B": dict(peak=16, pre_attn=1.38, pre_ffn=-1.16,
                            post_ffn_b=0.69, post_ffn_b1=-1.23, note="Part I"),
}


def _skip_lm_head(model):
    """Replace the output head with a stub that returns a one-token logit.

    The measurements here read hidden states and block contributions; the logits
    are never used. On a model with a very large vocabulary computing them costs
    more memory than everything else combined, and on a 12 GB device that alone
    can end the run. Returns a callable that restores the original head."""
    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        return lambda: None
    import torch.nn as nn
    w = head.weight
    stub = nn.Linear(w.shape[1], 1, bias=False).to(device=w.device, dtype=w.dtype)
    model.lm_head = stub

    def restore():
        model.lm_head = head
    return restore


def band(value, expected, tol):
    if expected is None:
        return "  (no published figure)"
    lo, hi = expected - tol, expected + tol
    inside = lo <= value <= hi
    return "  expected %.3f, band [%.3f, %.3f]   %s" % (
        expected, lo, hi, "INSIDE" if inside else "OUTSIDE")


def full_probe_auc(H, pidx, folds, seed, l2=1e-2, steps=300, lr=0.1):
    """Held-out AUC of a full linear probe, the ceiling the axis is compared to.

    The gap between this and the axis is the information living off-axis, and
    it is the currency of the dimensionality relation: instruction tuning does
    not add truth information, it realigns it onto a single direction, and the
    gap collapses while this ceiling barely moves."""
    aucs = []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        I = [i for p in tr for i in (pidx[p][0], pidx[p][1])]
        y = torch.tensor([1.0, 0.0] * len(tr))
        X = H[I].float()
        X = X - X.mean(0, keepdim=True)
        w = torch.zeros(X.shape[1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            z = X @ w
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) \
                + l2 * w.pow(2).sum()
            loss.backward()
            opt.step()
        J = [i for p in te for i in (pidx[p][0], pidx[p][1])]
        lab = torch.tensor([1, 0] * len(te))
        aucs.append(auc_score(H[J].float() @ w.detach(), lab))
    return sum(aucs) / len(aucs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--peak", type=int, default=None,
                    help="default: the published peak for this model")
    ap.add_argument("--moe", action="store_true", help="run the mixture-of-experts loop")
    ap.add_argument("--max-pairs", type=int, default=250,
                    help="pairs drawn at random for the axis, the probe and the "
                         "law: the canonical protocol uses 250")
    ap.add_argument("--k-relations", type=int, default=8)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--probe-l2", type=float, default=1e-2,
                    help="ridge on the full linear probe. With a few hundred pairs "
                         "in thousands of dimensions the probe is underdetermined, "
                         "so this value decides where its ceiling lands, and with "
                         "it the gap that carries the dimensionality relation.")
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--tol", type=float, default=0.06,
                    help="half-width of the agreement band")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--out", default="collaudo.json")
    a = ap.parse_args()

    exp = EXPECTED.get(a.model, {})
    peak = a.peak if a.peak is not None else exp.get("peak")
    if peak is None:
        sys.exit("no published peak for %s: pass --peak. The peak is "
                 "model-specific and must be measured, never guessed." % a.model)

    proto = CANONICAL.with_(seed=a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print()
    print("=" * 78)
    print("COMMISSIONING  truthprobe %s" % __version__)
    print("=" * 78)
    print("[model]    %s   peak block %d" % (a.model, peak))
    if exp.get("note"):
        print("[source]   %s" % exp["note"])
    print("[protocol] suffix %r, join %r  ->  %r"
          % (proto.suffix, proto.join,
             proto.sentence("The capital of France is", " Paris")))

    # TWO LOADERS, ON PURPOSE, AND NOT INTERCHANGEABLE.
    #
    # The canonical protocol fits the axis, the probe, the component figures and
    # the relational law on pairs drawn AT RANDOM from the whole dataset: many
    # relations with few pairs each, therefore heterogeneous material. Only the
    # arrangement needs pairs GROUPED by relation, because without categories
    # there are no per-category axes.
    #
    # Using the grouped set for everything, as an earlier version of this file
    # did, changes the composition of the material and with it the scale of the
    # AUC. The library offers both loaders and chooses neither: the choice
    # belongs to the script, and a script that leaves it implicit produces
    # numbers that cannot be compared with the published ones.
    ps_flat = counterfact_flat(proto, max_pairs=a.max_pairs,
                               local_file=a.file_counterfact, verbose=False)
    ps_cat = counterfact_by_relation(proto, k=a.k_relations,
                                     n_per=a.pairs_per_relation,
                                     local_file=a.file_counterfact, verbose=False)
    cats = ps_cat.categories
    print("[data]     axis and law : %d pairs drawn at random (canonical protocol)"
          % len(ps_flat.pidx))
    print("           arrangement  : %d pairs in %d relations"
          % (len(ps_cat.pidx), len(cats)))
    # one extraction over the union, so each sentence is passed once
    items = list(ps_flat.items)
    offset = len(items)
    items += list(ps_cat.items)
    pidx_flat = list(ps_flat.pidx)
    pidx_cat = [(i + offset, j + offset) for i, j in ps_cat.pidx]
    cat_of = {}
    for k, (i, j) in enumerate(pidx_cat):
        cat_of.setdefault(ps_cat.pairs[k].category, []).append((i, j))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, use_safetensors=True,
        trust_remote_code=False).to(device)
    model.eval()

    wiring = None
    if a.moe:
        L0 = getattr(getattr(model, "model", model), "layers")[0]
        moe = None
        for attr in ("block_sparse_moe", "mlp", "feed_forward"):
            cand = getattr(L0, attr, None)
            if cand is not None and hasattr(cand, "experts"):
                moe = attr
                break
        if moe is None:
            sys.exit("--moe: no experts found on the first block. Describe the "
                     "wiring explicitly in this file and pass it to describe().")
        gate_name = next((n for n in ("gate", "router", "gate_proj")
                          if hasattr(getattr(L0, moe), n)), None)
        wiring = Wiring(ffn_write=lambda L, m=moe: getattr(L, m),
                        router=(lambda L, m=moe, g=gate_name: getattr(getattr(L, m), g))
                        if gate_name else None,
                        experts=lambda L, m=moe: getattr(L, m).experts,
                        notes="wiring found at %r, router %r" % (moe, gate_name))
    arch = describe(model, wiring=wiring)
    print()
    print("[architecture]")
    for line in arch.summary():
        print("  " + line)

    # ---------------- extraction, one pass ----------------
    print()
    print("[extract] one forward pass per batch, block %d hooked" % peak)
    H_res, H_att, H_ffn, H_head, gates, routes = [], [], [], [], [], []
    # the law needs contributions from a block the frame does not contain, so
    # the NEXT block is captured too, in the same pass
    H_next_att, H_next_ffn, H_pre = [], [], []
    # no_grad is not an optimisation here, it is a requirement: without it every
    # pass keeps its autograd graph alive and the memory grows until the device
    # runs out. It is also what silences the requires_grad warning from the hooks.
    #
    # The lm_head is skipped: Gemma-2 has a 256k vocabulary and computing logits
    # for a batch costs more memory than everything measured here, and none of
    # it is used. Only hidden states and block contributions are needed.
    trim = _skip_lm_head(model)
    nxt = min(peak + 1, arch.n_blocks - 1)
    with torch.no_grad(), BlockCapture(model, arch, peak) as cap, \
            BlockCapture(model, arch, nxt) as cap2:
        for s in range(0, len(items), a.batch):
            enc = tok(items[s:s + a.batch], return_tensors="pt",
                      padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            H_res.append(hs[peak + 1][:, -1, :].float().cpu())
            H_pre.append(hs[peak][:, -1, :].float().cpu())
            att, ffn = cap.attn(), cap.ffn()
            H_att.append(att)
            H_ffn.append(ffn)
            H_head.append(cap.heads())
            H_next_att.append(cap2.attn())
            H_next_ffn.append(cap2.ffn())
            gates.append((identity_gate(hs[peak][:, -1, :].float().cpu(),
                                        hs[peak + 1][:, -1, :].float().cpu(),
                                        att, ffn)[0],
                          head_gate(cap.heads(), att, cap.bias_term())[0]))
            if a.moe:
                r = cap.routing()
                if r is not None:
                    routes.append(r)
            print("\r  %d/%d" % (min(s + a.batch, len(items)), len(items)),
                  end="", flush=True)
    print()
    trim()
    H_res = torch.cat(H_res, 0)
    H_att = torch.cat(H_att, 0)
    H_ffn = torch.cat(H_ffn, 0)
    H_head = torch.cat(H_head, 0)
    H_next_att = torch.cat(H_next_att, 0)
    H_next_ffn = torch.cat(H_next_ffn, 0)
    H_pre = torch.cat(H_pre, 0)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    g_block = float(torch.tensor([g[0] for g in gates]).median())
    g_head = float(torch.tensor([g[1] for g in gates]).median())
    print()
    print("--- GATES -------------------------------------------------------")
    print("  block: attn + ffn against the residual delta : %.2e   %s"
          % (g_block, "OK" if g_block < 1e-5 else "FAILED"))
    print("  heads: sum of heads against the attention    : %.2e   %s"
          % (g_head, "OK" if g_head < 1e-5 else "FAILED"))
    if g_block >= 1e-4 or g_head >= 1e-4:
        sys.exit("  ABORT: the decomposition does not hold; nothing downstream "
                 "would mean anything.")

    res = dict(model=a.model, peak=peak, truthprobe_version=__version__,
               protocol=proto.to_dict(), n_pairs_flat=len(pidx_flat),
               n_pairs_cat=len(pidx_cat), cats=cats,
               gate_block=g_block, gate_head=g_head,
               architecture=arch.to_dict())

    # ---------------- the axis, against the published figure -------------
    print()
    print("--- AXIS --------------------------------------------------------")
    aucs, pas = [], []
    for tr, te in kfold_pairs(len(pidx_flat), a.folds, a.seed):
        ax = fit_axis(H_res, [pidx_flat[p] for p in tr])
        I = [i for p in te for i in (pidx_flat[p][0], pidx_flat[p][1])]
        lab = torch.tensor([1, 0] * len(te))
        aucs.append(auc_score(project_and_score(H_res[I], ax), lab))
        st = project_and_score(torch.stack([H_res[pidx_flat[p][0]] for p in te], 0), ax)
        sf = project_and_score(torch.stack([H_res[pidx_flat[p][1]] for p in te], 0), ax)
        pas.append(paired_accuracy(st, sf))
    axis_auc = sum(aucs) / len(aucs)
    paired = sum(pas) / len(pas)
    se = se_binomial(axis_auc, 2 * len(pidx_flat))
    print("  axis, held-out AUC   : %.3f   (se %.3f)%s"
          % (axis_auc, se, band(axis_auc, exp.get("axis_auc"), a.tol)))
    print("  paired accuracy      : %.3f   (same geometry, different statistic:"
          % paired)
    print("                                  the topic is controlled within a pair)")
    probe_auc = full_probe_auc(H_res, pidx_flat, a.folds, a.seed,
                               l2=a.probe_l2, steps=a.probe_steps)
    print("  full linear probe    : %.3f%s"
          % (probe_auc, band(probe_auc, exp.get("probe_auc"), a.tol)))
    print("  gap probe minus axis : %+.3f   (the information living off-axis)"
          % (probe_auc - axis_auc))
    res.update(axis_auc=axis_auc, paired=paired, probe_auc=probe_auc,
               gap=probe_auc - axis_auc)

    # ---------------- propagate versus overwrite -------------------------
    print()
    print("--- COMPONENTS --------------------------------------------------")
    print("  TWO DIFFERENT MEASURES, never to be confused with one another.")
    print()
    # (a) separability of each contribution, its own axis: this is what
    #     anatomy.py reports, and the only figure comparable with its output.
    def contrib_auc(Hc):
        out = []
        for tr, te in kfold_pairs(len(pidx_flat), a.folds, a.seed):
            ax = fit_axis(Hc, [pidx_flat[p] for p in tr])
            I = [i for p in te for i in (pidx_flat[p][0], pidx_flat[p][1])]
            out.append(auc_score(project_and_score(Hc[I], ax),
                                 torch.tensor([1, 0] * len(te))))
        return sum(out) / len(out)

    auc_att, auc_ffn = contrib_auc(H_att), contrib_auc(H_ffn)
    print("  (a) SEPARABILITY of each contribution, each with its OWN axis")
    print("      residual  %.3f%s" % (axis_auc, band(axis_auc, exp.get("axis_auc"), a.tol)))
    print("      attention %.3f%s" % (auc_att, band(auc_att, exp.get("auc_attn"), a.tol)))
    print("      FFN       %.3f%s" % (auc_ffn, band(auc_ffn, exp.get("auc_ffn"), a.tol)))
    print("      comparable with anatomy.py. Says where the axis is most READABLE,")
    print("      which is not where it is necessary: a contribution can read high")
    print("      and cost nothing when removed.")
    print()
    # (b) the contribution read against a FIXED frame: this is ffn_erosion's
    #     quantity, and the one the propagate/overwrite law is stated in.
    tt = [i for i, _ in pidx_flat]
    ff = [j for _, j in pidx_flat]

    # TWO FRAMES, and the difference between them is a result, not a detail.
    #
    # POST frame: the residual at the OUTPUT of block b. It already contains what
    #   block b wrote, so f_b sits inside it and drags the axis fitted there
    #   toward itself. Under this frame the FFN reads PRO at b and ANTI at b+1.
    # PRE frame: the residual at the INPUT of block b, built from strictly
    #   earlier blocks. It contains neither f_b nor a_b, so nothing measures its
    #   own ingredient. Under this frame the FFN's "pro at its own block" term
    #   does not merely shrink, it inverts sign, and attention is its mirror.
    #
    # Comparing a post-frame number against a pre-frame expectation is a
    # category error, and the numbers are printed separately for that reason.
    v_post = unit(fit_axis(H_res, pidx_flat)["v1"])          # output of block peak
    v_pre = unit(fit_axis(H_pre, pidx_flat)["v1"])           # input of block peak

    print("  (b) d' of a contribution against a FIXED frame, never refitted")
    print()
    print("      POST frame: axis fitted on the residual at the OUTPUT of block %d"
          % peak)
    pa_b = frame_gap(H_att[tt], H_att[ff], v_post,
                     contrib_block=peak, frame_block=peak)
    pf_b = frame_gap(H_ffn[tt], H_ffn[ff], v_post,
                     contrib_block=peak, frame_block=peak)
    pa_b1 = frame_gap(H_next_att[tt], H_next_att[ff], v_post,
                      contrib_block=peak + 1, frame_block=peak)
    pf_b1 = frame_gap(H_next_ffn[tt], H_next_ffn[ff], v_post,
                      contrib_block=peak + 1, frame_block=peak)
    print("        at block %-3d  attention %+.3f   FFN %+.3f%s"
          % (peak, pa_b["dprime"], pf_b["dprime"],
             band(pf_b["dprime"], exp.get("post_ffn_b"), 0.30)))
    print("        at block %-3d  attention %+.3f   FFN %+.3f%s"
          % (peak + 1, pa_b1["dprime"], pf_b1["dprime"],
             band(pf_b1["dprime"], exp.get("post_ffn_b1"), 0.30)))
    print("        pro at b and anti at b+1 is the post-frame diagonal. The pro")
    print("        term is in substantial part the frame measuring its own")
    print("        ingredient: f_b is inside the state the axis was fitted on.")
    print()
    print("      PRE frame: axis fitted on the residual at the INPUT of block %d,"
          % peak)
    print("      which contains neither the attention nor the FFN of that block")
    ra = frame_gap(H_att[tt], H_att[ff], v_pre,
                   contrib_block=peak, frame_block=peak - 1)
    rf = frame_gap(H_ffn[tt], H_ffn[ff], v_pre,
                   contrib_block=peak, frame_block=peak - 1)
    print("        at block %-3d  attention %+.3f%s"
          % (peak, ra["dprime"], band(ra["dprime"], exp.get("pre_attn"), 0.50)))
    print("        %-12s FFN       %+.3f%s"
          % ("", rf["dprime"], band(rf["dprime"], exp.get("pre_ffn"), 0.50)))
    print("        under a clean frame the FFN inverts sign and attention is its")
    print("        mirror: that inversion is the relational law, and it is what")
    print("        the post frame was hiding.")
    print("        cosine between the two frames: %+.3f" % float(v_pre @ v_post))
    res.update(auc_attn=auc_att, auc_ffn=auc_ffn,
               post=dict(attn_b=pa_b["dprime"], ffn_b=pf_b["dprime"],
                         attn_b1=pa_b1["dprime"], ffn_b1=pf_b1["dprime"]),
               pre=dict(attn=ra["dprime"], ffn=rf["dprime"]),
               cos_pre_post=float(v_pre @ v_post))

    # ---------------- the arrangement ------------------------------------
    print()
    print("--- ARRANGEMENT -------------------------------------------------")
    axes = torch.stack([unit(fit_axis(H_res, cat_of[c])["v1"]) for c in cats], 0)
    C, _ = cosine_matrix(axes)
    s, marg, thin = consensus_gauge(C)
    eg = eigengap(C)
    off = [float(C[i, j]) for i in range(len(cats)) for j in range(len(cats)) if i != j]
    print("  off-diagonal cosine, median : %+.3f" % float(torch.tensor(off).median()))
    print("  anti-aligned cells          : %d of %d"
          % (sum(1 for x in off if x < -0.05), len(off)))
    print("  gauge margin, median        : %.3f   unsigned: %s"
          % (float(marg.median()), [cats[i] for i in thin] or "none"))
    print("  relative eigengap           : %.3f   (identifiability, a priori)" % eg["rel"])
    print("  effective rank of the axes  : %.2f out of %d"
          % (effective_rank(axes)["effective_rank"], len(cats)))
    res.update(cos_median=float(torch.tensor(off).median()),
               gauge_margin=float(marg.median()), eigengap=eg["rel"],
               unsigned=[cats[i] for i in thin])

    # ---------------- heads ----------------------------------------------
    print()
    print("--- HEADS -------------------------------------------------------")
    A = torch.stack([unit(fit_axis(H_head[:, h, :], pidx_flat)["v1"])
                     for h in range(H_head.shape[1])], 0)
    # the reference is the POST frame, the residual axis at the output of the
    # peak block: that is the axis the head contributions are compared against
    cov = float(subspace_fraction(v_post.unsqueeze(0), A)[0])
    print("  cosine of each head's axis with the residual axis (post frame):")
    print("    " + "  ".join("h%d %+.2f" % (h, float(A[h] @ v_post))
                             for h in range(A.shape[0])))
    print("  residual axis inside the head span : %.3f   (chance %.4f, ratio %.0fx)"
          % (cov, A.shape[0] / A.shape[1], cov / (A.shape[0] / A.shape[1])))
    pa_ = principal_angles(A, axes)
    print("  head span against category span: %d shared dimensions below 45 degrees "
          "(random would share %d)" % (pa_["shared_at_45"], pa_["random_shared_at_45"]))
    res.update(head_cosines=[float(A[h] @ v_post) for h in range(A.shape[0])],
               head_coverage=cov, shared_dims=pa_["shared_at_45"])

    # ---------------- mixture of experts ---------------------------------
    if a.moe:
        print()
        print("--- MIXTURE OF EXPERTS ------------------------------------------")
        if not routes:
            print("  the router was not captured: check the wiring.")
        else:
            ex = torch.cat([r["experts"] for r in routes], 0)
            wt = torch.cat([r["weights"] for r in routes], 0)
            n_exp = arch.n_experts or int(ex.max()) + 1
            counts = torch.bincount(ex.reshape(-1), minlength=n_exp)
            print("  experts activated at the READ TOKEN, over %d sentences:" % ex.shape[0])
            print("    " + "  ".join("e%d %d" % (i, int(counts[i])) for i in range(n_exp)))
            print("  mean weight of the top expert : %.3f" % float(wt[:, 0].mean()))
            same = sum(1 for i, j in pidx_flat
                       if set(ex[i].tolist()) == set(ex[j].tolist()))
            print("  pairs whose two sentences activate the SAME experts: %d/%d  %.1f%%"
                  % (same, len(pidx_flat), 100 * same / len(pidx_flat)))
            print()
            print("  This last number decides whether a per-expert study is possible.")
            print("  The router routes PER TOKEN, and with the period convention the")
            print("  read token is identical in both sentences of a pair. If the")
            print("  upstream target change does not alter which experts fire at the")
            print("  period, there is nothing for a per-expert decomposition to")
            print("  separate, and that is known before fitting anything.")
            res.update(expert_counts=[int(x) for x in counts],
                       same_experts=same / len(pidx_flat))

    print()
    print("=" * 78)
    print("done in %.1f s" % (time.time() - t0))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("written: %s" % a.out)
    print()


if __name__ == "__main__":
    main()
