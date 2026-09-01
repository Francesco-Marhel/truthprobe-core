# -*- coding: utf-8 -*-
"""
ablazione_teste.py  --  which heads are causally NECESSARY, not merely readable.

Converted to a thin wrapper over the truthprobe core.

WHY THIS TOOL EXISTS
The readability tables say what a head writes. They do not say whether it is
needed. On Gemma-2-2b-it the attention contribution at the peak reads 0.900,
higher than the residual itself, and removing it entirely costs 0.005: readable
and not necessary. That dissociation is only visible by intervening.

HOW A HEAD IS SWITCHED OFF
Inside a block the heads are concatenated and passed through o_proj. Zeroing a
head's slice of the INPUT to o_proj removes its contribution from the sum,
because o_proj is linear and, on these models, has no bias. The head still
computes, its output is discarded, the weights are untouched. Same form of
intervention as the whole-block ablations, applied to a slice.

On sandwich-norm architectures the post-norm sits after o_proj, so removing one
head also changes the normalisation factor of the others. That is the real
effect of the intervention and is reported as such: an ablation is not a linear
subtraction downstream.

QK OR OV
Two heads can oppose each other for two different reasons: they look at
different tokens (routing, that is QK) or they write oppositely from the same
tokens (OV). The captured contribution already contains both, so on its own it
does not attribute. The pattern section compares the attention distributions at
the read token: same tokens means the opposition lives in OV, different tokens
means it lives in the routing.

FOR MIXTURES OF EXPERTS
--experts ablates one expert at a time instead of one head, using the Wiring
description. The block stays additive either way, so the readout is unchanged;
what changes is which slice is silenced. Note that the router routes PER TOKEN:
with the period convention the read token is identical in both sentences of a
pair, so the question is whether the upstream target change alters which
experts fire at the period.

TWO STATISTICS, NEVER INTERCHANGEABLE
  paired   compares the two sentences of the SAME pair, so the topic is
           controlled and the value sits near the ceiling, compressing deltas
  auc      ROC on single sentences, the statistic of the canonical protocol
  auc_fix  the same with the axis FIXED from the intact condition, so a mere
           rotation caused by the ablation is not recovered by refitting

No verdicts are printed. Every delta comes with its standard error.

    python ablazione_teste.py --model google/gemma-2-2b --peak 11
    python ablazione_teste.py --model google/gemma-2-2b --peak 11 \\
           --band-start 12 --band-end 14 --readout 14 --head-pairs 4,5
    python ablazione_teste.py --model ... --peak 11 --full --pairs-per-relation 10
"""

import argparse
import json
import math
import os
import sys

import torch

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import counterfact_by_relation
from truthprobe.geometry import unit, fit_axis
from truthprobe.hooks import describe, Wiring
from truthprobe.stats import auc_score, kfold_pairs, paired_accuracy, se_binomial


# =====================================================================
#  interventions
# =====================================================================
def _layers(model):
    inner = getattr(model, "model", model)
    return getattr(inner, "layers")


def slice_hook(ids, width):
    """Zero the given slices of a module's INPUT."""
    def pre(_m, inputs):
        z = inputs[0].clone()
        for h in ids:
            z[..., h * width:(h + 1) * width] = 0.0
        return (z,) + tuple(inputs[1:])
    return pre


def zero_hook():
    """Zero a module's output. On sandwich architectures the vector then goes
    through the post-norm, and RMSNorm of the zero vector is zero, so the
    ablation stays exact."""
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)
    return hook


def expert_hook(ids):
    """Zero the output of the given experts. The router still routes to them,
    their weight is still spent, and their contribution is discarded: that is
    the intervention, and it is not the same as re-routing."""
    return zero_hook()


def install(model, actions, arch, wiring=None):
    """actions: list of (block, kind, ids). kind in {attn, ffn, expert}."""
    handles = []
    for b, kind, ids in actions:
        L = _layers(model)[b]
        if kind == "ffn":
            mod = wiring.ffn_write(L) if (wiring and wiring.ffn_write) else L.mlp
            handles.append(mod.register_forward_hook(zero_hook()))
        elif kind == "expert":
            experts = wiring.experts(L)
            for e in ids:
                handles.append(experts[e].register_forward_hook(zero_hook()))
        else:
            attn = getattr(L, "self_attn", None) or getattr(L, "attn")
            op = getattr(attn, "o_proj", None) or getattr(attn, "out_proj")
            handles.append(op.register_forward_pre_hook(
                slice_hook(ids, arch.head_dim)))
    return handles


@torch.no_grad()
def read_states(model, tok, texts, level, device, batch):
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states[level]
        out.append(hs[:, -1, :].float().cpu())
        print("\r    %d/%d" % (min(s + batch, len(texts)), len(texts)),
              end="", flush=True)
    print()
    return torch.cat(out, 0)


@torch.no_grad()
def attention_patterns(model, tok, texts, block, device, batch, n_sample, seed=0):
    """Cosines between the heads' attention distributions at the read token,
    averaged over sentences. Answers whether two heads look at the same tokens."""
    import random
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_sample]
    S, count = None, 0
    for s in range(0, len(idx), batch):
        chunk = [texts[i] for i in idx[s:s + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        att = model(**enc, output_attentions=True).attentions[block]
        last = att[:, :, -1, :].float() * enc["attention_mask"].unsqueeze(1).float()
        last = last / last.sum(-1, keepdim=True).clamp_min(1e-9)
        u = last / last.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        M = torch.einsum("bhs,bks->bhk", u, u).sum(0).cpu()
        S = M if S is None else S + M
        count += last.shape[0]
    return S / max(count, 1)


# =====================================================================
#  evaluation
# =====================================================================
def evaluate(H, pidx, folds, seed, subset=None, v_fixed=None):
    """paired, auc and auc with a fixed axis, on the same held-out folds."""
    keep = set(range(len(pidx))) if subset is None else set(subset)
    pa, au, af = [], [], []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        D = torch.stack([H[pidx[p][0]] - H[pidx[p][1]] for p in tr], 0)
        _, _, Vh = torch.linalg.svd(D, full_matrices=False)
        v = unit(Vh[0])
        if float((D @ v).sum()) < 0:
            v = -v
        sel = [p for p in te if p in keep]
        if not sel:
            continue
        st = torch.stack([H[pidx[p][0]] for p in sel], 0) @ v
        sf = torch.stack([H[pidx[p][1]] for p in sel], 0) @ v
        pa.append(paired_accuracy(st, sf))
        lab = torch.tensor([1, 0] * len(sel))
        idx = [i for p in sel for i in (pidx[p][0], pidx[p][1])]
        au.append(auc_score(H[idx] @ v, lab))
        if v_fixed is not None:
            af.append(auc_score(H[idx] @ v_fixed, lab))
    m = lambda x: (sum(x) / len(x)) if x else float("nan")
    return dict(paired=m(pa), auc=m(au), auc_fix=m(af))


# =====================================================================
#  main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--peak", type=int, required=True, help="BLOCK to ablate in")
    ap.add_argument("--readout", type=int, default=None, help="default: peak+3")
    ap.add_argument("--band-start", type=int, default=None, help="default: the peak")
    ap.add_argument("--band-end", type=int, default=None, help="default: the peak")
    ap.add_argument("--heads", type=int, nargs="*", default=None,
                    help="which heads to ablate one at a time (default: all)")
    ap.add_argument("--head-pairs", nargs="*", default=None,
                    help="joint ablations, e.g. --head-pairs 4,5 0,1")
    ap.add_argument("--experts", action="store_true",
                    help="ablate experts instead of heads (mixture of experts)")
    ap.add_argument("--k-relations", type=int, default=33)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--attn-sample", type=int, default=400,
                    help="sentences used for the attention-pattern comparison; "
                         "0 skips it")
    ap.add_argument("--no-ffn", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="scan EVERY block: attn-off and ffn-off, read at the block "
                         "itself. Costs 2 x n_blocks runs, so use few pairs.")
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="bfloat16 is safe here: measured deltas agree with float32 "
                         "to 0.002, a quarter of the sampling standard error. "
                         "float32 remains required for the identity gate.")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    readout = a.readout if a.readout is not None else a.peak + 3
    proto = (CANONICAL if a.suffix == "." else LEGACY_DICT).with_(seed=a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("[library] truthprobe %s" % __version__)
    ps = counterfact_by_relation(proto, k=a.k_relations, n_per=a.pairs_per_relation,
                                 local_file=a.file_counterfact)
    labels = [p.category for p in ps.pairs]
    cats = ps.categories
    print("[bands] ablate at block %d, read at block %d" % (a.peak, readout))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=(torch.bfloat16 if a.dtype == "bfloat16" else torch.float32),
        use_safetensors=True, trust_remote_code=False,
        attn_implementation="eager").to(device)
    model.eval()

    wiring = None
    if a.experts:
        L0 = _layers(model)[0]
        moe = getattr(L0, "block_sparse_moe", None) or getattr(L0, "mlp", None)
        if moe is None or not hasattr(moe, "experts"):
            sys.exit("--experts: no experts found. Describe the wiring explicitly "
                     "and pass it, or check the architecture.")
        wiring = Wiring(ffn_write=lambda L: getattr(L, "block_sparse_moe", L.mlp),
                        experts=lambda L: getattr(L, "block_sparse_moe", L.mlp).experts)
    arch = describe(model, wiring=wiring)
    print("[model] %s on %s (%s)" % (a.model, device, a.dtype))
    for line in arch.summary():
        print("  " + line)

    nB = arch.n_blocks
    band = list(range(a.band_start if a.band_start is not None else a.peak,
                      (a.band_end if a.band_end is not None else a.peak) + 1))
    if a.experts:
        n_units = len(wiring.experts(_layers(model)[a.peak]))
        kind, word = "expert", "expert"
    else:
        n_units = arch.n_heads
        kind, word = "attn", "head"
    units = a.heads if a.heads is not None else list(range(n_units))
    all_u = list(range(n_units))

    # ---- attention patterns, one pass ----
    if a.attn_sample and not a.experts:
        print()
        print("[patterns] attention distributions at the read token")
        S = attention_patterns(model, tok, ps.items, a.peak, device,
                               a.batch, a.attn_sample, a.seed)
        print("  " + " ".join("%6d" % h for h in range(n_units)))
        for i in range(n_units):
            print("  h%-2d " % i + " ".join("%+6.2f" % S[i, j] for j in range(n_units)))
        print("  (two heads with a high cosine look at the SAME tokens: an")
        print("   opposition between them then lives in OV, not in the routing)")

    # ---- configurations ----
    if a.full:
        configs = [("intact", [], None)]
        for b in range(nB):
            configs.append(("L%d-attn" % b, [(b, "attn", all_u)], b))
            if not a.no_ffn:
                configs.append(("L%d-ffn" % b, [(b, "ffn", None)], b))
    else:
        configs = [("intact", [], None)]
        for u in units:
            configs.append(("%s%d" % (word[0], u), [(b, kind, [u]) for b in band], None))
        for spec in (a.head_pairs or []):
            hs = [int(x) for x in spec.replace(" ", "").split(",") if x]
            configs.append((word[0] + "+".join(map(str, hs)),
                            [(b, kind, hs) for b in band], None))
        configs.append(("attn-off", [(b, "attn", list(range(arch.n_heads)))
                                     for b in band], None))
        if not a.no_ffn:
            configs.append(("ffn-off", [(b, "ffn", None) for b in band], None))
            configs.append(("attn+ffn", [(b, k, list(range(arch.n_heads)) if k == "attn" else None)
                                         for b in band for k in ("attn", "ffn")], None))

    print()
    print("[band] blocks %s" % (band if not a.full else "all (scan)"))
    print("[cost] %d configurations x %d sentences = %d passes"
          % (len(configs), len(ps.items), len(configs) * len(ps.items)))

    # ---- run ----
    res, percat, v_intact = {}, {}, None
    for name, actions, own in configs:
        lv = (own + 1) if own is not None else (readout + 1)
        print()
        print("[run] %-10s read at block %d" % (name, lv - 1))
        handles = install(model, actions, arch, wiring) if actions else []
        try:
            H = read_states(model, tok, ps.items, lv, device, a.batch)
        finally:
            for h in handles:
                h.remove()
        if name == "intact" and not a.full:
            D = torch.stack([H[i] - H[j] for i, j in ps.pidx], 0)
            _, _, Vh = torch.linalg.svd(D, full_matrices=False)
            v_intact = unit(Vh[0])
            if float((D @ v_intact).sum()) < 0:
                v_intact = -v_intact
        res[name] = evaluate(H, ps.pidx, a.folds, a.seed, v_fixed=v_intact)
        if not a.full:
            percat[name] = {c: evaluate(H, ps.pidx, a.folds, a.seed,
                                        subset=[i for i in range(len(ps.pidx))
                                                if labels[i] == c],
                                        v_fixed=v_intact) for c in cats}
        print("   paired %.3f   auc %.3f   auc fixed %.3f"
              % (res[name]["paired"], res[name]["auc"], res[name]["auc_fix"]))

    # ---- report ----
    b_pa, b_au = res["intact"]["paired"], res["intact"]["auc"]
    se_pa = se_binomial(b_pa, len(ps.pidx))
    se_au = se_binomial(b_au, 2 * len(ps.pidx))
    print()
    print("================  ABLATION  ================")
    print("paired: same-pair comparison, topic controlled, near the ceiling")
    print("auc:    ROC on single sentences, the canonical statistic")
    print("auc fixed: same axis as intact, no refit, so a rotation is not recovered")
    print()
    print("standard error: paired %.4f   auc %.4f" % (se_pa, se_au))
    print()
    if a.full:
        print("%-6s %9s %9s | %9s %9s" % ("block", "attn auc", "delta", "ffn auc", "delta"))
        print("-" * 50)
        for b in range(nB):
            ra = res.get("L%d-attn" % b, {}).get("auc", float("nan"))
            rf = res.get("L%d-ffn" % b, {}).get("auc", float("nan"))
            print("%-6d %9.3f %+9.3f | %9.3f %+9.3f" % (b, ra, ra - b_au, rf, rf - b_au))
    else:
        print("%-10s | %8s %8s | %8s %8s %7s | %9s %8s"
              % ("config", "paired", "delta", "auc", "delta", "sigma", "auc fixed", "delta"))
        print("-" * 80)
        for name, _, _ in configs:
            r = res[name]
            print("%-10s | %8.3f %+8.3f | %8.3f %+8.3f %7.1f | %9.3f %+8.3f"
                  % (name, r["paired"], r["paired"] - b_pa,
                     r["auc"], r["auc"] - b_au, abs(r["auc"] - b_au) / se_au,
                     r["auc_fix"], r["auc_fix"] - res["intact"]["auc_fix"]))

        se_c = se_binomial(b_au, 2 * a.pairs_per_relation)
        print()
        print("the 3 categories most damaged by each configuration (auc)")
        print("(standard error per category %.3f; %d cells examined)"
              % (se_c, (len(configs) - 1) * len(cats)))
        for name, _, _ in configs[1:]:
            dd = {c: percat[name][c]["auc"] - percat["intact"][c]["auc"] for c in cats}
            top = sorted(cats, key=lambda c: dd[c])[:3]
            print("   %-9s %s" % (name, ", ".join("%s %+.3f" % (c, dd[c]) for c in top)))

    tag = a.model.split("/")[-1].replace("/", "_")
    sfx = "full" if a.full else ("L%d" % band[0] if len(band) == 1
                                 else "L%d_%d" % (band[0], band[-1]))
    out = a.out or ("ablazione_%s_%s_%s.json" % (tag, sfx, a.dtype))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__, model=a.model,
                       protocol=proto.to_dict(), peak_block=a.peak,
                       readout_block=readout, band=band, unit=word,
                       n_units=n_units, full=bool(a.full), dtype=a.dtype,
                       cats=cats, n_pairs=len(ps.pidx),
                       results=res, per_category=percat),
                  fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % out)
    print()


if __name__ == "__main__":
    main()
