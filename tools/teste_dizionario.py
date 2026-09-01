# -*- coding: utf-8 -*-
"""
teste_dizionario.py  --  one dictionary per attention head, in a single pass.

Converted to a thin wrapper over the truthprobe core. Same bundles, same gates;
the shared machinery now lives in the library.

THE IDEA IN ONE LINE
The attention contribution to the residual stream is not a monolith: it is
already the sum of the individual heads. So no extra pass is needed per head,
only not throwing away the decomposition the model computes anyway.

WHAT THE CONVERSION BOUGHT
  The decomposition, the identity gates and the sandwich-norm handling come
  from truthprobe.hooks, which also carries the o_proj bias term. The original
  ignored the bias: on Gemma, Llama and Qwen2 there is none, so it worked, but
  on GPT-2 and others the per-head sum would silently miss it. The gate now
  catches that instead of exporting wrong numbers.

  Wiring means an architecture the library has never seen can be described from
  outside, in three lines, without touching the library.

  The sentence convention is a Protocol, printed at every run and written into
  every bundle.

WHAT THIS TOOL ANSWERS
  Whether the near-orthogonality between the attention-fitted and the
  residual-fitted global axis comes from one head or from the pack. The column
  to read is the cosine of each head's global axis with the residual one.

    python teste_dizionario.py --model google/gemma-2-2b --peak 11
    python teste_dizionario.py --model google/gemma-2-2b --peak 11 --transfer tutte
"""

import argparse
import json
import os
import sys

import torch

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import counterfact_by_relation
from truthprobe.geometry import unit, fit_axis, cosine_matrix
from truthprobe.hooks import (describe, BlockCapture, identity_gate, head_gate)
from truthprobe.stats import (auc_score, kfold_pairs, project_and_score,
                              consensus_gauge)


# =====================================================================
#  extraction: residual and every head, in one pass
# =====================================================================
@torch.no_grad()
def extract(model, tok, texts, peak, arch, device, batch=8, max_heads=None):
    """Returns the residual state at the peak block, the per-head contributions,
    and the two gate values.

    The block gate checks attention plus FFN against the residual delta. The
    head gate checks that the per-head contributions sum back to the captured
    attention contribution, bias included. Both must hold: the first says the
    block decomposition is real, the second that the head decomposition is."""
    H_res, H_head, g_block, g_head = [], [], [], []
    with BlockCapture(model, arch, peak) as cap:
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            H_res.append(hs[peak + 1][:, -1, :].float().cpu())
            a, f = cap.attn(), cap.ffn()
            per = cap.heads()
            if max_heads:
                per = per[:, :max_heads, :]
            H_head.append(per)
            g_block.append(identity_gate(hs[peak][:, -1, :].float().cpu(),
                                         hs[peak + 1][:, -1, :].float().cpu(), a, f)[0])
            g_head.append(head_gate(cap.heads(), a, cap.bias_term())[0])
            print("\r  [extract] %d/%d" % (min(s + batch, len(texts)), len(texts)),
                  end="", flush=True)
    print()
    med = lambda xs: float(torch.tensor(xs).median())
    return torch.cat(H_res, 0), torch.cat(H_head, 0), med(g_block), med(g_head)


# =====================================================================
#  one dictionary from one stream
# =====================================================================
def build_dictionary(H, ps, cats):
    """Global axis, K per-category axes oriented within their own category, and
    the signed cosine matrix."""
    cat_pairs = ps.by_category()
    t_global = unit(fit_axis(H, ps.pidx)["v1"])
    axes = torch.stack([unit(fit_axis(H, cat_pairs[c])["v1"]) for c in cats], 0)
    C, _ = cosine_matrix(axes)
    return t_global, axes, C


def transfer_matrix(H, cat_pairs, cats, folds=5, seed=0):
    """Canonical definition, identical to categories.py and crea_dizionario.py:
    the diagonal is held-out AUC WITHIN the category with the axis refitted on
    training folds, the off-diagonal is the axis of A read on all of B.

    The diagonal is the representational proxy for knowledge that the
    restriction threshold reads."""
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        pa = cat_pairs[a]
        aucs = []
        for tr, te in kfold_pairs(len(pa), folds, seed):
            ax = fit_axis(H, [pa[k] for k in tr])
            I, Y = [], []
            for k in te:
                it, iff = pa[k]
                I += [it, iff]
                Y += [1, 0]
            aucs.append(auc_score(project_and_score(H[I], ax), torch.tensor(Y)))
        M[i, i] = sum(aucs) / len(aucs)
        ax_full = fit_axis(H, pa)
        for j, b in enumerate(cats):
            if i == j:
                continue
            I, Y = [], []
            for it, iff in cat_pairs[b]:
                I += [it, iff]
                Y += [1, 0]
            M[i, j] = auc_score(project_and_score(H[I], ax_full), torch.tensor(Y))
        print("\r  [transfer] %d/%d" % (i + 1, K), end="", flush=True)
    print()
    return M


# =====================================================================
#  main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--peak", type=int, required=True, help="peak BLOCK")
    ap.add_argument("--k-relations", type=int, default=33)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-heads", type=int, default=None,
                    help="limit the number of heads, for memory")
    ap.add_argument("--transfer", default="resid", choices=["no", "resid", "tutte"],
                    help="compute the transfer matrix: never, for the residual "
                         "only (default), or for every head as well")
    ap.add_argument("--suffix", default=".",
                    help="what is appended after the target. '.' makes the LAST "
                         "token identical in the two sentences of a pair, so token "
                         "identity leaves the measurement. The two conventions are "
                         "NOT comparable and the filename records which was used.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="the identity gate needs float32: in bfloat16 it fails "
                         "by catastrophic cancellation")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--out-dir", default="teste")
    a = ap.parse_args()

    proto = (CANONICAL if a.suffix == "." else LEGACY_DICT).with_(seed=a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("[library] truthprobe %s" % __version__)
    ps = counterfact_by_relation(proto, k=a.k_relations, n_per=a.pairs_per_relation,
                                 local_file=a.file_counterfact)
    cats = ps.categories
    print("[convention] suffix %r   example: %r" % (proto.suffix, ps.items[0]))
    print("             last token is %s in the two sentences of a pair"
          % ("IDENTICAL" if proto.suffix else "DIFFERENT"))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=(torch.float32 if a.dtype == "float32" else torch.bfloat16),
        use_safetensors=True, trust_remote_code=False).to(device)
    model.eval()
    arch = describe(model)
    print("[model] %s on %s (%s)   peak block %d" % (a.model, device, a.dtype, a.peak))
    for line in arch.summary():
        print("  " + line)

    H_res, H_head, g_block, g_head = extract(model, tok, ps.items, a.peak, arch,
                                             device, a.batch, a.max_heads)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    nH = H_head.shape[1]

    print()
    print("[gates] block: attn + ffn against the residual delta = %.2e   %s"
          % (g_block, "OK" if g_block < 1e-4 else "FAILED"))
    print("        heads: sum of heads against the attention contribution = %.2e   %s"
          % (g_head, "OK" if g_head < 1e-4 else "FAILED"))
    if g_block >= 1e-4 or g_head >= 1e-4:
        sys.exit("  ABORT: the decomposition does not hold; numbers downstream "
                 "would be meaningless. In bfloat16 this is expected; on a "
                 "sandwich-norm model check the hook target.")

    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.model.split("/")[-1].replace(".", "").replace("-", "_")
    base = "K%d_n%d_s%d_L%d%s" % (a.k_relations, a.pairs_per_relation, a.seed,
                                  a.peak, "" if proto.suffix == "." else "_nodot")
    cat_pairs = ps.by_category()

    def save(name, tg, axes, C, H=None, extra=None):
        s, marg, thin = consensus_gauge(C)
        T = None
        if H is not None and (a.transfer == "tutte"
                              or (a.transfer == "resid" and name == "resid")):
            T = transfer_matrix(H, cat_pairs, cats, a.folds, a.seed)
        b = dict(cats=cats, axes=axes, t_global=tg, cos_peak=C, transfer=T,
                 gauge_signs=s, gauge_margins=marg,
                 meta=dict(model=a.model, peak_block=a.peak, view=name,
                           protocol=proto.to_dict(), truthprobe_version=__version__,
                           k_relations=len(cats),
                           pairs_per_relation=a.pairs_per_relation, seed=a.seed,
                           architecture=arch.to_dict(),
                           gate_block=g_block, gate_heads=g_head,
                           **(extra or {})))
        path = os.path.join(a.out_dir, "dict_%s_%s_%s.pt" % (tag, base, name))
        torch.save(b, path)
        return path, marg, thin

    tg_res, ax_res, C_res = build_dictionary(H_res, ps, cats)
    p, m, thin = save("resid", tg_res, ax_res, C_res, H=H_res)
    print()
    print("[residual] saved %s" % p)
    print("  median gauge margin %.3f   unsigned categories: %s"
          % (float(m.median()), [cats[i] for i in thin] or "none"))
    if a.transfer != "no":
        bb = torch.load(p, map_location="cpu")
        w = torch.tensor([bb["transfer"][i, i] for i in range(len(cats))])
        order = sorted(range(len(cats)), key=lambda i: -w[i])
        print("  within-category AUC: median %.3f   above 0.6: %d/%d"
              % (float(w.median()), int((w >= .6).sum()), len(cats)))
        print("    highest: %s" % ", ".join("%s %.2f" % (cats[i], w[i]) for i in order[:5]))
        print("    lowest : %s" % ", ".join("%s %.2f" % (cats[i], w[i]) for i in order[-5:]))

    print()
    print("[heads] cosine of each head's global axis with the residual one")
    print("  %5s %10s %10s %10s" % ("head", "cos resid", "median norm", "gauge med"))
    print("  " + "-" * 40)
    rows = []
    for h in range(nH):
        tg, axes, C = build_dictionary(H_head[:, h, :], ps, cats)
        cosr = float(tg @ tg_res)
        nrm = float(H_head[:, h, :].norm(dim=1).median())
        _, mh, th = save("head%02d" % h, tg, axes, C, H=H_head[:, h, :],
                         extra=dict(head=h, cos_global_vs_resid=cosr))
        print("  %5d %+10.3f %10.2f %10.3f" % (h, cosr, nrm, float(mh.median())))
        rows.append(dict(head=h, cos_global_vs_resid=cosr, norm_median=nrm,
                         gauge_margin_median=float(mh.median()),
                         unsigned=[cats[i] for i in th]))

    summ = os.path.join(a.out_dir, "summary_%s_%s.json" % (tag, base))
    with open(summ, "w", encoding="utf-8") as fh:
        json.dump(dict(model=a.model, peak_block=a.peak, cats=cats, n_heads=nH,
                       protocol=proto.to_dict(), truthprobe_version=__version__,
                       gate_block=g_block, gate_heads=g_head, heads=rows),
                  fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % summ)
    print()


if __name__ == "__main__":
    main()
