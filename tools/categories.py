# -*- coding: utf-8 -*-
"""
categories.py  --  is the truth axis a MIXTURE of category-specific components?

Converted to a thin wrapper over the truthprobe core. Same three registered
predictions, same surface control, same statistics; the shared machinery now
lives in the library instead of being restated here.

THE PRE-REGISTERED PREDICTIONS (written before any run, unchanged)
  P1  cosines between per-category truth axes at the peak sit well below 1 but
      above 0: a shared core plus category-specific tails.
  P2  transfer matrix: within-category held-out AUC above cross-category AUC.
  P3  the category of a pair is decodable from the FFN's class-signed write
      direction, above a permutation null.
  SURFACE CONTROL (mandatory): the structure at the peak must DIVERGE from the
      same structure at an early block. If the early block shows the same
      picture, the tails are lexical surface (targets and templates differ by
      relation), not contextual truth.

WHAT CHANGED, AND WHY IT MATTERS
  The sentence convention is now explicit. The original built sentences as
  "prompt" + raw target, while truth_probe.py built "prompt target." with a
  final period. Those two produce axes at cosine +0.52, i.e. different
  directions, while the arrangement between categories survives at Mantel
  +0.775. Nobody noticed for months, because the convention lived inside each
  file. Here it is a Protocol, it is printed at every run, and it is written
  into every artifact. Use --legacy to reproduce the original numbers.

  Only ONE block is hooked, not all of them. The original captured every
  layer's attention and FFN contributions to use two of them; this captures
  the FFN at the write layer and reads the residual from hidden_states. Same
  numbers, far less memory.

  float32 is required: the additive identity gate subtracts large residual
  states to obtain small per-block deltas, and in bfloat16 that suffers
  catastrophic cancellation.

    python categories.py --model Qwen/Qwen2.5-1.5B --peak 15
    python categories.py --model google/gemma-2-2b --peak 11 --legacy
"""

import argparse
import sys

import torch

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import counterfact_by_relation
from truthprobe.geometry import unit, fit_axis, cosine_matrix
from truthprobe.hooks import describe, BlockCapture, identity_gate
from truthprobe.stats import (auc_score, kfold_pairs, project_and_score,
                              decoding_with_null)


# =====================================================================
#  the two matrices
# =====================================================================
def axis_cosine_matrix(H, cat_pairs):
    """Full-fit axis per category, then the KxK SIGNED cosine matrix.

    Each axis is oriented true-positive within its OWN category, so the sign
    carries information: negative means the shared direction reads the other
    category's truth backwards. But the sign of a single cell depends on the
    orientation of BOTH categories involved; only products around closed
    cycles are gauge-invariant."""
    cats = sorted(cat_pairs)
    axes = {c: fit_axis(H, cat_pairs[c])["v1"] for c in cats}
    M, _ = cosine_matrix([axes[c] for c in cats])
    return cats, M, axes


def transfer_matrix(H, cat_pairs, folds, seed):
    """KxK held-out AUC. Diagonal: cross-validation WITHIN the category, the
    axis refitted on training folds only. Off-diagonal: axis fitted on ALL of
    category A and evaluated on ALL of category B, which are disjoint.

    The diagonal is the quantity used elsewhere as a representational proxy for
    knowledge, and the threshold that restricts to known relations reads it."""
    cats = sorted(cat_pairs)
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
    return cats, M


def show_matrix(cats, M, fmt="%8.2f"):
    print("       " + "".join("%8s" % c for c in cats))
    for i, c in enumerate(cats):
        print("%7s" % c + "".join(fmt % float(M[i, j]) for j in range(len(cats))))


# =====================================================================
#  extraction: one forward pass per sentence, one block hooked
# =====================================================================
@torch.no_grad()
def collect(model, tok, texts, peak, early, write_layer, arch, device, batch):
    """Residual states at two blocks, plus the FFN contribution at the write
    layer, from a single pass. Also runs the additive identity gate at the
    write layer, because if that fails nothing downstream means anything."""
    H_peak, H_early, F_write, gates = [], [], [], []
    with BlockCapture(model, arch, write_layer) as cap:
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            H_peak.append(hs[peak + 1][:, -1, :].float().cpu())
            H_early.append(hs[early + 1][:, -1, :].float().cpu())
            a, f = cap.attn(), cap.ffn()
            F_write.append(f)
            gates.append((hs[write_layer][:, -1, :].float().cpu(),
                          hs[write_layer + 1][:, -1, :].float().cpu(), a, f))
            print("\r  [extract] %d/%d" % (min(s + batch, len(texts)), len(texts)),
                  end="", flush=True)
    print()
    med = max(identity_gate(*g)[0] for g in gates)
    return (torch.cat(H_peak, 0), torch.cat(H_early, 0), torch.cat(F_write, 0), med)


# =====================================================================
#  main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--peak", type=int, default=15, help="truth-peak BLOCK")
    ap.add_argument("--early-block", type=int, default=2,
                    help="surface-control block, the lexical baseline")
    ap.add_argument("--write-layer", type=int, default=None,
                    help="FFN block for Delta f (default: peak+1)")
    ap.add_argument("--k-relations", type=int, default=5)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8,
                    help="batch size. Safe in float32: measured 2e-06 relative "
                         "error against single-sentence extraction. Use 1 for "
                         "bit-exact reproduction of the original tool.")
    ap.add_argument("--legacy", action="store_true",
                    help="use the historical dictionary-family convention "
                         "(no final period, space taken from the dataset). "
                         "Needed to reproduce bundles produced before the "
                         "convention was made explicit.")
    ap.add_argument("--file-counterfact", default=None)
    a = ap.parse_args()
    wl = a.write_layer if a.write_layer is not None else a.peak + 1

    proto = (LEGACY_DICT if a.legacy else CANONICAL).with_(seed=a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("[task] categories: is the truth axis a mixture of per-category components?")
    print("[library] truthprobe %s" % __version__)
    print("[protocol] suffix %r, join %r  ->  %r"
          % (proto.suffix, proto.join,
             proto.sentence("The capital of France is", " Paris")))
    if a.legacy:
        print("  LEGACY convention: comparable with pre-conversion bundles, "
              "NOT with truth_probe numbers")
    print("[blocks] axes at %d, surface control at %d, FFN writes at %d"
          % (a.peak, a.early_block, wl))

    ps = counterfact_by_relation(proto, k=a.k_relations, n_per=a.pairs_per_relation,
                                 local_file=a.file_counterfact)
    cats = ps.categories
    cat_pairs = ps.by_category()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, use_safetensors=True,
        trust_remote_code=False).to(device)
    model.eval()
    arch = describe(model)
    print("[model] %s on %s (float32)" % (a.model, device))
    for line in arch.summary():
        print("  " + line)

    H_peak, H_early, F_write, gate = collect(
        model, tok, ps.items, a.peak, a.early_block, wl, arch, device, a.batch)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("[identity gate] median relative error %.2e" % gate)
    if gate > 1e-3:
        sys.exit("  ABORT: the additive decomposition does not hold; "
                 "numbers downstream would be meaningless.")

    # ---------- P1: axis cosines, peak against early ----------
    for name, H in (("PEAK (block %d)" % a.peak, H_peak),
                    ("EARLY (block %d)" % a.early_block, H_early)):
        cs, M, _ = axis_cosine_matrix(H, cat_pairs)
        print()
        print("=== P1  signed cosines between per-category truth axes @ %s ===" % name)
        show_matrix(cs, M)
        off = [float(M[i, j]) for i in range(len(cs)) for j in range(len(cs)) if i != j]
        neg = [x for x in off if x < -0.05]
        print("  off-diagonal mean %+.2f   anti-aligned pairs: %d of %d"
              % (sum(off) / len(off), len(neg), len(off)))

    # ---------- P2: transfer at the peak ----------
    cs, TM = transfer_matrix(H_peak, cat_pairs, a.folds, a.seed)
    print()
    print("=== P2  held-out AUC transfer @ PEAK (row axis -> column data) ===")
    show_matrix(cs, TM, "%8.3f")
    diag = [float(TM[i, i]) for i in range(len(cs))]
    off = [float(TM[i, j]) for i in range(len(cs)) for j in range(len(cs)) if i != j]
    print("  within mean %.3f   cross mean %.3f   difference %+.3f"
          % (sum(diag) / len(diag), sum(off) / len(off),
             sum(diag) / len(diag) - sum(off) / len(off)))

    # ---------- P3: category decoding from write directions ----------
    labels = [p.category for p in ps.pairs]
    Df = torch.stack([unit(F_write[it] - F_write[iff]) for it, iff in ps.pidx], 0)
    De = torch.stack([unit(H_early[it] - H_early[iff]) for it, iff in ps.pidx], 0)
    print()
    print("=== P3  category decoding from class-signed directions "
          "(nearest centroid, %d folds, %d perms) ===" % (a.folds, a.perm))
    for name, D in ((("FFN write Delta f @ block %d" % wl), Df),
                    (("EARLY residual Delta h @ block %d (lexical baseline)"
                      % a.early_block), De)):
        r = decoding_with_null(D, labels, a.folds, a.seed, a.perm)
        print("  %s:" % name)
        print("    accuracy %.2f%%   null mean %.2f%%   null 95pct %.2f%%   "
              "p=%.4f   (chance %.2f%%)"
              % (100 * r["acc"], 100 * r["null_mean"], 100 * r["null_p95"],
                 r["p"], 100 * r["chance"]))

    print()
    print("=== reading guide (no automatic verdict; predictions were registered) ===")
    print("  P1  peak off-diagonal cosines well below 1 but above 0 means a shared")
    print("      core plus category tails. SURFACE CONTROL: if the EARLY matrix")
    print("      shows the same structure, those tails are lexical, not contextual.")
    print("  P2  within above cross at the peak means category-specific truth")
    print("      components exist.")
    print("  P3  Delta f decoding above the null means the FFN writes truth along")
    print("      category-dependent directions. But if the EARLY lexical baseline")
    print("      decodes just as well, target-token identity may explain it:")
    print("      compare the two accuracies, never read P3 alone.")
    print("  A signed cosine cell is not gauge-free: only cycle products are.")
    print()


if __name__ == "__main__":
    main()
