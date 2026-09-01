#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scansione_picco.py

WHERE does the truth axis peak, on a model the campaign has never seen.

Every other tool in this family takes the peak block as an input. On the four
models of the paper the peak is in the MODELS registry, measured once. On a new
model there is nothing to look up, and the registry says it in words:

    "The peak is MODEL-SPECIFIC and must be MEASURED with a per-layer signal
     scan, never guessed, before a model is added here."

This is that scan. It answers one question and reports two things.

THE QUESTION.  Following the paper, the peak p is the block whose intact
residual maximises held-out truth AUC.

THE TRAP, AND WHY THERE ARE TWO NUMBERS.  If the peak is chosen by looking at
the held-out AUC, and then the held-out AUC at that block is reported, the block
was chosen on the test data and the number is optimistic. The paper is explicit
that the choice of peak layer, like the axis and the calibration, must be fit on
training folds only. So this tool reports:

  1. the DESCRIPTIVE curve, held-out AUC at every block, averaged over folds and
     seeds. This is the picture: where the signal rises, where it falls. It is
     not a number to quote for the peak.
  2. the HONEST estimate: inside each fold the peak is picked on the training
     pairs alone, and the AUC is then read on that fold's held-out pairs. The
     block may differ from fold to fold, and when it does that is itself the
     result: the peak is not sharply identified.

NO VERDICT.  The tool prints the curve, the two numbers, the per-block
alternation between attention and FFN, and a proposed line for the MODELS
registry. It does not add the model to anything. The reading belongs to you.

CONTROL AT LEVEL 0.  With the period convention the last token is identical in
the two sentences of a pair, so at the embedding level there is nothing to
separate and the curve must sit at chance. That row is printed first. If it is
not near 0.500 the convention is not doing what it is supposed to do, and no
number below it means anything.

    python scansione_picco.py --model <path> --file-counterfact counterfact.parquet
    python scansione_picco.py --model <path> --max-pairs 400 --seeds 3   # faster

Companion to arXiv:2607.16741. License CC BY 4.0.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch

from truthprobe import CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import counterfact_by_relation, counterfact_flat
from truthprobe.geometry import fit_axis
from truthprobe.hooks import describe, BlockCapture
from truthprobe.stats import auc_score, kfold_pairs, project_and_score
from truthprobe.bundle import save as save_bundle


# =====================================================================
#  the core, with no model in it
# =====================================================================
def auc_at_level(H, pidx, train, test, seed_unused=None):
    """Fit the axis on the training pairs, read AUC on the held-out ones.

    Both index into the SAME row matrix: a pair is a couple of rows, and the
    fold split is over pairs, never over rows, so the shared topic of a pair
    cannot cross the split.
    """
    ax = fit_axis(H, [pidx[i] for i in train])
    rows, y = [], []
    for i in test:
        it, iff = pidx[i]
        rows += [it, iff]
        y += [1, 0]
    return float(auc_score(project_and_score(H[rows], ax), torch.tensor(y)))


def scan_stream(H_by_level, pidx, folds=5, seeds=(0, 1, 2, 3, 4)):
    """Held-out AUC at every level, for one stream.

    H_by_level  [N, L, d]: level index first after the batch.
    Returns two lists of length L: the mean over folds and seeds, and the
    standard deviation ACROSS SEEDS of the per-seed mean. The second is what the
    alternation test compares against, so it has to be the seed-level spread,
    not the fold-level one.
    """
    n_lev = H_by_level.shape[1]
    per_seed = [[] for _ in range(n_lev)]
    for sd in seeds:
        splits = list(kfold_pairs(len(pidx), folds, seed=sd))
        for lev in range(n_lev):
            H = H_by_level[:, lev, :]
            vals = [auc_at_level(H, pidx, tr, te) for tr, te in splits]
            per_seed[lev].append(sum(vals) / len(vals))
    mean = [sum(v) / len(v) for v in per_seed]
    sd = [statistics.pstdev(v) if len(v) > 1 else 0.0 for v in per_seed]
    return mean, sd, per_seed


def nested_peak(H_resid, pidx, folds=5, seeds=(0, 1, 2, 3, 4), skip_level0=True):
    """The peak chosen on training folds only, evaluated on held-out folds.

    Inside each fold the axis is fit on the training pairs at every block and
    scored on those same training pairs; the block that wins there is the
    candidate. Only then is the held-out fold read, once, at that block. The
    test pairs never take part in the choice.

    Returns the held-out AUC (mean over folds and seeds), the list of blocks
    chosen, and how often each was chosen.
    """
    n_lev = H_resid.shape[1]
    first = 1 if skip_level0 else 0
    aucs, chosen = [], []
    for sd in seeds:
        for tr, te in kfold_pairs(len(pidx), folds, seed=sd):
            best_lev, best_val = None, -1.0
            for lev in range(first, n_lev):
                H = H_resid[:, lev, :]
                v = auc_at_level(H, pidx, tr, tr)          # training folds only
                if v > best_val:
                    best_lev, best_val = lev, v
            chosen.append(best_lev)
            aucs.append(auc_at_level(H_resid[:, best_lev, :], pidx, tr, te))
    counts = {}
    for c in chosen:
        counts[c] = counts.get(c, 0) + 1
    return sum(aucs) / len(aucs), chosen, counts


def alternation(mean_a, sd_a, mean_f, sd_f, floor=0.03):
    """Which stream is more readable at each block, when the gap is real.

    The paper's rule, kept literally: a difference counts only if it exceeds
    both twice its own across-seed spread and a floor of 0.03. Everything else
    is reported as a tie, not as a small effect.
    """
    out = []
    for b in range(len(mean_a)):
        d = mean_a[b] - mean_f[b]
        s = (sd_a[b] ** 2 + sd_f[b] ** 2) ** 0.5
        real = abs(d) > max(2.0 * s, floor)
        out.append((d, s, ("attn" if d > 0 else "ffn") if real else "tie"))
    return out


# =====================================================================
#  extraction
# =====================================================================
@torch.no_grad()
def collect(model, tok, texts, arch, device, batch):
    """One forward pass per batch, every block captured.

    Returns the residual stack [N, L+1, d] and the attention and FFN
    contributions [N, L, d]. Hooking every block is what makes the scan
    possible in a single pass; it is also what makes it expensive in memory,
    which is why --max-pairs exists.
    """
    caps = [BlockCapture(model, arch, b) for b in range(arch.n_blocks)]
    for c in caps:
        c.__enter__()
    R, A, F = [], [], []
    t0 = time.time()
    try:
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            R.append(torch.stack([h[:, -1, :].float().cpu()
                                  for h in out.hidden_states], 1))
            A.append(torch.stack([c.attn() for c in caps], 1))
            F.append(torch.stack([c.ffn() for c in caps], 1))
            print("\r  [extract] %d/%d" % (min(s + batch, len(texts)), len(texts)),
                  end="", flush=True)
        print("   (%.1f s)" % (time.time() - t0))
    finally:
        for c in caps:
            c.__exit__(None, None, None)
    return torch.cat(R, 0), torch.cat(A, 0), torch.cat(F, 0)


def identity_over_blocks(H_resid, H_attn, H_ffn, tol=1e-3):
    """h_{b+1} = h_b + a_b + f_b, median over all (sentence, block).

    The median, not the maximum: where a block adds almost nothing the delta is
    near zero and the relative error explodes even on an exact reconstruction.
    A gate that fails stops the run; nothing below it would mean anything.
    """
    nB = H_attn.shape[1]
    delta = H_resid[:, 1:nB + 1, :] - H_resid[:, 0:nB, :]
    rel = ((H_attn + H_ffn - delta).norm(dim=-1)
           / delta.norm(dim=-1).clamp_min(1e-8))
    return float(rel.median()), float(rel.median()) < tol


# =====================================================================
#  report
# =====================================================================
def bar(v, lo=0.45, hi=0.95, width=34):
    n = int(round(max(0.0, min(1.0, (v - lo) / (hi - lo))) * width))
    return "#" * n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sampling", default="relation", choices=["relation", "flat"],
                    help="grouped by relation (the dictionary family) or drawn "
                         "flat over the whole dataset (the Part I protocol). "
                         "The two are different populations and the AUC scale "
                         "moves with them: they are NOT comparable")
    ap.add_argument("--k-relations", type=int, default=33)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap on the number of pairs actually extracted. The "
                         "scan does not need the full campaign to locate a peak")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.03,
                    help="minimum attention-FFN gap counted as real")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--suffix", default=".",
                    help="'.' the canonical convention, '' the legacy one. "
                         "Bundles built under the two are not comparable")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--prereg", default=None,
                    help="a file with the predictions written BEFORE this run. "
                         "Its text and hash are stored in the output")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    proto = (CANONICAL if a.suffix == "." else LEGACY_DICT)
    seeds = tuple(range(a.seeds))

    print("=" * 64)
    print("SCANSIONE PICCO   truthprobe %s" % __version__)
    print("=" * 64)
    print("[model] %s" % a.model)

    prereg = None
    if a.prereg:
        with open(a.prereg, "r", encoding="utf-8") as fh:
            txt = fh.read()
        prereg = dict(path=a.prereg, sha256=hashlib.sha256(txt.encode()).hexdigest(),
                      text=txt)
        print("[prereg] %s  sha256 %s" % (a.prereg, prereg["sha256"][:12]))
    else:
        print("[prereg] NONE. Nothing was written down before this run.")

    # ---- pairs -------------------------------------------------------
    if a.sampling == "relation":
        ps = counterfact_by_relation(proto, k=a.k_relations,
                                     n_per=a.pairs_per_relation,
                                     local_file=a.file_counterfact)
    else:
        ps = counterfact_flat(proto, max_pairs=a.max_pairs or 250,
                              local_file=a.file_counterfact)
    pidx = list(ps.pidx)
    items = list(ps.items)
    if a.max_pairs and len(pidx) > a.max_pairs:
        pidx = pidx[:a.max_pairs]
        keep = sorted({i for p in pidx for i in p})
        remap = {old: new for new, old in enumerate(keep)}
        items = [items[i] for i in keep]
        pidx = [(remap[t], remap[f]) for t, f in pidx]
    print("[data] %s sampling, %d pairs, %d sentences"
          % (a.sampling, len(pidx), len(items)))
    print("       example %r" % items[0])
    print("       last token is %s within a pair"
          % ("IDENTICAL" if proto.suffix else "DIFFERENT"))

    # ---- model -------------------------------------------------------
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
    print("[arch]")
    for line in arch.summary():
        print("  " + line)

    H_resid, H_attn, H_ffn = collect(model, tok, items, arch, device, a.batch)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    med, ok = identity_over_blocks(H_resid, H_attn, H_ffn)
    print("[gate] attn + ffn = residual delta, median %.2e   %s"
          % (med, "OK" if ok else "FAILED"))
    if not ok:
        sys.exit("[stop] the additive decomposition does not hold on this model. "
                 "Nothing below this line would mean anything.")

    # ---- the three curves --------------------------------------------
    print("\n--- PER-LAYER HELD-OUT AUC (descriptive) ---------------------")
    m_r, s_r, _ = scan_stream(H_resid, pidx, a.folds, seeds)
    m_a, s_a, _ = scan_stream(H_attn, pidx, a.folds, seeds)
    m_f, s_f, _ = scan_stream(H_ffn, pidx, a.folds, seeds)
    alt = alternation(m_a, s_a, m_f, s_f, a.floor)

    print("  level 0 is the embedding, before any block: under the period "
          "convention it must sit at chance.")
    print("  lvl 0  resid %.3f  %s" % (m_r[0], bar(m_r[0])))
    print("  %-4s %-7s %-7s %-7s  %s" % ("blk", "resid", "attn", "ffn", "leads"))
    for b in range(len(m_a)):
        print("  %-4d %.3f   %.3f   %.3f    %-5s %s"
              % (b, m_r[b + 1], m_a[b], m_f[b], alt[b][2], bar(m_r[b + 1])))

    top = sorted(range(len(m_a)), key=lambda b: -m_r[b + 1])[:3]
    print("\n  highest residual blocks, descriptive: %s"
          % ", ".join("%d (%.3f)" % (b, m_r[b + 1]) for b in top))

    # ---- the honest number -------------------------------------------
    print("\n--- PEAK CHOSEN ON TRAINING FOLDS ----------------------------")
    auc_nested, chosen, counts = nested_peak(H_resid, pidx, a.folds, seeds)
    modal = max(counts, key=lambda k: counts[k])
    print("  held-out AUC at the fold's own peak    %.3f" % auc_nested)
    print("  block chosen, by frequency             %s"
          % ", ".join("%d x%d" % (k - 1, v)
                      for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    print("  modal block                            %d" % (modal - 1))
    spread = len(counts)
    if spread > 1:
        print("  the choice moved across %d blocks: the peak is a region, not a "
              "point, and that is a result" % spread)

    print("\n--- PROPOSAL, NOT A VERDICT ----------------------------------")
    print("  If you accept the modal block, the registry line is:")
    print('    "%s": dict(peak=%d, write_layer=%d, early_block=2),'
          % (a.model, modal - 1, modal))
    print("  Read the curve first. A flat top means the peak is not identified,")
    print("  and pinning one block would hide that.")

    out = a.out or ("scansione_picco_%s.pt"
                    % os.path.basename(a.model.rstrip("/\\")).replace("/", "_"))
    save_bundle(out, proto, a.model,
                dict(auc_resid=m_r, auc_attn=m_a, auc_ffn=m_f,
                     sd_resid=s_r, sd_attn=s_a, sd_ffn=s_f,
                     alternation=[x[2] for x in alt],
                     nested_auc=auc_nested,
                     chosen_blocks=[c - 1 for c in chosen],
                     modal_block=modal - 1),
                analysis=dict(tool="scansione_picco", sampling=a.sampling,
                              k_relations=a.k_relations,
                              pairs_per_relation=a.pairs_per_relation,
                              n_pairs=len(pidx), folds=a.folds, seeds=list(seeds),
                              floor=a.floor, identity_gate=med,
                              prereg=prereg))
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
