# -*- coding: utf-8 -*-
"""
behav_contrasto.py  --  does the model itself prefer the labelled-true target?

Converted to a thin wrapper over the truthprobe core.

WHY THIS EXISTS
"Verify behaviour before geometry" applied to a CONTRAST rather than to a
single target. The know-rate answers "does the model produce the true target"
with a string match on a short greedy completion, which fails whenever the
answer is a whole sentence. This asks the comparative question instead: under
the same prefix, does the model give more probability to the target labelled
true or to the one labelled false?

It measures at the OUTPUT layer and never touches hidden states, so it is
independent of the geometric probe and can be compared with it without
contaminating it. That comparison is the point: a probe cannot be faulted for
failing to read a belief the model does not hold.

THREE CRITERIA, NEVER COLLAPSED INTO ONE
  sum    log P(target | prompt) summed over the target's tokens. Favours SHORT
         targets: fewer negative terms.
  mean   the same divided by the number of tokens. The declared criterion.
  pmi    log P(target | prompt) minus log P(target | a neutral prefix). Controls
         for how generic or frequent the target is on its own.

On single-word targets the three nearly coincide: on CounterFact obscure facts
they read 89.2, 89.2 and 90.8 per cent. On whole-sentence targets they diverge
badly: on TruthfulQA assertions they read 43.2, 34.1 and 56.8. When they
disagree, the notion of preference is itself ill-defined on that material, and
that is a result about the material, not a defect of the tool. All three are
printed for exactly this reason.

The false target comes from 'false_targets' when present, otherwise by SWAP
within the same relation, using the library's rule.

No verdicts are printed.

    python behav_contrasto.py --facts fatti.json --model google/gemma-2-2b-it --pmi
"""

import argparse
import json
import os
import statistics as stx
import sys

import torch

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import from_json

NEUTRAL_PREFIX = "Answer:"


@torch.no_grad()
def logprob(tok, model, device, prefix, target, proto):
    """log P(target | prefix), summed over the target's tokens only.

    The sentence is built by the Protocol, the same function the geometric side
    uses, so the two never drift apart. The suffix is dropped here: scoring a
    target followed by a period would mix the period's probability into the
    comparison."""
    text = proto.with_(suffix="").sentence(prefix, target)
    pre = tok(prefix.rstrip(), return_tensors="pt").input_ids
    full = tok(text, return_tensors="pt").input_ids
    n_pre, n_full = pre.shape[1], full.shape[1]
    if n_full <= n_pre:
        return float("nan"), 0
    full = full.to(device)
    lp = torch.log_softmax(model(full).logits[0, :-1, :].float(), dim=-1)
    tgt = full[0, 1:]
    sel = lp[n_pre - 1:, :].gather(1, tgt[n_pre - 1:].unsqueeze(1)).squeeze(1)
    return float(sel.sum()), int(sel.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--facts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pmi", action="store_true",
                    help="add the genericity control (one extra pass per target)")
    ap.add_argument("--show-pairs", action="store_true",
                    help="print the contrast actually scored, to check the swap rule")
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not os.path.isfile(a.facts):
        sys.exit("file not found: %s" % a.facts)
    proto = (CANONICAL if a.suffix == "." else LEGACY_DICT)
    ps = from_json(a.facts, proto)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=(torch.bfloat16 if a.dtype == "bfloat16" else torch.float32),
        use_safetensors=True, trust_remote_code=False).to(device)
    model.eval()

    print()
    print("[library] truthprobe %s" % __version__)
    print("[model]   %s on %s (%s)" % (a.model, device, a.dtype))
    print("[criterion] preference on 'mean', log P per token")

    header = "%-30s %-9s %4s %4s %9s %9s %9s %9s" % (
        "id", "origin", "nT", "nF", "meanT", "meanF", "dMean", "dSum")
    if a.pmi:
        header += " %9s" % "dPMI"
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for pr in ps.pairs:
        sT, nT = logprob(tok, model, device, pr.prompt, pr.target_true, proto)
        sF, nF = logprob(tok, model, device, pr.prompt, pr.target_false, proto)
        if nT == 0 or nF == 0:
            print("%-30s  skipped (empty after tokenisation)" % str(pr.ident)[:30])
            continue
        mT, mF = sT / nT, sF / nF
        r = dict(id=pr.ident, origin=pr.origin, category=pr.category,
                 prompt=pr.prompt, target_true=pr.target_true,
                 target_false=pr.target_false, n_true=nT, n_false=nF,
                 sum_true=sT, sum_false=sF, mean_true=mT, mean_false=mF,
                 d_mean=mT - mF, d_sum=sT - sF,
                 prefers_true_mean=bool(mT > mF), prefers_true_sum=bool(sT > sF))
        line = "%-30s %-9s %4d %4d %9.3f %9.3f %+9.3f %+9.3f" % (
            str(pr.ident)[:30], pr.origin, nT, nF, mT, mF, mT - mF, sT - sF)
        if a.pmi:
            bT, kT = logprob(tok, model, device, NEUTRAL_PREFIX, pr.target_true, proto)
            bF, kF = logprob(tok, model, device, NEUTRAL_PREFIX, pr.target_false, proto)
            pT = sT - bT if kT else float("nan")
            pF = sF - bF if kF else float("nan")
            r["d_pmi"] = pT - pF
            r["prefers_true_pmi"] = bool(pT > pF)
            line += " %+9.3f" % (pT - pF)
        if a.show_pairs:
            line += "\n    true: %r   false: %r" % (pr.target_true, pr.target_false)
        print(line)
        rows.append(r)

    if not rows:
        sys.exit("no item scored")
    n = len(rows)
    pm = sum(r["prefers_true_mean"] for r in rows)
    pss = sum(r["prefers_true_sum"] for r in rows)
    print()
    print("================  SUMMARY  ================")
    print("items scored                : %d" % n)
    print("prefers the true target")
    print("  mean (declared criterion) : %d/%d   %.1f%%" % (pm, n, 100 * pm / n))
    print("  sum                       : %d/%d   %.1f%%" % (pss, n, 100 * pss / n))
    if a.pmi:
        pp = sum(r["prefers_true_pmi"] for r in rows)
        print("  pmi                       : %d/%d   %.1f%%" % (pp, n, 100 * pp / n))
        agree = sum(1 for r in rows
                    if r["prefers_true_mean"] == r["prefers_true_sum"] == r["prefers_true_pmi"])
        print("  all three agree           : %d/%d" % (agree, n))
        print("  (wide disagreement means preference is ill-defined on this")
        print("   material, which is a result about the material)")
    print("median margin (mean, true minus false) : %+.3f"
          % stx.median(r["d_mean"] for r in rows))
    print("median target length in tokens, true / false: %d / %d"
          % (stx.median(r["n_true"] for r in rows),
             stx.median(r["n_false"] for r in rows)))

    out = a.out or ("behav_%s__%s.json"
                    % (os.path.splitext(os.path.basename(a.facts))[0],
                       a.model.replace("/", "_")))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__, model=a.model,
                       facts=os.path.abspath(a.facts), protocol=proto.to_dict(),
                       criterion="mean", pmi=bool(a.pmi), items=rows,
                       skipped=getattr(ps, "skipped", [])),
                  fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % out)
    print()


if __name__ == "__main__":
    main()
