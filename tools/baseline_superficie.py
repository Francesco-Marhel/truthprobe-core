# -*- coding: utf-8 -*-
"""
baseline_superficie.py  --  how much separates from the TEXT alone, no model?

Converted to a thin wrapper over the truthprobe core.

WHY THIS IS THE FIRST THING TO RUN
The permutation null prices layer-selection optimism, not surface. Swapping
true and false within a pair destroys the truth signal and the lexical signal
together, so an axis reading the shape of the sentences clears the null all the
same. This tool supplies the missing control: a classifier on text features
alone, never loading a model, evaluated with the same protocol.

If this baseline approaches what the model achieves, the separation is surface
and the geometry adds nothing. Measured: on CounterFact it sits at chance
(0.510, p = 0.45), because the pairs are type-matched and there is nothing to
read in the text; on TruthfulQA it reaches 0.814, above the best layer of every
model tested, because the wrong answers there have a recognisable vocabulary.
That is a property of the benchmark's construction, not of any model.

The features know nothing about the world: length in words and characters,
presence of negation, opening with yes or no, presence of digits, and a bag of
the most frequent corpus words.

No verdicts are printed.

    python baseline_superficie.py --dataset counterfact
    python baseline_superficie.py --facts strati_tqa/assertion.json
"""

import argparse
import json
import math
import os
import random
import re
import statistics as stx
from collections import Counter

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import counterfact_flat, from_json
from truthprobe.stats import kfold_pairs

NEG = {"not", "no", "never", "none", "nothing", "cannot", "can't", "don't",
       "doesn't", "didn't", "isn't", "aren't", "wasn't", "won't", "neither", "nor"}
_PUNCT = re.compile(r"[^a-z0-9' ]+")


def words(s):
    return _PUNCT.sub(" ", s.lower().replace("\u2019", "'")).split()


def build_vocab(texts, top):
    c = Counter()
    for t in texts:
        c.update(set(words(t)))
    return [w for w, _ in c.most_common(top)]


def featurize(s, vocab):
    w = words(s)
    ws = set(w)
    return [len(w) / 10.0, len(s) / 50.0,
            1.0 if any(x in NEG for x in w) else 0.0,
            1.0 if w and w[0] == "yes" else 0.0,
            1.0 if w and w[0] == "no" else 0.0,
            1.0 if any(any(ch.isdigit() for ch in x) for x in w) else 0.0
            ] + [1.0 if v in ws else 0.0 for v in vocab] + [1.0]


def logistic(X, y, epochs=400, lr=0.5, l2=1e-3):
    d, n = len(X[0]), len(X)
    w = [0.0] * d
    for _ in range(epochs):
        g = [0.0] * d
        for xi, yi in zip(X, y):
            z = sum(a * b for a, b in zip(w, xi))
            e = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) - yi
            for k in range(d):
                g[k] += e * xi[k]
        for k in range(d):
            w[k] -= lr * (g[k] / n + l2 * w[k])
    return w


def paired_acc(w, ft, ff):
    good = tie = 0
    for a, b in zip(ft, ff):
        sa = sum(x * y for x, y in zip(w, a))
        sb = sum(x * y for x, y in zip(w, b))
        if sa > sb:
            good += 1
        elif sa == sb:
            tie += 1
    return (good + 0.5 * tie) / len(ft)


def cv(pairs, vocab, folds, seed, signs=None):
    n = len(pairs)
    if signs is None:
        signs = [1] * n
    out = []
    for tr, te in kfold_pairs(n, folds, seed):
        X, y = [], []
        for i in tr:
            t, f = pairs[i]
            if signs[i] < 0:
                t, f = f, t
            X.append(featurize(t, vocab)); y.append(1.0)
            X.append(featurize(f, vocab)); y.append(0.0)
        w = logistic(X, y)
        ft = [featurize(pairs[i][0] if signs[i] > 0 else pairs[i][1], vocab) for i in te]
        ff = [featurize(pairs[i][1] if signs[i] > 0 else pairs[i][0], vocab) for i in te]
        out.append(paired_acc(w, ft, ff))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", default=None, choices=["counterfact"],
                    help="pull pairs from CounterFact")
    ap.add_argument("--facts", default=None, help="or read a facts file")
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--vocab", type=int, default=40)
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not (a.dataset or a.facts):
        ap.error("give either --dataset counterfact or --facts <file>")

    proto = (CANONICAL if a.suffix == "." else LEGACY_DICT).with_(seed=a.seed)
    print()
    print("[library] truthprobe %s   (NO model is loaded)" % __version__)
    if a.facts:
        ps = from_json(a.facts, proto)
        src = os.path.basename(a.facts)
    else:
        ps = counterfact_flat(proto, max_pairs=a.max_pairs,
                              local_file=a.file_counterfact)
        src = proto.dataset
    pairs = [(ps.items[i], ps.items[j]) for i, j in ps.pidx]

    vocab = build_vocab([t for t, _ in pairs] + [f for _, f in pairs], a.vocab)
    obs = cv(pairs, vocab, a.folds, a.seed)
    rng = random.Random(a.seed)
    null = []
    for b in range(a.perms):
        null.append(cv(pairs, vocab, a.folds, a.seed,
                       [1 if rng.random() < 0.5 else -1 for _ in pairs]))
        print("\r  [null] %d/%d" % (b + 1, a.perms), end="", flush=True)
    print()
    null.sort()
    q95 = null[int(0.95 * (len(null) - 1))]
    p = (1 + sum(1 for x in null if x >= obs)) / (len(null) + 1)

    jac = stx.median(len(set(words(t)) & set(words(f))) /
                     max(1, len(set(words(t)) | set(words(f)))) for t, f in pairs)
    lt = stx.median(len(words(t)) for t, _ in pairs)
    lf = stx.median(len(words(f)) for _, f in pairs)

    print()
    print("================  SURFACE ONLY  ================")
    print("source                 : %s   %d pairs" % (src, len(pairs)))
    print("held-out accuracy (%d folds) : %.3f" % (a.folds, obs))
    print("null: mean %.3f   95pct %.3f   p = %.4f"
          % (sum(null) / len(null), q95, p))
    print("margin over the 95th percentile : %+.3f" % (obs - q95))
    print("median target length, true / false : %d / %d words" % (lt, lf))
    print("median lexical overlap between the two : %.2f" % jac)
    print()
    print("put the model's layer-0 and first-block AUC beside this number.")

    out = a.out or ("superficie_%s.json"
                    % (os.path.splitext(os.path.basename(a.facts))[0] if a.facts
                       else "counterfact"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__, source=src,
                       protocol=proto.to_dict(), n_pairs=len(pairs),
                       accuracy=obs, null_mean=sum(null) / len(null),
                       null_p95=q95, p=p, jaccard=jac,
                       len_true=lt, len_false=lf), fh, ensure_ascii=False, indent=2)
    print("written: %s" % out)
    print()


if __name__ == "__main__":
    main()
