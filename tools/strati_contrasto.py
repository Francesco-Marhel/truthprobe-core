# -*- coding: utf-8 -*-
"""
strati_contrasto.py  --  partition a facts file by the KIND of contrast it offers.

Formerly strati_tqa.py. Renamed because it is not tied to TruthfulQA: it takes
any facts file and asks what kind of comparison each item actually poses. That
question matters for any hand-written corpus, and it will matter for a set of
texts that say the same thing in code, in words and in numbers.

No model is loaded. Text only, on CPU, in a second.

THE THREE STRATA, applied in this order (the order is part of the rule)

  abstention  the true target is not an assertion but a refusal to assert.
              There is no truth contrast to read: an abstention is being
              compared against a claim. Measured on TruthfulQA, the axis reads
              these BACKWARDS, 13 items out of 14 on two models, because on
              unpaired text it orders something closer to assertiveness than to
              truth.
  polarity    the two targets oppose in polarity rather than in content.
              P1: one opens with Yes and the other with No.
              P2: negation present in one and absent in the other.
              This is the contrast the design excludes on purpose, and its cost
              is known: an affirmative-fitted axis anti-classifies negated pairs.
  assertion   everything else: two affirmative assertions about the same fact.
              Inside it, 'minimal' marks pairs of equal word length differing in
              exactly one position, which is the contrast the axis was built for.

TWO DESCRIPTIVE COLUMNS THAT ARE NOT DECORATION
  jac   lexical overlap between the two targets. High means close to a minimal
        pair, low means two different sentences. It is the measure of how much
        the intra-pair difference can actually cancel the surface.
  nT/nF length in words. Feeding an axis targets of very different length lets
        length into the measurement.

The rule that fired is always printed, and no verdict is: P2 also fires on
genuine assertions containing a negation word, and whether that matters is the
reader's call.

    python strati_contrasto.py --facts fatti.json
"""

import argparse
import hashlib
import json
import os
import re
import statistics as stx
import sys

from truthprobe import __version__

RULESET = "strati_contrasto_v1"

ABSTENTION_EXACT = {"unknown", "no comment", "i have no comment", "i don't know",
                    "i do not know", "i have no idea", "nothing in particular",
                    "nothing"}
ABSTENTION_CONTAINS = ["i have no comment", "there is no answer",
                       "impossible to say", "does not make sense",
                       "doesn't make sense", "no way to know",
                       "cannot be determined"]
NEGATION = {"not", "no", "never", "none", "nothing", "neither", "nor", "cannot",
            "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
            "weren't", "won't", "shouldn't", "couldn't", "wouldn't", "hasn't",
            "haven't", "hadn't"}

_PUNCT = re.compile(r"[^a-z0-9' ]+")
_SPACE = re.compile(r"\s+")


def normalize(t):
    return _SPACE.sub(" ", _PUNCT.sub(" ", t.lower().replace("\u2019", "'"))).strip()


def toks(t):
    n = normalize(t)
    return n.split() if n else []


def is_abstention(t):
    n = normalize(t)
    return n in ABSTENTION_EXACT or any(m in n for m in ABSTENTION_CONTAINS)


def polarity_rule(tt, tf):
    a, b = toks(tt), toks(tf)
    if a and b and {a[0], b[0]} == {"yes", "no"}:
        return "P1"
    strip = lambda t: t[1:] if (t and t[0] in ("yes", "no")) else t
    na = sum(1 for x in strip(a) if x in NEGATION)
    nb = sum(1 for x in strip(b) if x in NEGATION)
    return "P2" if (na > 0) != (nb > 0) else None


def jaccard(tt, tf):
    x, y = set(toks(tt)), set(toks(tf))
    return len(x & y) / len(x | y) if (x or y) else 0.0


def is_minimal(tt, tf):
    a, b = toks(tt), toks(tf)
    return bool(a) and len(a) == len(b) and sum(1 for x, y in zip(a, b) if x != y) == 1


def first(o, key):
    v = o.get(key)
    if isinstance(v, list):
        for x in v:
            if isinstance(x, str) and x.strip():
                return x
        return ""
    return v if isinstance(v, str) else ""


def classify(item):
    tt, tf = first(item, "targets"), first(item, "false_targets")
    if not tt or not tf:
        return "incomplete", "missing target", "excluded from the strata"
    if is_abstention(tt):
        return "abstention", "true target", (
            "false one is an abstention too" if is_abstention(tf) else "")
    if is_abstention(tf):
        return "abstention", "false target", "abstention on the false side"
    r = polarity_rule(tt, tf)
    return ("polarity", r, "") if r else ("assertion", "", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: a folder beside the input file")
    ap.add_argument("--prefix", default=None)
    a = ap.parse_args()
    if not os.path.isfile(a.facts):
        sys.exit("file not found: %s" % a.facts)

    with open(a.facts, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    data = [x for x in data if isinstance(x, dict)]

    h = hashlib.sha256()
    with open(a.facts, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    digest = h.hexdigest()

    prefix = a.prefix or os.path.splitext(os.path.basename(a.facts))[0]
    out_dir = a.out_dir or os.path.join(os.path.dirname(a.facts) or ".",
                                        prefix + "_strati")
    os.makedirs(out_dir, exist_ok=True)

    rows, buckets, minimal = [], {k: [] for k in
                                  ("abstention", "polarity", "assertion", "incomplete")}, []
    for item in data:
        st, rule, note = classify(item)
        tt, tf = first(item, "targets"), first(item, "false_targets")
        nT, nF = len(toks(tt)), len(toks(tf))
        jac = jaccard(tt, tf) if (tt and tf) else 0.0
        mini = is_minimal(tt, tf) if st == "assertion" else False
        tagged = dict(item, stratum=st, stratum_rule=rule, ruleset=RULESET)
        if mini:
            tagged["minimal_contrast"] = True
            minimal.append(tagged)
        buckets[st].append(tagged)
        rows.append(dict(id=item.get("id", "?"), stratum=st, rule=rule,
                         nT=nT, nF=nF, jac=jac, minimal=mini, note=note))

    print()
    print("[library] truthprobe %s   ruleset %s   (NO model is loaded)"
          % (__version__, RULESET))
    print("[file] %s" % os.path.basename(a.facts))
    print("[sha256] %s" % digest)
    print()
    print("%-12s %-11s %-14s %5s %5s %6s %5s  %s"
          % ("id", "stratum", "rule", "nT", "nF", "jac", "min", "note"))
    print("-" * 78)
    for r in rows:
        print("%-12s %-11s %-14s %5d %5d %6.2f %5s  %s"
              % (str(r["id"])[:12], r["stratum"], r["rule"], r["nT"], r["nF"],
                 r["jac"], "yes" if r["minimal"] else "", r["note"]))

    n = len(rows)
    cnt = {k: len(v) for k, v in buckets.items()}
    p1 = sum(1 for r in rows if r["rule"] == "P1")
    p2 = sum(1 for r in rows if r["rule"] == "P2")
    asr = [r for r in rows if r["stratum"] == "assertion"]
    med = lambda xs: stx.median(xs) if xs else float("nan")

    print()
    print("================  PARTITION  ================")
    print("items                : %d" % n)
    print("abstention           : %d" % cnt["abstention"])
    print("polarity             : %d   (P1 yes/no %d, P2 negation %d)"
          % (cnt["polarity"], p1, p2))
    print("assertion            : %d   (of which minimal %d)"
          % (cnt["assertion"], len(minimal)))
    print("incomplete           : %d" % cnt["incomplete"])
    print()
    print("lexical overlap between the two targets (median)")
    print("  all items          : %.2f" % med([r["jac"] for r in rows]))
    print("  assertion only     : %.2f" % med([r["jac"] for r in asr]))
    print("median length difference in words, assertion only: %.1f"
          % med([abs(r["nT"] - r["nF"]) for r in asr]))

    written = []
    for k in ("abstention", "polarity", "assertion", "incomplete"):
        if not buckets[k]:
            continue
        p = os.path.join(out_dir, "%s_%s.json" % (prefix, k))
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(buckets[k], fh, ensure_ascii=False, indent=2)
        written.append((p, len(buckets[k])))
    if minimal:
        p = os.path.join(out_dir, "%s_assertion_minimal.json" % prefix)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(minimal, fh, ensure_ascii=False, indent=2)
        written.append((p, len(minimal)))

    rep = os.path.join(out_dir, "%s_report.txt" % prefix)
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write("ruleset %s\ntruthprobe %s\ninput %s\nsha256 %s\nitems %d\n\n"
                 % (RULESET, __version__, os.path.abspath(a.facts), digest, n))
        fh.write("abstention %d\npolarity %d (P1 %d, P2 %d)\nassertion %d "
                 "(minimal %d)\nincomplete %d\n\n"
                 % (cnt["abstention"], cnt["polarity"], p1, p2,
                    cnt["assertion"], len(minimal), cnt["incomplete"]))
        fh.write("ABSTENTION_EXACT %s\n\nABSTENTION_CONTAINS %s\n\nNEGATION %s\n\n"
                 % (sorted(ABSTENTION_EXACT), ABSTENTION_CONTAINS, sorted(NEGATION)))
        fh.write("order: abstention, then polarity, then assertion\n\n")
        for r in rows:
            fh.write("%s\t%s\t%s\t%d\t%d\t%.3f\t%s\n"
                     % (r["id"], r["stratum"], r["rule"], r["nT"], r["nF"],
                        r["jac"], "minimal" if r["minimal"] else ""))

    print()
    print("written:")
    for p, k in written:
        print("  %-56s %3d items" % (p, k))
    print("  %-56s" % rep)
    print()
    print("each stratum keeps the input schema, so it can be passed straight to")
    print("any tool with --facts.")
    print()


if __name__ == "__main__":
    main()
