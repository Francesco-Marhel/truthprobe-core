# -*- coding: utf-8 -*-
"""
inventario.py  --  read every bundle, identify it by CONTENT, and file it.

Replaces identikit and inventario with one tool, because they answered two
halves of the same question: what is this file, and where should it go.

WHY IDENTITY MUST COME FROM CONTENT
Filenames lie. During this project a bundle was once filed under another
bundle's name, and it was caught only because a dictionary's cosine matrix is a
fingerprint: two bundles with the same content share it whatever they are
called, and matrices that agree to five decimals identify content regardless of
the label on the box.

So this tool never trusts a name. It reads each file, computes a fingerprint of
the cosine matrix, hashes the bytes, and reads the protocol and provenance
recorded inside. Two files with the same content are duplicates even under
different names; two with the same name and different content are a problem
that gets reported rather than silently resolved.

TWO KINDS OF IDENTITY, KEPT APART
  bytes    the sha256 of the file. Two identical files.
  content  a fingerprint of the cosine matrix, invariant to the ORDER of the
           categories and to the filename. Two files measuring the same thing,
           possibly saved twice with different metadata.

The second is the useful one, and it is the one that caught the mix-up.

AND A THIRD THING THAT IS NOT IDENTITY
The PROTOCOL says which question a bundle answers. Two bundles built with
different sentence conventions are not duplicates and not versions of each
other: they are different measurements, and grouping them together would be the
category error this whole apparatus exists to prevent. They are listed apart.

WHAT IT WRITES
With --archive, the surviving bundles are copied into one folder, renamed from
their own content so the name states what the file is, and an INDEX.md is
written listing every bundle with its model, protocol, scale, fingerprint and
what it contains. That index is meant to be read by a person six months later,
which is the situation this tool is for.

    python inventario.py --root .                       # just look
    python inventario.py --root . --archive bundles     # look and file
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict

import torch

from truthprobe import Protocol, __version__


# =====================================================================
#  reading
# =====================================================================
def sha256(path, chunk=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(C, cats):
    """A fingerprint of the cosine matrix, invariant to category ORDER.

    The cells are sorted by the pair of category NAMES, so the same content
    saved with the categories in a different order gives the same fingerprint.
    That invariance is the point: a reordering is not a different measurement,
    and treating it as one is how a bundle gets filed twice.

    Rounded to four decimals before hashing, because two runs of the same
    computation differ in the last bits and an exact hash would call them
    different."""
    K = len(cats)
    order = sorted(range(K), key=lambda i: cats[i])
    vals = []
    for a in range(K):
        for b in range(a + 1, K):
            i, j = order[a], order[b]
            vals.append(round(float(C[i][j]), 4))
    h = hashlib.sha256()
    h.update(",".join(sorted(cats)).encode())
    h.update(b"|")
    h.update(",".join("%.4f" % v for v in vals).encode())
    return h.hexdigest()[:16]


def read_bundle(path):
    """Everything knowable about one file, without trusting its name."""
    rec = dict(path=os.path.abspath(path), name=os.path.basename(path),
               size=os.path.getsize(path), sha=sha256(path), ok=False, why="")
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        rec["why"] = "unreadable (%s)" % type(e).__name__
        return rec
    if not isinstance(d, dict) or "cats" not in d or "cos_peak" not in d:
        rec["why"] = "not a dictionary bundle (no cats / cos_peak)"
        return rec

    cats = [str(c) for c in d["cats"]]
    C = d["cos_peak"]
    C = C.tolist() if hasattr(C, "tolist") else C
    meta = d.get("meta", {}) or {}
    proto = Protocol.from_dict(meta.get("protocol"))

    rec.update(ok=True, cats=cats, K=len(cats),
               fingerprint=fingerprint(C, cats),
               model=meta.get("model", d.get("model", "?")),
               view=meta.get("view", meta.get("component", "?")),
               peak=meta.get("peak_block", d.get("peak_block", "?")),
               n_per=meta.get("pairs_per_relation",
                              d.get("pairs_per_relation", "?")),
               seed=meta.get("seed", "?"),
               head=meta.get("head", None),
               version=meta.get("truthprobe_version", None),
               protocol=(proto.to_dict() if proto else None),
               proto_key=(str(proto.key()) if proto else "UNRECORDED"),
               suffix=(proto.suffix if proto else "?"),
               has=[k for k in ("axes", "t_global", "transfer", "cos_early",
                                "write_centroids", "cos_by_layer", "flip",
                                "gauge_signs", "gauge_margins") if k in d],
               d=(d["axes"].shape[1] if "axes" in d else None),
               identity=meta.get("identity_check_median",
                                 meta.get("gate_block", None)))
    return rec


def archive_name(r):
    """A name that states what the file is, built from its own content."""
    tag = str(r["model"]).split("/")[-1].replace(".", "").replace("-", "_")
    parts = [tag, "K%s" % r["K"], "n%s" % r["n_per"], "s%s" % r["seed"],
             "L%s" % r["peak"]]
    if r["view"] not in ("resid", "?", None):
        parts.append(str(r["view"]))
    if r["suffix"] == "":
        parts.append("nodot")
    return "_".join(parts) + "__" + r["fingerprint"] + ".pt"


# =====================================================================
#  main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=".", help="folder to walk")
    ap.add_argument("--archive", default=None,
                    help="copy the surviving bundles into this folder and write "
                         "an INDEX.md next to them")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy. Off by default: an archive that "
                         "destroys the source cannot be re-run if it is wrong.")
    ap.add_argument("--reindex", default=None,
                    help="rebuild INDEX.md for a folder that is ALREADY an "
                         "archive, reading the bundles that are in it. Nothing "
                         "is copied, moved or renamed: it only rewrites the "
                         "index, so a folder curated by hand keeps its shape.")
    ap.add_argument("--out", default="inventario.json")
    a = ap.parse_args()

    if a.reindex:
        if not os.path.isdir(a.reindex):
            sys.exit("not a folder: %s" % a.reindex)
        files = sorted(f for f in os.listdir(a.reindex) if f.endswith(".pt"))
        if not files:
            sys.exit("no .pt files in %s" % a.reindex)
        print()
        print("[library] truthprobe %s" % __version__)
        print("[reindex] %d bundles in %s" % (len(files), os.path.abspath(a.reindex)))
        recs = []
        for i, f in enumerate(files):
            r = read_bundle(os.path.join(a.reindex, f))
            r["archived_as"] = f
            recs.append(r)
            print("\r  reading %d/%d" % (i + 1, len(files)), end="", flush=True)
        print()
        keep = [r for r in recs if r["ok"]]
        bad = [r for r in recs if not r["ok"]]
        seen, dup = set(), []
        for r in keep:
            if r["fingerprint"] in seen:
                dup.append(r)
            seen.add(r["fingerprint"])
        write_index(a.reindex, keep, dup, {}, {}, bad)
        print("written: %s" % os.path.join(a.reindex, "INDEX.md"))
        if dup:
            print("note: %d bundles in this folder share content with another"
                  % len(dup))
        print()
        return

    paths = []
    for dirpath, _, names in os.walk(a.root):
        if a.archive and os.path.abspath(dirpath).startswith(
                os.path.abspath(a.archive)):
            continue
        for n in names:
            if n.endswith(".pt"):
                paths.append(os.path.join(dirpath, n))
    paths.sort()
    if not paths:
        sys.exit("no .pt files found under %s" % a.root)

    print()
    print("[library] truthprobe %s" % __version__)
    print("[scan] %d files under %s" % (len(paths), os.path.abspath(a.root)))
    recs = []
    for i, p in enumerate(paths):
        recs.append(read_bundle(p))
        print("\r  reading %d/%d" % (i + 1, len(paths)), end="", flush=True)
    print()

    good = [r for r in recs if r["ok"]]
    bad = [r for r in recs if not r["ok"]]

    # ---------------- duplicates ----------------
    by_content = defaultdict(list)
    for r in good:
        by_content[r["fingerprint"]].append(r)
    dupes = {k: v for k, v in by_content.items() if len(v) > 1}
    by_bytes = defaultdict(list)
    for r in good:
        by_bytes[r["sha"]].append(r)
    byte_dupes = {k: v for k, v in by_bytes.items() if len(v) > 1}

    print()
    print("================  WHAT IS HERE  ================")
    print("readable bundles : %d" % len(good))
    print("not bundles      : %d" % len(bad))
    for r in bad:
        print("    %-44s %s" % (r["name"][:44], r["why"]))

    # ---------------- protocols ----------------
    by_proto = defaultdict(list)
    for r in good:
        by_proto[r["proto_key"]].append(r)
    print()
    print("PROTOCOLS  (bundles under different conventions are not versions of")
    print("           each other: they answer different questions)")
    for k, v in sorted(by_proto.items(), key=lambda x: -len(x[1])):
        ex = v[0]
        lab = ("suffix %r, join %r" % (ex["protocol"]["suffix"],
                                       ex["protocol"]["join"])) \
            if ex["protocol"] else "NOT RECORDED (predates the Protocol object)"
        print("  %-52s %3d bundles" % (lab, len(v)))

    print()
    print("DUPLICATES")
    if byte_dupes:
        print("  identical files (same bytes):")
        for k, v in byte_dupes.items():
            print("    %s" % k[:12])
            for r in v:
                print("        %s" % r["path"])
    if dupes:
        print("  same CONTENT under different names or category orders:")
        for k, v in dupes.items():
            if len(set(r["sha"] for r in v)) == 1:
                continue                       # already reported as byte-identical
            print("    fingerprint %s" % k)
            for r in v:
                print("        %-46s  %s" % (r["name"][:46], r["path"]))
    if not dupes and not byte_dupes:
        print("  none")

    # ---------------- a name that lies ----------------
    by_name = defaultdict(set)
    for r in good:
        by_name[r["name"]].add(r["fingerprint"])
    liars = {n: f for n, f in by_name.items() if len(f) > 1}
    if liars:
        print()
        print("SAME NAME, DIFFERENT CONTENT  (a name is not an identity)")
        for n, f in liars.items():
            print("  %-46s %d different contents" % (n[:46], len(f)))

    # ---------------- the list ----------------
    print()
    print("BUNDLES")
    print("  %-30s %-7s %4s %5s %5s %-8s %-16s" %
          ("model", "view", "K", "n", "seed", "suffix", "fingerprint"))
    print("  " + "-" * 82)
    for r in sorted(good, key=lambda x: (str(x["model"]), str(x["view"]),
                                         str(x["K"]), str(x["n_per"]))):
        print("  %-30s %-7s %4s %5s %5s %-8s %-16s"
              % (str(r["model"]).split("/")[-1][:30], str(r["view"])[:7],
                 r["K"], r["n_per"], r["seed"],
                 repr(r["suffix"]), r["fingerprint"]))

    # ---------------- archive ----------------
    if a.archive:
        os.makedirs(a.archive, exist_ok=True)
        keep, skipped = [], []
        seen = set()
        for r in sorted(good, key=lambda x: x["path"]):
            if r["fingerprint"] in seen:
                skipped.append(r)
                continue
            seen.add(r["fingerprint"])
            keep.append(r)
        print()
        print("================  ARCHIVE  ================")
        print("keeping %d, skipping %d duplicates" % (len(keep), len(skipped)))
        for r in keep:
            dst = os.path.join(a.archive, archive_name(r))
            r["archived_as"] = os.path.basename(dst)
            if os.path.abspath(dst) == r["path"]:
                continue
            if a.move:
                shutil.move(r["path"], dst)
            else:
                shutil.copy2(r["path"], dst)
        write_index(a.archive, keep, skipped, dupes, liars, bad)
        print("written: %s" % os.path.join(a.archive, "INDEX.md"))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(truthprobe_version=__version__,
                       root=os.path.abspath(a.root), n_files=len(paths),
                       bundles=good, not_bundles=bad), fh,
                  ensure_ascii=False, indent=2)
    print()
    print("written: %s" % a.out)
    print()


def write_index(folder, keep, skipped, dupes, liars, bad):
    """A table, and the two things you must know before picking a file.

    Written for the person who opens this folder in six months. Deliberately
    short: an index nobody reads protects nothing."""
    by_proto = defaultdict(list)
    for r in keep:
        by_proto[r["proto_key"]].append(r)

    L = ["# Bundle archive\n\n",
         "Identified by CONTENT: the fingerprint is a hash of the cosine "
         "matrix, invariant to category order and to the filename.\n\n",
         "1. **Different protocol, different measurement.** Do not compare "
         "across the groups below.\n",
         "2. **A bundle carries the axes, not the gauge.** Signs are fixed "
         "afterwards by `reorient_gauge.py`, which writes `_gauge.json` "
         "beside it.\n"]

    for key, group in sorted(by_proto.items(), key=lambda x: -len(x[1])):
        ex = group[0]
        head = ("suffix `%r`, join `%r`, pool `%r`"
                % (ex["protocol"]["suffix"], ex["protocol"]["join"],
                   ex["protocol"]["pool"])) if ex["protocol"] \
            else "protocol NOT RECORDED"
        L.append("\n## %s\n\n" % head)
        L.append("| file | model | view | K | n | seed | block | contains | fingerprint |\n")
        L.append("|---|---|---|---:|---:|---:|---:|---|---|\n")
        for r in sorted(group, key=lambda x: (str(x["model"]), str(x["view"]),
                                              -int(x["K"] or 0))):
            L.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | `%s` |\n"
                     % (r.get("archived_as", r["name"]),
                        str(r["model"]).split("/")[-1], r["view"], r["K"],
                        r["n_per"], r["seed"], r["peak"],
                        ", ".join(r["has"]), r["fingerprint"]))

    if skipped or liars or bad:
        L.append("\n## Notes\n\n")
        for r in skipped:
            L.append("- duplicate content, not archived: `%s`\n"
                     % r.get("path", r["name"]))
        for n, f in (liars or {}).items():
            L.append("- same name, %d different contents: `%s`\n" % (len(f), n))
        for r in bad:
            L.append("- not a bundle: `%s` (%s)\n"
                     % (r.get("path", r["name"]), r["why"]))

    L.append("\n## Columns\n\n")
    L.append("- **view**: the stream the axes were fitted on. `resid`, `attn`, "
             "`ffn`, or `headNN`.\n")
    L.append("- **block**: the peak block.\n")
    L.append("- **contains**: `transfer` holds the within-category held-out AUC "
             "on its diagonal, the knowledge proxy the restricted arrangement "
             "law reads. `cos_early` is the surface control, under the OLD "
             "orientation because a shallow block has no stable gauge.\n")
    L.append("- **fingerprint**: same fingerprint means same measurement.\n")
    L.append("\nRebuild this file with `inventario.py --reindex <folder>`.\n")

    with open(os.path.join(folder, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(L))


if __name__ == "__main__":
    main()
