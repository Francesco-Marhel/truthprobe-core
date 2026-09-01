# -*- coding: utf-8 -*-
"""
crea_dizionario.py  --  build and export truth-dictionary bundles.

Converted to a thin wrapper over the truthprobe core. Same bundle format, same
statistics, same registry-driven workflow; the shared machinery now lives in
the library instead of being restated here.

WHAT A BUNDLE IS
A measured dictionary: K per-category truth axes, the pooled global axis, the
FFN write centroids, the signed cosine matrices at the peak and at an early
block, and the held-out transfer matrix. A few tens of kilobytes that act as a
K-direction truth and category monitor, and as the reference any sparse
autoencoder has to beat.

WHAT CHANGED IN THE CONVERSION
  The sentence convention is a Protocol object, written into every bundle and
  checked before any comparison. It used to be implicit: this file built
  "prompt" + raw target while truth_probe.py built "prompt target." with a
  final period, and the two produce axes at cosine +0.52 on Gemma-2-2b while
  the arrangement survives at Mantel +0.775. Nobody noticed for months.

  The reference archive now carries the protocol it was produced under, so the
  check is skipped with a reason instead of failing forever once you switch
  convention. The archive is the K=8 no-period one.

  The per-block flip curves come from truthprobe.stats.frame_curves, which
  works on any contribution, so the same measure applies to a single head or a
  single expert without rewriting it.

  float32 is required: the additive identity gate subtracts large residual
  states to obtain small per-block deltas, and bfloat16 loses that to
  catastrophic cancellation. The gate refuses to proceed instead of warning.

    python crea_dizionario.py                          # every model in MODELS
    python crea_dizionario.py --models google/gemma-2-2b --peak 11 --write-layer 12
    python crea_dizionario.py --component attn --out-dir dizionari_attn
    python crea_dizionario.py --suffix "" --out-dir dizionari_legacy
"""

import argparse
import json
import os
import sys

import torch

from truthprobe import Protocol, CANONICAL, LEGACY_DICT, __version__
from truthprobe.data import counterfact_by_relation
from truthprobe.geometry import unit, fit_axis, cosine_matrix
from truthprobe.hooks import describe, BlockCapture, identity_gate
from truthprobe.stats import (auc_score, kfold_pairs, project_and_score,
                              decoding_with_null, frame_curves)


# =====================================================================
#  ================   USER CONFIG   ================
# =====================================================================
# (1) MODELS registry. name -> the block where the truth axis PEAKS, the layer
#     where the FFN WRITES (= peak + 1), and the shallow EARLY control block.
#     The peak is MODEL-SPECIFIC and must be MEASURED with a per-layer signal
#     scan, never guessed, before a model is added here.
MODELS = {
    "Qwen/Qwen2.5-3B":         dict(peak=16, write_layer=17, early_block=2),
    "meta-llama/Llama-3.2-3B": dict(peak=9,  write_layer=10, early_block=2),
    "google/gemma-2-2b":       dict(peak=11, write_layer=12, early_block=2),
    "Qwen/Qwen2.5-1.5B":       dict(peak=15, write_layer=16, early_block=2),
    # "meta-llama/Llama-3.2-1B": dict(peak=7, write_layer=8, early_block=2),
    # ---- ADD A NEW MODEL HERE ----
}

# (2) CATEGORIES.
K_RELATIONS = 8
PAIRS_PER_RELATION = 60
RELATION_WHITELIST = None    # None = automatic top-K. To pin an exact set:
                             # ["P103","P1412","P176","P27","P30","P37","P413","P495"]
EARLY_BLOCK = 2
# =====================================================================
#  ================   END USER CONFIG   ================
# =====================================================================


# =====================================================================
#  the canonical K=8 archive, WITH the protocol it was produced under
# =====================================================================
CANON_CATS = ["P103", "P1412", "P176", "P27", "P30", "P37", "P413", "P495"]

REFERENCE_PROTOCOL = LEGACY_DICT      # no final period, space from the dataset

REFERENCE = {
    "Qwen/Qwen2.5-3B": dict(cats=CANON_CATS, cos_peak=[
        [1.00, 0.36, 0.13, 0.28, -0.10, 0.27, -0.02, 0.11],
        [0.36, 1.00, 0.11, 0.38, -0.13, 0.76, -0.06, 0.09],
        [0.13, 0.11, 1.00, 0.25, 0.07, 0.00, 0.00, 0.03],
        [0.28, 0.38, 0.25, 1.00, -0.20, 0.28, -0.04, 0.16],
        [-0.10, -0.13, 0.07, -0.20, 1.00, -0.07, -0.09, 0.05],
        [0.27, 0.76, 0.00, 0.28, -0.07, 1.00, -0.06, 0.09],
        [-0.02, -0.06, 0.00, -0.04, -0.09, -0.06, 1.00, -0.01],
        [0.11, 0.09, 0.03, 0.16, 0.05, 0.09, -0.01, 1.00]], cos_early=[
        [1.00, 0.31, 0.04, -0.19, -0.08, -0.26, -0.05, 0.09],
        [0.31, 1.00, -0.02, -0.24, -0.03, -0.90, -0.09, 0.08],
        [0.04, -0.02, 1.00, -0.03, 0.01, 0.03, 0.03, 0.01],
        [-0.19, -0.24, -0.03, 1.00, 0.23, 0.23, 0.03, 0.27],
        [-0.08, -0.03, 0.01, 0.23, 1.00, 0.04, -0.10, 0.13],
        [-0.26, -0.90, 0.03, 0.23, 0.04, 1.00, 0.08, -0.09],
        [-0.05, -0.09, 0.03, 0.03, -0.10, 0.08, 1.00, 0.01],
        [0.09, 0.08, 0.01, 0.27, 0.13, -0.09, 0.01, 1.00]]),
    "meta-llama/Llama-3.2-3B": dict(cats=CANON_CATS, cos_peak=[
        [1.00, 0.69, 0.40, 0.66, -0.14, 0.56, 0.00, 0.51],
        [0.69, 1.00, 0.38, 0.62, -0.15, 0.62, 0.01, 0.52],
        [0.40, 0.38, 1.00, 0.51, -0.01, 0.40, 0.06, 0.51],
        [0.66, 0.62, 0.51, 1.00, -0.09, 0.55, 0.06, 0.75],
        [-0.14, -0.15, -0.01, -0.09, 1.00, -0.09, -0.06, -0.03],
        [0.56, 0.62, 0.40, 0.55, -0.09, 1.00, 0.03, 0.52],
        [0.00, 0.01, 0.06, 0.06, -0.06, 0.03, 1.00, 0.06],
        [0.51, 0.52, 0.51, 0.75, -0.03, 0.52, 0.06, 1.00]], cos_early=[
        [1.00, 0.08, 0.05, -0.05, -0.06, -0.18, -0.03, 0.03],
        [0.08, 1.00, 0.04, -0.16, -0.06, -0.86, -0.03, 0.08],
        [0.05, 0.04, 1.00, 0.10, -0.05, -0.04, 0.00, 0.08],
        [-0.05, -0.16, 0.10, 1.00, 0.17, 0.21, 0.04, 0.11],
        [-0.06, -0.06, -0.05, 0.17, 1.00, 0.08, -0.01, 0.07],
        [-0.18, -0.86, -0.04, 0.21, 0.08, 1.00, 0.03, -0.11],
        [-0.03, -0.03, 0.00, 0.04, -0.01, 0.03, 1.00, -0.01],
        [0.03, 0.08, 0.08, 0.11, 0.07, -0.11, -0.01, 1.00]]),
}


def verify_against_reference(model_name, bundle, proto, tol):
    """Compare a fresh bundle with the archived K=8 matrices.

    The archive was produced under REFERENCE_PROTOCOL. Comparing a bundle built
    under a different convention would fail forever for a reason that has
    nothing to do with the code, so the check declares that and stops."""
    ref = REFERENCE.get(model_name)
    if ref is None:
        print("  [verify] no archived reference for %s; skipped." % model_name)
        return
    if not proto.compatible_with(REFERENCE_PROTOCOL):
        print("  [verify] skipped: the archive was produced with suffix %r, this "
              "bundle uses %r." % (REFERENCE_PROTOCOL.suffix, proto.suffix))
        print("           Not an error: they are different measurements. The "
              "arrangement survives the change (Mantel ~+0.775 on Gemma-2-2b), "
              "the axes do not (cosine ~+0.52).")
        return
    if list(bundle["cats"]) != ref["cats"]:
        print("  [verify] category set differs from the K=8 archive; skipped.")
        return
    print("  [verify] %s against the canonical K=8 archive (tol %.3f; the archive "
          "is rounded to 2 decimals, so ~0.005 per cell plus fp32 drift is "
          "normal):" % (model_name, tol))
    for key in ("cos_peak", "cos_early"):
        M = bundle[key].float()
        R = torch.tensor(ref[key], dtype=torch.float32)
        off = ~torch.eye(len(R), dtype=torch.bool)
        mx = float((M - R).abs()[off].max())
        print("    %-9s max off-diagonal |fresh - archived| = %.3f   -> %s"
              % (key, mx, "MATCH" if mx <= tol else "DIVERGES"))


# =====================================================================
#  the two matrices
# =====================================================================
def axis_cosine_matrix(H, cat_pairs):
    """Full-fit axis per category, then the KxK SIGNED cosine matrix.

    Each axis is oriented true-positive within its OWN category, so a negative
    cell means the shared direction reads the other category's truth backwards.
    But that sign depends on the orientation of BOTH categories involved: only
    products around closed cycles are gauge-invariant."""
    cats = sorted(cat_pairs)
    axes = {c: fit_axis(H, cat_pairs[c])["v1"] for c in cats}
    M, _ = cosine_matrix([axes[c] for c in cats])
    return cats, M, axes


def transfer_matrix(H, cat_pairs, folds, seed):
    """KxK held-out AUC. Diagonal: cross-validation WITHIN the category, axis
    refitted on training folds only. Off-diagonal: axis fitted on ALL of A,
    evaluated on ALL of B, which are disjoint.

    The diagonal is the representational proxy for knowledge, and the threshold
    that restricts to known relations reads it."""
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


# =====================================================================
#  extraction
# =====================================================================
@torch.no_grad()
def collect(model, tok, texts, arch, device, batch, need_all_blocks):
    """One forward pass per batch. Returns the full residual stack when the
    caller needs every block, otherwise only what is asked for, plus the
    attention and FFN contributions at every block when required.

    Hooking every block costs memory proportional to n_sentences times n_blocks
    times d; hooking one costs a fraction of that. --all-layers and
    --flip-layers are the only options that need everything."""
    layers = list(range(arch.n_blocks)) if need_all_blocks else []
    caps = [BlockCapture(model, arch, b) for b in layers]
    for c in caps:
        c.__enter__()
    R, A, F = [], [], []
    try:
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            R.append(torch.stack([h[:, -1, :].float().cpu()
                                  for h in out.hidden_states], 1))    # [B, L+1, d]
            if caps:
                A.append(torch.stack([c.attn() for c in caps], 1))    # [B, L, d]
                F.append(torch.stack([c.ffn() for c in caps], 1))
            print("\r  [extract] %d/%d" % (min(s + batch, len(texts)), len(texts)),
                  end="", flush=True)
        print()
    finally:
        for c in caps:
            c.__exit__(None, None, None)
    H_resid = torch.cat(R, 0)
    H_attn = torch.cat(A, 0) if A else None
    H_ffn = torch.cat(F, 0) if F else None
    return H_resid, H_attn, H_ffn


def gate_all_blocks(H_resid, H_attn, H_ffn, tol=1e-3, report=True):
    """The additive identity across every block.

    THE VERDICT IS THE MEDIAN OVER ALL (sentence, block) PAIRS, not the maximum
    and not the worst block. This is not a softer criterion, it is the correct
    one, and the reason is arithmetic: where a block adds almost nothing to the
    residual the delta is near zero, so the relative error divides by a tiny
    number and explodes even when the reconstruction is perfect in absolute
    terms. On Qwen2.5-1.5B block 27 is exactly such a block, and the paper
    already declares it excluded from interpretation.

    Taking the worst block instead would abort on a healthy model because of
    one degenerate layer. The per-block table is still printed, next to the
    median norm of that block's delta, so a large per-block error can be seen
    for what it is: a small denominator, not a broken decomposition."""
    nB = H_attn.shape[1]
    delta = H_resid[:, 1:nB + 1, :] - H_resid[:, 0:nB, :]
    rel = ((H_attn + H_ffn - delta).norm(dim=-1)
           / delta.norm(dim=-1).clamp_min(1e-8))              # [N, nB]
    med = float(rel.median())
    if report:
        per = rel.median(dim=0).values
        dn = delta.norm(dim=-1).median(dim=0).values
        bad = [b for b in range(nB) if float(per[b]) > tol]
        print("[identity gate] median over all (sentence, block) = %.2e   %s"
              % (med, "OK" if med < tol else "FAILED"))
        if bad:
            print("  blocks above tolerance, with the median norm of their delta:")
            for b in bad:
                print("    block %-3d rel %.2e   |delta| %.3e%s"
                      % (b, float(per[b]), float(dn[b]),
                         "   <- near-zero delta, ratio inflated" if float(dn[b]) < 1e-2 else ""))
    return med


# =====================================================================
#  one bundle
# =====================================================================
def build_and_export(model_name, peak, write_layer, early_block, proto, a):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print()
    print("=" * 88)
    print("[dictionary] %s   peak %d   FFN write %d   early %d"
          % (model_name, peak, write_layer, early_block))
    print("=" * 88)

    tag = model_name.split("/")[-1].replace(".", "").replace("-", "_")
    comp_sfx = "" if a.component == "resid" else "_%s" % a.component
    conv_sfx = "" if proto.suffix == "." else "_nodot"
    out = os.path.join(a.out_dir, "truth_dictionary_%s%s%s.pt" % (tag, comp_sfx, conv_sfx))
    if os.path.exists(out) and not a.force:
        print("[skip] %s already exists and would not be saved, so the expensive" % out)
        print("       extraction would be wasted. Use --force, or --out-dir.")
        return None

    ps = counterfact_by_relation(proto, k=a.k_relations, n_per=a.pairs_per_relation,
                                 whitelist=RELATION_WHITELIST,
                                 local_file=a.file_counterfact)
    cat_pairs = ps.by_category()
    print("[convention] suffix %r   example: %r" % (proto.suffix, ps.items[0]))
    print("             last token is %s in the two sentences of a pair"
          % ("IDENTICAL" if proto.suffix else "DIFFERENT"))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, use_safetensors=True,
        trust_remote_code=False).to(device)
    model.eval()
    arch = describe(model)
    print("[model] %s on %s (float32)" % (model_name, device))
    for line in arch.summary():
        print("  " + line)

    need_all = a.component != "resid" or a.flip_layers or a.all_layers
    H_resid, H_attn, H_ffn = collect(model, tok, ps.items, arch, device,
                                     a.batch, need_all or True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    med = gate_all_blocks(H_resid, H_attn, H_ffn)
    if med > 1e-3:
        print("  ABORT: the additive decomposition does not hold; not exporting.")
        return None

    # WHICH STREAM the axes are fitted on. Index convention: H_resid[:, b+1, :]
    # is the OUTPUT of block b, while H_attn[:, b, :] and H_ffn[:, b, :] are the
    # vectors that block b ADDS.
    if a.component == "resid":
        H_peak, H_early = H_resid[:, peak + 1, :], H_resid[:, early_block + 1, :]
    elif a.component == "attn":
        H_peak, H_early = H_attn[:, peak, :], H_attn[:, early_block, :]
    else:
        H_peak, H_early = H_ffn[:, peak, :], H_ffn[:, early_block, :]
    print("[component] axes fitted on %r at block %d (early control: same stream "
          "at block %d)" % (a.component, peak, early_block))

    cs, cos_peak, axes_d = axis_cosine_matrix(H_peak, cat_pairs)
    _, cos_early, _ = axis_cosine_matrix(H_early, cat_pairs)
    _, transfer = transfer_matrix(H_peak, cat_pairs, a.folds, a.seed)
    axes = torch.stack([unit(axes_d[c]) for c in cs], 0)
    t_global = unit(fit_axis(H_peak, ps.pidx)["v1"])
    t_global_resid = unit(fit_axis(H_resid[:, peak + 1, :], ps.pidx)["v1"])
    cos_vs_resid = float(torch.dot(t_global, t_global_resid))
    if a.component != "resid":
        print("[control] cos(global axis on %r, global axis on residual) = %+.3f"
              % (a.component, cos_vs_resid))
        print("          near +/-1 means this view rediscovers the residual axis; "
              "low means it is a different direction. No verdict is printed.")

    labels = [pr.category for pr in ps.pairs]
    Df = torch.stack([unit(H_ffn[it, write_layer, :] - H_ffn[iff, write_layer, :])
                      for it, iff in ps.pidx], 0)
    De = torch.stack([unit(H_early[it] - H_early[iff]) for it, iff in ps.pidx], 0)
    centroids = torch.stack(
        [unit(Df[[i for i in range(len(ps.pidx)) if labels[i] == c]].mean(0)) for c in cs], 0)
    dec_f = decoding_with_null(Df, labels, a.folds, a.seed, a.perm)
    dec_e = decoding_with_null(De, labels, a.folds, a.seed, a.perm)

    meta = dict(model=model_name, component=a.component,
                cos_global_vs_resid=cos_vs_resid,
                peak_block=peak, write_layer=write_layer, early_block=early_block,
                protocol=proto.to_dict(), truthprobe_version=__version__,
                seed=a.seed, folds=a.folds,
                k_relations=len(cs), pairs_per_relation=a.pairs_per_relation,
                templates={c: "" for c in cs},   # non esposto dal PairSet
                decoding=dict(delta_f=dec_f, lexical=dec_e),
                identity_check_median=med)
    bundle = dict(cats=cs, axes=axes, t_global=t_global, write_centroids=centroids,
                  cos_peak=cos_peak, cos_early=cos_early, transfer=transfer, meta=meta)

    if a.all_layers:
        nH = H_resid.shape[1]
        print("[all-layers] cosine matrix at each of the %d hidden levels "
              "(reuses cached states, no extra passes)..." % nH)
        bundle["cos_by_layer"] = torch.stack(
            [axis_cosine_matrix(H_resid[:, h, :], cat_pairs)[1] for h in range(nH)], 0)
        meta.update(all_layers=True, n_hidden=nH, peak_hidden=peak + 1)

    if a.flip_layers:
        print("[flip-layers] contribution gap on the fixed peak axis, per block...")
        fc = frame_curves({"ffn": H_ffn, "attn": H_attn},
                          H_resid[:, peak + 1, :], ps.pidx, a.folds, a.seed)
        bundle["flip"] = dict(gap_ffn=fc["ffn"]["gap"], dprime_ffn=fc["ffn"]["dprime"],
                              gap_attn=fc["attn"]["gap"], dprime_attn=fc["attn"]["dprime"],
                              axis_block=peak, flip_block=peak + 1,
                              n_blocks=H_ffn.shape[1])
        dpf = fc["ffn"]["dprime"]
        if peak + 1 < len(dpf):
            print("             d' ffn @peak %d = %+.2f   @block %d = %+.2f (%s)"
                  % (peak, dpf[peak], peak + 1, dpf[peak + 1],
                     "ANTI-truth flip" if dpf[peak + 1] < 0 else "still pro-truth"))

    os.makedirs(a.out_dir, exist_ok=True)
    torch.save(bundle, out)
    with open(out.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, cats=cs,
                       cos_peak=[[round(float(x), 3) for x in r] for r in cos_peak],
                       cos_early=[[round(float(x), 3) for x in r] for r in cos_early],
                       transfer=[[round(float(x), 3) for x in r] for r in transfer]),
                  f, indent=2)
    print()
    print("[saved] %s  (+ .json human summary)" % out)
    print("  cats: %s" % cs)
    print("  axes [K,d] = %s   t_global [d] = %s" % (tuple(axes.shape), tuple(t_global.shape)))
    print("  decoding: Delta f %.2f%% vs lexical %.2f%% (chance %.0f%%)"
          % (100 * dec_f["acc"], 100 * dec_e["acc"], 100 * dec_f["chance"]))
    print("  load with:  b = torch.load(path); scores = states @ b['axes'].T")
    if a.verify:
        verify_against_reference(model_name, bundle, proto, a.verify_tol)
    return bundle


# =====================================================================
#  jobs and main
# =====================================================================
def resolve_jobs(a):
    if a.models:
        jobs = []
        for name in a.models:
            if name in MODELS:
                jobs.append((name, dict(MODELS[name])))
            elif a.peak is not None:
                jobs.append((name, dict(peak=a.peak,
                                        write_layer=(a.write_layer if a.write_layer
                                                     is not None else a.peak + 1),
                                        early_block=a.early_block)))
            else:
                sys.exit("[error] %r is not in the MODELS registry. Add it to the "
                         "USER CONFIG block, or pass --peak and --write-layer "
                         "(measure the peak with a per-layer signal scan first: "
                         "it is model-specific and must never be guessed)." % name)
        return jobs
    return [(n, dict(v)) for n, v in MODELS.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--models", nargs="*", default=None,
                    help="models to run (default: the whole MODELS registry)")
    ap.add_argument("--peak", type=int, default=None,
                    help="peak BLOCK, for a model not in the registry")
    ap.add_argument("--write-layer", type=int, default=None, help="default: peak+1")
    ap.add_argument("--early-block", type=int, default=EARLY_BLOCK)
    ap.add_argument("--k-relations", type=int, default=K_RELATIONS)
    ap.add_argument("--pairs-per-relation", type=int, default=PAIRS_PER_RELATION)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8,
                    help="batch size. Safe in float32: 2e-06 relative error "
                         "against single-sentence extraction. Use 1 for bit-exact "
                         "reproduction of the pre-conversion tool.")
    ap.add_argument("--component", default="resid", choices=["resid", "attn", "ffn"],
                    help="WHICH STREAM the per-category axes are fitted on")
    ap.add_argument("--suffix", default=".",
                    help="what is appended after the target. '.' makes the LAST "
                         "token identical in both sentences of a pair, so token "
                         "identity leaves the measurement. '' is the historical "
                         "convention; the two are NOT comparable and the filename "
                         "records which was used.")
    ap.add_argument("--all-layers", action="store_true",
                    help="also store the cosine matrix at every hidden level")
    ap.add_argument("--flip-layers", action="store_true",
                    help="also store the per-block contribution gap on the peak axis")
    ap.add_argument("--verify", action="store_true",
                    help="compare with the archived K=8 matrices")
    ap.add_argument("--verify-tol", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="dizionari")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    a = ap.parse_args()

    proto = (CANONICAL if a.suffix == "." else LEGACY_DICT).with_(seed=a.seed)
    print()
    print("[library] truthprobe %s" % __version__)
    print("[protocol] %s" % proto.label())

    jobs = resolve_jobs(a)
    print("[jobs] %d model(s): %s" % (len(jobs), ", ".join(n for n, _ in jobs)))
    for name, cfg in jobs:
        try:
            build_and_export(name, cfg["peak"], cfg["write_layer"],
                             cfg.get("early_block", EARLY_BLOCK), proto, a)
        except Exception as e:
            print("[error] %s: %s" % (name, e))
            if a.models and len(a.models) == 1:
                raise
    print()
    print("[done] dictionaries in %s" % a.out_dir)
    print()


if __name__ == "__main__":
    main()
