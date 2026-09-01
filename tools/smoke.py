# -*- coding: utf-8 -*-
"""
smoke.py  --  does this model work with this library, before spending an hour?

Runs every stage of the pipeline on a handful of pairs and reports what passed,
what failed and how long each took. Nothing here is a scientific measurement:
the numbers come from too few pairs to mean anything, and are printed only so
that a stage which returns nonsense is visible. What IS meaningful is the
gates, which are exact identities and hold at any sample size.

WHY THIS EXISTS
A full campaign takes tens of minutes and can fail at the last stage for a
reason that was knowable at the first. The expensive failures are architectural:
a decomposition that does not hold, a wiring that points at the wrong module, a
model that does not fit in memory. All three show up in one minute here.

WHAT IT CHECKS, IN ORDER
  loading         weights, dtype, where the model actually sits
  architecture    what was detected, and what the wiring had to be told
  extraction      one forward pass, hooks attached
  gate: block     attention + FFN reconstruct the residual delta
  gate: heads     the per-head contributions sum back to attention
  gate: FFN       g * u reproduces the hidden state fed to the down projection
  MoE             experts, router, and the one number that decides whether a
                  per-expert study is possible at all
  geometry        the axis fits, separates, and its calibration is finite
  arrangement     per-category axes, gauge margins, eigengap
  memory          peak allocation, so a longer run can be sized

WHAT A FAILING GATE MEANS
Not that the library is broken: that the decomposition does not describe THIS
model. bfloat16 fails the identity gate by construction. A sandwich-norm model
hooked at the module instead of the post-norm fails it too. Either way the
numbers downstream would be meaningless, which is why the gate refuses instead
of warning.

    python smoke.py --model modelli/OLMo-2-1B --peak 8
    python smoke.py --model modelli/OLMoE-1B-7B-0125 --peak 8 --moe
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

from truthprobe import CANONICAL, __version__
from truthprobe.data import counterfact_by_relation
from truthprobe.geometry import unit, fit_axis, project_fields, cosine_matrix
from truthprobe.hooks import (Wiring, describe, BlockCapture,
                              identity_gate, head_gate)
from truthprobe.stats import (auc_score, kfold_pairs, project_and_score,
                              consensus_gauge, eigengap, frame_gap)


class Report:
    """Every stage records what it found, so a failure late in the run does not
    erase what the earlier stages established."""

    def __init__(self):
        self.rows = []
        self.t0 = time.time()

    def stage(self, name):
        self.t = time.time()
        print()
        print("--- %s %s" % (name, "-" * max(0, 56 - len(name))))
        return self

    def ok(self, what, value="", note=""):
        self.rows.append((True, what, str(value)))
        print("  [ok]   %-34s %s" % (what, value))
        if note:
            print("         %s" % note)

    def no(self, what, value="", note=""):
        self.rows.append((False, what, str(value)))
        print("  [FAIL] %-34s %s" % (what, value))
        if note:
            print("         %s" % note)

    def info(self, what, value=""):
        print("  %-41s %s" % (what, value))

    def done(self):
        dt = time.time() - self.t
        print("  (%.1f s)" % dt)
        return dt

    def failures(self):
        return [w for ok, w, _ in self.rows if not ok]


def mem():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True, help="name or local folder")
    ap.add_argument("--peak", type=int, default=None,
                    help="block to hook. Any block works for a smoke test: the "
                         "gates are identities and do not care where they are "
                         "checked. Default: the middle of the stack.")
    ap.add_argument("--moe", action="store_true",
                    help="also exercise the mixture-of-experts path")
    ap.add_argument("--k-relations", type=int, default=4)
    ap.add_argument("--pairs-per-relation", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="float32 is what the identity gate needs. Run with "
                         "bfloat16 once to SEE the gate fail: that failure is "
                         "the gate working.")
    ap.add_argument("--device-map", default=None,
                    help="'auto' spreads the weights across GPU and RAM. Needed "
                         "when the model does not fit on the card in float32.")
    ap.add_argument("--max-vram", default=None,
                    help="cap for device_map, e.g. 10GiB")
    ap.add_argument("--probe-heads", type=int, default=2,
                    help="how many heads to keep. Keeping all of them costs "
                         "n_heads times the residual per sentence, which on a "
                         "large model fills memory before the weights do. Two "
                         "are enough to verify the decomposition; 0 keeps all.")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--out", default="smoke.json")
    a = ap.parse_args()

    R = Report()
    proto = CANONICAL
    print()
    print("=" * 64)
    print("SMOKE TEST   truthprobe %s" % __version__)
    print("=" * 64)
    print("[model] %s" % a.model)
    print("[note]  the numbers here are NOT measurements: too few pairs.")
    print("        The gates are exact identities and hold at any size.")

    # ---------------- data ----------------
    R.stage("DATA")
    try:
        ps = counterfact_by_relation(proto, k=a.k_relations,
                                     n_per=a.pairs_per_relation,
                                     local_file=a.file_counterfact, verbose=False)
        R.ok("pairs built", "%d pairs, %d relations"
             % (len(ps.pidx), len(ps.categories)))
        R.info("example", repr(ps.items[0]))
    except Exception as e:
        R.no("pairs built", type(e).__name__, str(e)[:70])
        sys.exit(1)
    R.done()

    # ---------------- loading ----------------
    R.stage("LOADING")
    t = time.time()
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = dict(torch_dtype=(torch.float32 if a.dtype == "float32"
                               else torch.bfloat16),
                  use_safetensors=True, trust_remote_code=False)
        if a.device_map:
            kw["device_map"] = a.device_map
            if a.max_vram:
                kw["max_memory"] = {0: a.max_vram, "cpu": "100GiB"}
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
        if not a.device_map:
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        R.ok("weights loaded", "%.1f s" % (time.time() - t))
        devs = {str(p.device) for p in model.parameters()}
        R.info("parameters live on", ", ".join(sorted(devs)))
        n = sum(p.numel() for p in model.parameters())
        R.info("parameters", "%.2f B  (%.1f GB in %s)"
               % (n / 1e9, n * (4 if a.dtype == "float32" else 2) / 1024 ** 3,
                  a.dtype))
    except Exception as e:
        R.no("weights loaded", type(e).__name__, str(e)[:70])
        print()
        if ("\\" in a.model or "/" in a.model) and not os.path.isdir(a.model):
            print("  %r looks like a PATH, and that folder does not exist." % a.model)
            print("  A name with a separator is not a valid repository id, so it")
            print("  was tried on the Hub and rejected there. Use the full path,")
            print("  or run from the folder that contains it.")
        elif "memory" in str(e).lower() or "alloc" in str(e).lower():
            print("  Out of memory: --device-map auto --max-vram 10GiB --batch 1")
        sys.exit(1)
    R.done()

    # ---------------- architecture ----------------
    R.stage("ARCHITECTURE")
    wiring = None
    if a.moe:
        L0 = getattr(getattr(model, "model", model), "layers")[0]
        mlp = getattr(L0, "mlp", None) or getattr(L0, "block_sparse_moe", None)
        gate = next((n for n in ("gate", "router", "gate_proj")
                     if hasattr(mlp, n)), None)
        pack = getattr(mlp, "experts", None)
        packed = pack is not None and hasattr(pack, "gate_up_proj")
        wiring = Wiring(
            ffn_write=lambda L: getattr(L, "mlp", None) or L.block_sparse_moe,
            router=(lambda L, g=gate: getattr(getattr(L, "mlp", None)
                                              or L.block_sparse_moe, g))
            if gate else None,
            expert_pack=(lambda L: getattr(getattr(L, "mlp", None)
                                           or L.block_sparse_moe, "experts"))
            if packed else None,
            experts=(lambda L: getattr(getattr(L, "mlp", None)
                                       or L.block_sparse_moe, "experts"))
            if (pack is not None and not packed) else None,
            notes="router at %r, experts %s" % (gate, "packed" if packed
                                                else "as modules"))
        R.info("wiring supplied", wiring.notes)
    try:
        arch = describe(model, wiring=wiring)
        for line in arch.summary():
            R.info("", line)
        R.ok("architecture described", arch.family)
    except Exception as e:
        R.no("architecture described", type(e).__name__, str(e)[:70])
        sys.exit(1)
    peak = a.peak if a.peak is not None else arch.n_blocks // 2
    if not 0 <= peak < arch.n_blocks:
        sys.exit("block %d out of range (0..%d)" % (peak, arch.n_blocks - 1))
    R.info("block to hook", peak)
    R.done()

    device = next(model.parameters()).device
    res = dict(model=a.model, truthprobe_version=__version__, block=peak,
               dtype=a.dtype, architecture=arch.to_dict())

    # ---------------- extraction ----------------
    R.stage("EXTRACTION AND GATES")
    t = time.time()
    H_res, H_pre, H_att, H_ffn, H_head = [], [], [], [], []
    g_block, g_head, g_ffn, routes = [], [], [], []
    packed, expert_probe, router_dirs = None, None, None
    try:
        with torch.no_grad(), BlockCapture(model, arch, peak) as cap:
            for s in range(0, len(ps.items), a.batch):
                enc = tok(ps.items[s:s + a.batch], return_tensors="pt",
                          padding=True).to(device)
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states
                H_res.append(hs[peak + 1][:, -1, :].float().cpu())
                H_pre.append(hs[peak][:, -1, :].float().cpu())
                at, ff = cap.attn(), cap.ffn()
                H_att.append(at)
                H_ffn.append(ff)
                sel = (None if a.probe_heads <= 0
                       else range(min(a.probe_heads, arch.n_heads)))
                H_head.append(cap.heads(which=sel))
                g_block.append(identity_gate(hs[peak][:, -1, :].float().cpu(),
                                             hs[peak + 1][:, -1, :].float().cpu(),
                                             at, ff)[0])
                if sel is None:
                    g_head.append(head_gate(cap.heads(), at, cap.bias_term())[0])
                gg, _ = cap.swiglu_gate()
                if gg == gg:
                    g_ffn.append(gg)
                if a.moe:
                    if packed is None:
                        # letto DENTRO il with: fuori gli hook sono rimossi, e
                        # anche se questi metodi leggono solo pesi, dipendere da
                        # un contesto chiuso e' il genere di fragilita' che
                        # rompe alla prima modifica
                        packed = cap.experts_packed()
                        if packed is not None:
                            router_dirs = cap.router_directions(packed)
                            try:
                                expert_probe = cap.expert_contribution(
                                    packed, 0, hs[peak][:, -1, :].float().cpu())
                            except Exception as e:
                                expert_probe = e
                    try:
                        routes.append(cap.routing())
                    except Exception:
                        pass
        el = time.time() - t
        R.ok("forward passes", "%d sentences in %.1f s (%.2f s each)"
             % (len(ps.items), el, el / len(ps.items)))
    except Exception as e:
        R.no("forward passes", type(e).__name__, str(e)[:70])
        traceback.print_exc(limit=2)
        sys.exit(1)

    H_res = torch.cat(H_res, 0)
    H_pre = torch.cat(H_pre, 0)
    H_att = torch.cat(H_att, 0)
    H_ffn = torch.cat(H_ffn, 0)
    H_head = torch.cat(H_head, 0)
    med = lambda x: float(torch.tensor(x).median()) if x else float("nan")
    gb, gh, gf = med(g_block), med(g_head), med(g_ffn)

    (R.ok if gb < 1e-4 else R.no)(
        "gate: attn + ffn = residual delta", "%.2e" % gb,
        "" if gb < 1e-4 else "in bfloat16 this is expected; on a sandwich-norm "
                             "model check the hook target")
    if gh != gh:
        R.info("gate: heads sum to attention",
               "skipped: only %d of %d heads kept, so the sum cannot reconstruct "
               "the whole. Use --probe-heads 0 to check it." % (a.probe_heads,
                                                                arch.n_heads))
    else:
        (R.ok if gh < 1e-4 else R.no)(
            "gate: heads sum to attention", "%.2e" % gh,
            "" if gh < 1e-4 else "a bias on the output projection is not "
                                 "attributable to any head and must be counted "
                                 "separately")
    if gf == gf:
        (R.ok if gf < 1e-4 else R.no)(
            "gate: g * u = hidden state", "%.2e" % gf,
            "" if gf < 1e-4 else "the feed-forward may not be gated, or the "
                                 "activation is applied as a function with no "
                                 "module to hook")
    else:
        R.info("gate: g * u = hidden state", "not a gated feed-forward")
    res.update(gate_block=gb, gate_heads=gh, gate_ffn=gf)
    tex = R.done()

    # ---------------- MoE ----------------
    if a.moe:
        R.stage("MIXTURE OF EXPERTS")
        if packed is None:
            R.no("expert weights", "not found",
                 "the wiring may point at the wrong module")
        else:
            R.ok("expert weights", "%d experts, inner dim %d"
                 % (packed["n_experts"], packed["d_inter"]))
            if isinstance(expert_probe, Exception):
                R.no("expert contribution computed", type(expert_probe).__name__)
            elif expert_probe is not None:
                R.ok("expert contribution computed", tuple(expert_probe.shape),
                     "computed from weights, so it works for experts the router "
                     "never selected: that is what makes a counterfactual possible")
            rd = router_dirs
            if rd:
                C = rd["cosines"]
                off = C[~torch.eye(C.shape[0], dtype=torch.bool)]
                R.ok("router directions", "%d directions" % rd["n_experts"],
                     "median cosine between them %+.3f: these rows are directions "
                     "in the residual stream and can be compared with the "
                     "category axes directly" % float(off.median()))
        if routes:
            ex = torch.cat([r["experts"] for r in routes], 0)
            wt = torch.cat([r["weights"] for r in routes], 0)
            src = routes[0].get("source", "?")
            R.ok("routing captured", "%d sentences, source %r" % (ex.shape[0], src),
                 "'model' means the model's own scores; 'recomputed' means they "
                 "were derived here and may not match a non-renormalising router")
            R.info("weights sum to", "%.3f" % float(wt[0].sum()))
            same = sum(1 for i, j in ps.pidx
                       if i < ex.shape[0] and j < ex.shape[0]
                       and set(ex[i].tolist()) == set(ex[j].tolist()))
            R.ok("pairs routed to the SAME experts",
                 "%d/%d  %.0f%%" % (same, len(ps.pidx), 100 * same / len(ps.pidx)),
                 "this is the number that decides whether a per-expert study is "
                 "possible: the router routes PER TOKEN, and with the period "
                 "convention the read token is identical in both sentences. If "
                 "this is near 100%, a per-expert decomposition of minimal pairs "
                 "has nothing to separate.")
            res["same_experts"] = same / len(ps.pidx)
        else:
            R.no("routing captured", "nothing",
                 "the router hook did not fire: check the wiring")
        R.done()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- geometry ----------------
    R.stage("GEOMETRY")
    try:
        ax = fit_axis(H_res, ps.pidx)
        pf = project_fields(H_res, ax)
        finite = all(torch.isfinite(pf[k]).all() for k in ("Re", "Im", "b", "m"))
        (R.ok if finite else R.no)("axis fits, projections finite",
                                   "s1 %.2f  r %.2f" % (ax["s1"], ax["r"]))
        aucs = []
        for tr, te in kfold_pairs(len(ps.pidx), 3, 0):
            axf = fit_axis(H_res, [ps.pidx[p] for p in tr])
            I = [i for p in te for i in ps.pidx[p]]
            aucs.append(auc_score(project_and_score(H_res[I], axf),
                                  torch.tensor([1, 0] * len(te))))
        R.ok("held-out AUC runs", "%.3f" % (sum(aucs) / len(aucs)),
             "too few pairs to mean anything: only its being finite matters here")
        v1 = unit(ax["v1"])
        tt = [i for i, _ in ps.pidx]
        ff = [j for _, j in ps.pidx]
        ga = frame_gap(H_att[tt], H_att[ff], v1, contrib_block=peak, frame_block=peak)
        gfn = frame_gap(H_ffn[tt], H_ffn[ff], v1, contrib_block=peak, frame_block=peak)
        R.ok("frame gap computes", "attn %+.2f  ffn %+.2f"
             % (ga["dprime"], gfn["dprime"]))
        if ga["warning"]:
            R.info("", "containment flagged, as it should be: the frame here "
                       "contains what it measures")
        res.update(auc=sum(aucs) / len(aucs), dprime_attn=ga["dprime"],
                   dprime_ffn=gfn["dprime"])
    except Exception as e:
        R.no("geometry", type(e).__name__, str(e)[:70])
        traceback.print_exc(limit=2)
    R.done()

    # ---------------- arrangement ----------------
    R.stage("ARRANGEMENT")
    try:
        cp = ps.by_category()
        cats = ps.categories
        axes = torch.stack([unit(fit_axis(H_res, cp[c])["v1"]) for c in cats], 0)
        C, _ = cosine_matrix(axes)
        s, marg, thin = consensus_gauge(C)
        eg = eigengap(C)
        R.ok("per-category axes", "%d axes" % len(cats))
        R.info("median off-diagonal cosine",
               "%+.3f" % float(C[~torch.eye(len(cats), dtype=torch.bool)].median()))
        R.info("gauge margin, median", "%.3f" % float(marg.median()))
        R.info("unsigned categories", [cats[i] for i in thin] or "none")
        R.info("relative eigengap", "%.3f" % eg["rel"])
        A = torch.stack([unit(fit_axis(H_head[:, h, :], ps.pidx)["v1"])
                         for h in range(H_head.shape[1])], 0)
        R.ok("per-head axes", "%d heads" % A.shape[0])
        R.info("cosine with the residual axis",
               " ".join("%+.2f" % float(A[h] @ v1) for h in range(A.shape[0])))
        res["eigengap"] = eg["rel"]
    except Exception as e:
        R.no("arrangement", type(e).__name__, str(e)[:70])
    R.done()

    # ---------------- verdict ----------------
    peak_mem = mem()
    print()
    print("=" * 64)
    fails = R.failures()
    if fails:
        print("FAILED: %s" % ", ".join(fails))
        print()
        print("A failing gate does not mean the library is broken: it means the")
        print("decomposition does not describe this model as configured. Read the")
        print("note beside the failure before changing anything.")
    else:
        print("all stages passed")
    print()
    print("total time      %.1f s" % (time.time() - R.t0))
    print("extraction      %.2f s per sentence" % (tex / max(len(ps.items), 1)))
    if peak_mem:
        print("peak GPU memory %.2f GB" % peak_mem)
    n_full = 33 * 60 * 2
    print()
    print("SIZING A FULL RUN")
    print("  a K=33, n=60 campaign is %d sentences: about %.0f minutes of"
          % (n_full, n_full * (tex / max(len(ps.items), 1)) / 60))
    print("  extraction at this rate, per model and per configuration.")
    if peak_mem:
        print("  memory scales with the batch, not with the number of pairs.")
    res.update(peak_gpu_gb=peak_mem, sec_per_sentence=tex / max(len(ps.items), 1),
               failures=fails)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print()
    print("written: %s" % a.out)
    print()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
