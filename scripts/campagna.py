#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""campagna.py  --  the whole core campaign on ONE model, in one file.

Replaces the assembly line (truth_probe -> anatomy -> flip_consolidate ->
categories -> crea_dizionario -> reorient_gauge) with a single file written on
top of `truthprobe`. The mathematics is not here: it lives in the library,
where it is already verified against the canonical code function by function.
What is here is only the DECISIONS: what to measure, in what order, under which
protocol.

WHY IT EXISTS
The canonical chain breaks on models outside the known set, because every file
implements the decomposition assuming a pre-norm architecture. On OLMo 2, which
normalises on the way out, the identity check comes out at 6.65 and the stage
aborts, as it should. `truthprobe.hooks` DESCRIBES the architecture instead of
assuming it, and verifies itself with a gate. This file is that chain rebuilt on
those foundations.

TWO EXTRACTIONS, NOT FIVE
The canonical chain passes over the model five times to read the same states.
Here there are two: one on the flat sample of 250 pairs (signal, anatomy, flip)
and one on the grouped sample of K x n (categories, dizionario). Everything else
is loops over tensors already in memory. It is the principle flip_consolidate
already states: extraction is the only expensive step.

HOUSE RULES, UNCHANGED
  - nothing is measured until <outdir>/00_predizioni.txt exists and is not empty
  - the behavioural know-rate runs BEFORE any geometry
  - YOU read the peak from the log and write it into <outdir>/landmarks.json
  - no concluding verdict: the tables are to be read, not summarised
  - provenance travels inside every artifact produced

UNITS
The signal log prints a hidden-state LEVEL (0 = embedding, b+1 = output of
block b). landmarks.json wants a BLOCK: peak = best_level - 1. The anatomy
stage prints "peak residual layer = block N" directly.

    python campagna.py --model <path> --stages behav,signal
    # read the peak, write landmarks.json, then:
    python campagna.py --model <path> --stages rest

Companion to arXiv:2607.16741. Licensed CC BY 4.0.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from truthprobe import CANONICAL, __version__
from truthprobe.data import counterfact_flat, counterfact_by_relation
from truthprobe.geometry import unit, fit_axis, project_fields
from truthprobe.hooks import describe, BlockCapture, _layers
from truthprobe.stats import auc_score, kfold_pairs, project_and_score
from truthprobe.subspace import effective_rank
from truthprobe.protocol import Protocol

# Le otto relazioni della famiglia dizionari, fissate invece che ricalcolate
# come "top-8 per coppie uniche". Il top-8 dipende dal dataset e non dal
# modello, quindi in teoria coincide sempre; fissarlo rende il bundle
# confrontabile anche se il dataset cambia sotto i piedi.
WHITELIST_8 = ["P103", "P1412", "P176", "P27", "P30", "P37", "P413", "P495"]

STAGES = ["behav", "signal", "anatomy", "flip", "frames", "ablazione",
          "geometria", "categories", "dizionario", "gauge"]
DOPO_PICCO = {"anatomy", "flip", "frames", "ablazione", "geometria",
              "categories", "dizionario", "gauge"}


# =====================================================================
#  cancelli
# =====================================================================
def gate_predizioni(out, modello):
    """no measurements without pre-registered prediction

    """
    p = os.path.join(out, "00_predizioni.txt")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, encoding="utf-8") as f:
            return f.read()
    with open(p, "w", encoding="utf-8") as f:
        f.write(
            "# Predictions BEFORE runs for " + modello + "\n"
            "# (fill in and rerun; the gate will not measure without it)\n"
            "# 1. expected know-rate:\n"
            "# 2. peak position (block, and rationale):\n"
            "# 3. axis AUC at peak:\n"
            "# 4. at peak, does attention or FFN read/contribute more:\n"
            "# 5. FFN flip at peak+1 (yes/no, and sign):\n"
            "# 6. how many categories pass the knowledge gate:\n"
            "# 7. expected Mantel correlation against Qwen and against Llama:\n"
        )
    sys.exit("[gate] empty predictions created: fill in %s and rerun." % p)


def gate_landmarks(out, stages):
    """Il picco lo decide il ricercatore, non un argmax dentro un tool."""
    lm = os.path.join(out, "landmarks.json")
    if not any(s in DOPO_PICCO for s in stages):
        return None
    if not os.path.exists(lm):
        sys.exit("[gate] %s missing.\n"
            "  If you haven't run 'signal' on this model yet, that is the\n"
            "  first step: --stages signal. The peak is not guessed; it is\n"
            "  read from the curve and decided by you.\n"
            "  ATTENTION TO UNITS: 02_signal outputs a LEVEL,\n"
            "  landmarks.json expects a BLOCK: peak = level - 1.\n"
            "  Sanity check: 03_anatomy says 'peak residual layer = block N'\n"
            "  and that exact N should be written as is.\n"
            "  Write in %s: {'peak': N}" % (lm, lm) )
    P = int(json.load(open(lm))["peak"])
    print("[landmarks] peak = Block %d  (layer %d of log signal, "
          "write layer %d, band %d-%d)" % (P, P + 1, P + 1, P + 1, P + 3))
    return P


# =====================================================================
#  modello ed estrazione
# =====================================================================
def stima_gb(nome):
    """Quanti GB occupera' il modello in float32, letti dal config."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(nome)
        d = getattr(c, "hidden_size", 0)
        L = getattr(c, "num_hidden_layers", 0)
        di = getattr(c, "intermediate_size", 4 * d)
        V = getattr(c, "vocab_size", 0)
        n = L * (4 * d * d + 3 * d * di) + 2 * V * d
        return n * 4 / 1e9
    except Exception:
        return 0.0


def carica(nome, device, max_vram=None, offload_folder=None, tronca=None,
           dtype=torch.float32):
    """Il modello in float32, con offload ordinato se non ci sta in VRAM.

    float32 NON e' negoziabile PER LA GEOMETRIA: in bfloat16 il delta del
    residuo soffre cancellazione catastrofica (norme grandi meno differenze
    piccole) e il cancello di identita' fallisce per motivi numerici, non
    fisici. Il know-rate e' un'altra cosa: confronta completazioni greedy, non
    identita' additive, e in bfloat16 da' lo stesso risultato occupando meta'
    memoria. Per questo stage_behav passa dtype esplicitamente.

    IL TRANELLO DELLA VRAM. Su Windows il driver NVIDIA, invece di dare errore
    quando la memoria finisce, travasa in silenzio nella RAM di sistema. Il
    modello gira lo stesso, ma ogni forward attraversa il bus PCIe e il tempo
    esplode: misurato, 176 secondi contro i 2.3 di un modello che ci sta. Il
    sintomo laterale sono i buffer audio che saltano, perche' il bus e' saturo.

    Con --max-vram si passa ad accelerate, che divide i blocchi fra GPU e CPU
    in modo dichiarato invece che lasciarlo decidere al driver. Gli stage di
    questo file leggono solo attivazioni, mai pesi, quindi il dispositivo meta
    non e' un problema; se in futuro servisse leggere un peso offloadato,
    hooks.offloaded_tensor lo recupera dalla mappa di accelerate.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(nome, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=dtype, use_safetensors=True, trust_remote_code=False)
    if tronca:
        # TRONCAMENTO. Il blocco b dipende solo dai blocchi sotto di lui, quindi
        # per leggere al picco i blocchi sopra non servono. transformers
        # istanzia solo quelli chiesti e ignora i pesi degli altri: il forward
        # attraverso i blocchi tenuti e' identico a quello del modello intero,
        # non un'approssimazione. Si perdono i logit finali, quindi NON si puo'
        # misurare il know-rate su un modello troncato.
        kw["num_hidden_layers"] = tronca
        print("[load] modello troncato a %d blocks: esatto fino al blocco %d, "
              "final logit no-sense" % (tronca, tronca - 1))
    if max_vram:
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: max_vram, "cpu": "24GiB"}
        if offload_folder:
            kw["offload_folder"] = offload_folder
        m = AutoModelForCausalLM.from_pretrained(nome, **kw)
        print("[carica] offload dichiarato, tetto VRAM %s" % max_vram)
    else:
        m = AutoModelForCausalLM.from_pretrained(nome, **kw).to(device)
    return tok, m.eval()


def avvisa_taglia(nome, device, max_vram):
    """Se il modello non ci sta e non e' stato chiesto l'offload, dillo PRIMA."""
    if max_vram or device != "cuda" or not torch.cuda.is_available():
        return
    gb = stima_gb(nome)
    libera = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gb and gb > 0.85 * libera:
        print("[attenzione] il modello pesa circa %.1f GB in float32 e la GPU "
              "ne ha %.1f." % (gb, libera))
        print("             Senza --max-vram il driver travasera' in RAM di "
              "sistema e l'estrazione sara' decine di volte piu' lenta.")
        print("             Consigliato: --max-vram %dGiB" % max(4, int(libera) - 3))


@torch.no_grad()
def estrai(model, tok, items, arch, device, batch=8):
    """One forward pass per batch, with every block hooked.
 
    Returns the residual [N, L+1, d] and the two contributions [N, L, d], read
    at the final token. Hooking every block is what makes it possible to run
    the whole campaign from a single extraction.
    """
    caps = [BlockCapture(model, arch, b) for b in range(arch.n_blocks)]
    for c in caps:
        c.__enter__()
    R, A, Fq = [], [], []
    t0 = time.time()
    try:
        for s in range(0, len(items), batch):
            enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(device)
            o = model(**enc, output_hidden_states=True)
            R.append(torch.stack([h[:, -1, :].float().cpu() for h in o.hidden_states], 1))
            A.append(torch.stack([c.attn() for c in caps], 1))
            Fq.append(torch.stack([c.ffn() for c in caps], 1))
            print("\r[extract] %d/%d" % (min(s + batch, len(items)), len(items)),
                  end="", flush=True)
        print("   (%.1f s)" % (time.time() - t0))
    finally:
        for c in caps:
            c.__exit__(None, None, None)
    return torch.cat(R, 0), torch.cat(A, 0), torch.cat(Fq, 0)


def errore_identita(R, A, Fq):
    """h_{b+1} - h_b == a_b + f_b, relative error per (sentence, block).
 
    The MEDIAN, not the maximum: where a block adds almost nothing the delta is
    near zero and the ratio explodes even on an exact reconstruction.
    """
    L = A.shape[1]
    delta = R[:, 1:L + 1, :] - R[:, 0:L, :]
    rel = (A + Fq - delta).norm(dim=-1) / delta.norm(dim=-1).clamp_min(1e-6)
    return rel


# =====================================================================
#  stimatori canonici, portati alla lettera
# =====================================================================
def punteggi_tre(Hl, ax):
    """The canonical three columns: 1D, MAG, PHASE.
 
    MAG is -risk, that is (sigmoid(Re) - 0.5) * tanh(m/r): position squashed
    and reweighted by energy, not the bare modulus. It preserves the sign of
    Re, so its ordering almost always coincides with that of 1D and its AUC can
    hardly differ. PHASE is -phase_dev, the angular distance from the mean
    direction of the true class, and is the only one of the three that is not a
    monotone transform of position.
    """
    f = project_fields(Hl, ax)
    return {"1D": f["Re"], "MAG": -f["risk"], "PHASE": -f["phase_dev"]}


def auc_variante(Hl, pidx, folds, seed, variante="1D"):
    aucs = []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        ax = fit_axis(Hl, [pidx[p] for p in tr])
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]
            I += [it, iff]
            Y += [1, 0]
        aucs.append(float(auc_score(punteggi_tre(Hl[I], ax)[variante],
                                    torch.tensor(Y))))
    return sum(aucs) / len(aucs)


def null_permutazione(Hl, pidx, folds, seed, nperm, variante="1D", scegli=None):
    """Permutation Null: swap true and false, and rifit.

    Se `scegli` e' una funzione, viene applicata a ogni ripetizione per
    riprodurre anche la selezione (per esempio la scelta dello strato
    migliore): cosi' il null assorbe l'ottimismo di selezione invece di
    lasciarlo dentro la stima. E' il motivo per cui nei log canonici la media
    del null sta sopra 0.500.
    """
    rng = random.Random(seed)
    out = []
    for b in range(nperm):
        o = [1 if rng.random() < 0.5 else -1 for _ in pidx]
        if scegli is not None:
            out.append(scegli(o))
        else:
            aucs = []
            for tr, te in kfold_pairs(len(pidx), folds, seed):
                ax = fit_axis(Hl, [pidx[p] for p in tr], [o[p] for p in tr])
                I, Y = [], []
                for p in te:
                    it, iff = pidx[p]
                    if o[p] >= 0:
                        I += [it, iff]
                    else:
                        I += [iff, it]
                    Y += [1, 0]
                aucs.append(float(auc_score(punteggi_tre(Hl[I], ax)[variante],
                                            torch.tensor(Y))))
            out.append(sum(aucs) / len(aucs))
        print("\r[permutation] %d/%d" % (b + 1, nperm), end="", flush=True)
    print()
    return torch.tensor(out)


def probe_canonico(H, pidx, folds=5, seed=0, iters=300, lr=0.05, l2=1e-2):
    """canonical linear probe.

    Standardizza per feature con le statistiche del solo training, ha
    un'intercetta, e aggrega accumulando TUTTI i punteggi fuori campione in un
    vettore solo su cui calcola UNA AUC. L'asse invece media le AUC per fold:
    sono due aggregazioni diverse ai due lati della sottrazione che produce il
    gap, ed e' bene saperlo quando si legge quel numero.
    """
    N = H.shape[0]
    oof = torch.zeros(N)
    etichetta = torch.zeros(N, dtype=torch.long)
    visti = torch.zeros(N, dtype=torch.bool)
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        I, Y = [], []
        for p in tr:
            it, iff = pidx[p]
            I += [it, iff]
            Y += [1.0, 0.0]
        X = H[I].float()
        mu = X.mean(0)
        sd = X.std(0).clamp_min(1e-6)
        Xs = (X - mu) / sd
        y = torch.tensor(Y)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            z = Xs @ w + b
            loss = F.binary_cross_entropy_with_logits(z, y) + l2 * w.pow(2).sum()
            loss.backward()
            opt.step()
        J, YT = [], []
        for p in te:
            it, iff = pidx[p]
            J += [it, iff]
            YT += [1, 0]
        Xt = (H[J].float() - mu) / sd
        oof[J] = (Xt @ w.detach() + b.detach()).float()
        for k, i in enumerate(J):
            etichetta[i] = YT[k]
        visti[J] = True
    # l'etichetta si legge dalle coppie, non dalla parita' dell'indice: dedurla
    # dall'ordine funziona solo se items e' costruito vero-falso adiacenti, e
    # non e' una proprieta' su cui valga la pena scommettere
    idx = visti.nonzero().flatten().tolist()
    return float(auc_score(oof[idx], etichetta[idx]))


def assi_per_fold(Hax, pidx, folds, seed, orient=None):
    """The fixed axis, fit once per fold and reused across all layers."""
    out = []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        o = None if orient is None else [orient[p] for p in tr]
        out.append((fit_axis(Hax, [pidx[p] for p in tr], o), te))
    return out


def gap_su_asse(Hc, pidx, axes, orient=None):
    """Held-out class gap of a contribution, on prefit axes.
 
    The gap is in RAW units of the projection on v1, so its scale depends on
    the model. d' divides by the pooled standard deviation and is
    dimensionless. The distinction matters: a fixed floor applied to the gap is
    harsher on models with compressed states.
    """
    gaps, ds = [], []
    for ax, te in axes:
        pt, pf = [], []
        for p in te:
            it, iff = pidx[p]
            if orient is None or orient[p] >= 0:
                pt.append(it); pf.append(iff)
            else:
                pt.append(iff); pf.append(it)
        prt = Hc[pt].float() @ ax["v1"]
        prf = Hc[pf].float() @ ax["v1"]
        g = float(prt.mean() - prf.mean())
        pooled = float(torch.sqrt((prt.var() + prf.var()) / 2).clamp_min(1e-8))
        gaps.append(g); ds.append(g / pooled)
    return sum(gaps) / len(gaps), sum(ds) / len(ds)


def matrice_coseni(H, per_cat):
    cats = sorted(per_cat)
    assi = {c: fit_axis(H, per_cat[c])["v1"] for c in cats}
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            M[i, j] = float(torch.dot(assi[a], assi[b]))
    return cats, M, assi


def matrice_trasferimento(H, per_cat, folds, seed):
    """Diagonal: cross-validation within the category. Off-diagonal: axis fit on
       all of A, read on B."""
    cats = sorted(per_cat)
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        pa = per_cat[a]
        aucs = []
        for tr, te in kfold_pairs(len(pa), folds, seed):
            ax = fit_axis(H, [pa[k] for k in tr])
            I, Y = [], []
            for k in te:
                it, iff = pa[k]
                I += [it, iff]; Y += [1, 0]
            aucs.append(float(auc_score(project_and_score(H[I], ax), torch.tensor(Y))))
        M[i, i] = sum(aucs) / len(aucs)
        ax_full = fit_axis(H, pa)
        for j, b in enumerate(cats):
            if i == j:
                continue
            I, Y = [], []
            for it, iff in per_cat[b]:
                I += [it, iff]; Y += [1, 0]
            M[i, j] = float(auc_score(project_and_score(H[I], ax_full), torch.tensor(Y)))
    return cats, M


def centroide_cv(D, labels, folds, seed):
    cats = sorted(set(labels))
    n = len(labels)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    tagli = [idx[k::folds] for k in range(folds)]
    giusti = 0
    for k in range(folds):
        te = tagli[k]
        tr = [i for j in range(folds) if j != k for i in tagli[j]]
        cen = {}
        for c in cats:
            righe = [i for i in tr if labels[i] == c]
            if righe:
                cen[c] = unit(D[righe].mean(0))
        for i in te:
            v = unit(D[i])
            best = max(cen, key=lambda c: float(torch.dot(v, cen[c])))
            giusti += int(best == labels[i])
    return giusti / n


def decodifica_con_null(D, labels, folds, seed, perms):
    acc = centroide_cv(D, labels, folds, seed)
    rng = random.Random(seed)
    null = []
    for _ in range(perms):
        sh = list(labels)
        rng.shuffle(sh)
        null.append(centroide_cv(D, sh, folds, seed))
    null = sorted(null)
    media = sum(null) / len(null)
    p95 = null[int(0.95 * (len(null) - 1))]
    p = (1 + sum(1 for x in null if x >= acc)) / (len(null) + 1)
    return acc, media, p95, p


# =====================================================================
#  stage
# =====================================================================
def stage_behav(a, out):
    """verify behav any geometry.

    Corrispondenza di stringa su una completazione greedy corta: e' un limite
    inferiore rumoroso, non un'etichetta. E NON e' confrontabile fra famiglie,
    perche' eredita tokenizer e stile di completamento.
    """
    print("\n[task behav] know-rate, before geometry")
    ps = counterfact_flat(CANONICAL, max_pairs=a.n_behav, local_file=a.file_counterfact)
    dev = a.device
    # bfloat16: qui si guarda cosa il modello SCRIVE, non identita' additive.
    # Dimezza la memoria, il che su un modello da 11 GB in float32 significa
    # stare tutto in VRAM invece di ciclare i pesi da RAM a ogni token, e con
    # 200 prompt per 10 token sarebbero 2000 passaggi.
    bf = torch.bfloat16 if a.device == "cuda" else torch.float32
    ridotto = None if bf == torch.bfloat16 else a.max_vram
    tok, model = carica(a.model, dev, ridotto, a.offload_folder, dtype=bf)
    print("[behav] %s: il know-rate dont need float32 "
           % ("bfloat16" if bf == torch.bfloat16 else "float32"))
    dev = getattr(model, "device", dev)
    noti = 0
    with torch.no_grad():
        for i, pr in enumerate(ps.pairs):
            enc = tok(pr.prompt.strip(), return_tensors="pt").to(dev)
            g = model.generate(**enc, max_new_tokens=10, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            testo = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ok = pr.target_true.strip().lower() in testo.lower()
            noti += int(ok)
            if i < 3 or (i + 1) % 50 == 0:
                print("  [%s] %r -> %r   (vero: %r)"
                      % ("KNOWN  " if ok else "unknown", pr.prompt.strip(),
                         testo.strip()[:40], pr.target_true.strip()))
                print("  %d/%d  know-rate corrente %d%%"
                      % (i + 1, len(ps.pairs), round(100 * noti / (i + 1))))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    kr = noti / len(ps.pairs)
    print("\n=== KNOWN fraction: %d%% (%d/%d) ===" % (round(100 * kr), noti, len(ps.pairs)))
    print("  (string match under greedy decoding: a lower bound")
    print("   noisy, and not comparable across different families)")
    return kr


def stage_signal(a, R, pidx):
    print("\n[task signal] truth axis', probe, permutation null")
    nL = R.shape[1]
    curva = {}
    print("layer |  1D   |  MAG  | PHASE")
    for h in range(nL):
        Hl = R[:, h, :]
        riga = {v: auc_variante(Hl, pidx, a.folds, a.seed, v)
                for v in ("1D", "MAG", "PHASE")}
        curva[h] = riga
        print("   %4d | %.3f | %.3f | %.3f" % (h, riga["1D"], riga["MAG"], riga["PHASE"]))
    best = max(curva, key=lambda h: curva[h]["1D"])
    print("\nbest layer (1D, fuori campione): %d   AUC %.3f"
          % (best, curva[best]["1D"]))
    print("  -> in landmarks.json write block, it is:' %d" % (best - 1))
    print("ablation @layer %d: 1D=%.3f MAG=%.3f PHASE=%.3f"
          % (best, curva[best]["1D"], curva[best]["MAG"], curva[best]["PHASE"]))
    pr = probe_canonico(R[:, best, :], pidx, a.folds, a.seed)
    print("full linear probe @layer %d: %.3f   (geometry 1D: %.3f)"
          % (best, pr, curva[best]["1D"]))
    if a.perm:
        def scegli(o):
            vals = []
            for h in range(nL):
                Hl = R[:, h, :]
                aucs = []
                for tr, te in kfold_pairs(len(pidx), a.folds, a.seed):
                    ax = fit_axis(Hl, [pidx[p] for p in tr], [o[p] for p in tr])
                    I, Y = [], []
                    for p in te:
                        it, iff = pidx[p]
                        I += ([it, iff] if o[p] >= 0 else [iff, it])
                        Y += [1, 0]
                    aucs.append(float(auc_score(project_and_score(Hl[I], ax),
                                                torch.tensor(Y))))
                vals.append(sum(aucs) / len(aucs))
            return max(vals)
        null = null_permutazione(None, pidx, a.folds, a.seed, a.perm, scegli=scegli)
        oss = curva[best]["1D"]
        p = float((null >= oss).sum() + 1) / (len(null) + 1)
        print("permutation null: media %.3f  95pct %.3f  osservato %.3f  p=%.4f"
              % (float(null.mean()), float(null.sort().values[int(0.95 * (len(null) - 1))]),
                 oss, p))
        print("  (il null include la scelta del livello migliore, quindi la sua")
        print("   media sopra 0.500 MISURA l'ottimismo di selezione)")
    return curva, best


METRICHE = {}


def _ms(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if len(v) > 1 else 0.0
    return m, s


def stage_anatomy(a, R, A, Fq, pidx):
    """Per-layer readability of the three streams, averaged over several seeds.
 
    The seed governs the fold partition, not the sample: averaging over it
    separates the variability of the estimate from that of the model. Without
    the spread the table cannot say whether a difference between two columns is
    distinguishable from zero, and with 250 pairs the standard error is 0.022.
    """
    print("\n[task anatomy] dove nasce l'asse: attenzione o FFN?")
    print("       media su %d semi, il piu' meno e' la dispersione fra semi"
          % a.seeds)
    L = A.shape[1]
    print("block  |        resid |         attn |          ffn")
    tab, tab_sd = [], []
    for b in range(L):
        rs = [auc_variante(R[:, b + 1, :], pidx, a.folds, s) for s in range(a.seeds)]
        ats = [auc_variante(A[:, b, :], pidx, a.folds, s) for s in range(a.seeds)]
        ffs = [auc_variante(Fq[:, b, :], pidx, a.folds, s) for s in range(a.seeds)]
        (r, sr), (at, sa), (ff, sf) = _ms(rs), _ms(ats), _ms(ffs)
        tab.append((r, at, ff)); tab_sd.append((sr, sa, sf))
        print("  %4d | %.3f ±%.3f | %.3f ±%.3f | %.3f ±%.3f"
              % (b, r, sr, at, sa, ff, sf))
    pk = max(range(L), key=lambda b: tab[b][0])
    print("\n=== peak residual layer = block %d (residual AUC %.3f) ===" % (pk, tab[pk][0]))
    print("  contribute: attention AUC : %.3f ±%.3f" % (tab[pk][1], tab_sd[pk][1]))
    print("  contribute: FFN AUC        : %.3f ±%.3f" % (tab[pk][2], tab_sd[pk][2]))
    scarto = tab[pk][1] - tab[pk][2]
    disp = (tab_sd[pk][1] ** 2 + tab_sd[pk][2] ** 2) ** 0.5
    print("  gap attn - ffn      : %+.3f   (dispersione %.3f)" % (scarto, disp))
    if a.perm:
        print("\n  permutation null at peak, on both contribute:")
        for nome, H in (("attn", A[:, pk, :]), ("ffn", Fq[:, pk, :])):
            null = null_permutazione(H, pidx, a.folds, a.seed, a.perm)
            oss = tab[pk][1] if nome == "attn" else tab[pk][2]
            p = float((null >= oss).sum() + 1) / (len(null) + 1)
            q = float(null.sort().values[int(0.95 * (len(null) - 1))])
            print("    %-5s media %.3f  95pct %.3f  osservato %.3f  p=%.4f"
                  % (nome, float(null.mean()), q, oss, p))
    return tab, tab_sd, pk


def stage_flip(a, R, A, Fq, pidx, P):
    print("\n[task flip] il gap on fix axis, per block, multi-seed")
    L = A.shape[1]
    scan = list(range(max(1, P - 4), min(P + 6, L - 1) + 1))
    Hax = R[:, P + 1, :]
    per_blocco = {b: {"ga": [], "gf": [], "df": []} for b in scan}
    for sd in range(a.seeds):
        axes = assi_per_fold(Hax, pidx, a.folds, sd)
        for b in scan:
            ga, _ = gap_su_asse(A[:, b, :], pidx, axes)
            gf, df = gap_su_asse(Fq[:, b, :], pidx, axes)
            per_blocco[b]["ga"].append(ga)
            per_blocco[b]["gf"].append(gf)
            per_blocco[b]["df"].append(df)

    def ms(v):
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if len(v) > 1 else 0.0
        return m, s

    print("block  |         gap_attn |          gap_ffn |   d'_ffn | verdict (gap) | verdict (d')")
    for b in scan:
        ma, sa = ms(per_blocco[b]["ga"])
        mf, sf = ms(per_blocco[b]["gf"])
        md, sdv = ms(per_blocco[b]["df"])
        # criterio canonico: pavimento su una quantita' in unita' grezze
        v1 = ("pro-truth STABLE" if mf > 0 else "ANTI-truth STABLE") \
            if abs(mf) > max(2 * sf, 0.03) else "~ non stabile"
        # stesso criterio su d', che e' adimensionale: se i due divergono,
        # il verdetto canonico sta guardando la scala del modello
        v2 = ("pro-truth STABLE" if md > 0 else "ANTI-truth STABLE") \
            if abs(md) > max(2 * sdv, 0.05) else "~ non stabile"
        segna = "  <- flip" if b == P + 1 else ""
        print("  %4d | %+8.3f±%.3f | %+8.3f±%.3f | %+8.2f | %-14s | %-14s%s"
              % (b, ma, sa, mf, sf, md, v1, v2, segna))

    print("\n=== rotation versus erosion: mean pairwise difference of FFN ===")
    v1v = fit_axis(Hax, pidx)["v1"]
    print("blocco |   ||d|| |    lungo |    orto | cos(d,v1)")
    for b in scan:
        d = torch.stack([Fq[it, b, :] - Fq[iff, b, :] for it, iff in pidx]).mean(0)
        nd = float(d.norm())
        lungo = float(torch.dot(d, v1v))
        orto = float((d - lungo * v1v).norm())
        print("  %4d | %7.2f | %+8.3f | %7.2f | %+9.3f"
              % (b, nd, lungo, orto, lungo / max(nd, 1e-8)))
    print("  rotation predicts a SMOOTH zero-crossing of the cosine over multiple")
    print("  blocks at roughly constant ||d||; erosion predicts a sharp sign flip")
    print("  at peak+1. Read the column, the verdict is yours.")



# =====================================================================
#  frames: l'asse misurato contro righelli che lo contengono o no
# =====================================================================
def stage_frames(a, R, A, Fq, pidx, P):
    """The contribution of a block, measured against different rulers.

    THE PROBLEM. Measuring f_b against the output of block b means fitting the
    ruler on a state that CONTAINS f_b. The resulting alignment can be
    fabricated by the ruler measuring its own ingredient. It is the control
    that bites the one holding it.

    THE TWO RULERS, from the same caches:
      POST  h_{b+1} = h_b + a_b + f_b      contains f_b
      PRE   h_b + a_b                       f_b excluded

    The discrimination criterion was pre-registered. If under the PRE ruler
    the term pro to its own block remains positive, the alignment is genuine:
    the FFN is writing along the direction the stream is already building.
    If it collapses toward zero, that term was circular and the clean law
    is strictly "against the preceding state".

    The anti-term at b+1 is not at risk, because f_{b+1} is not in the states
    of block b, and must appear under both rulers.

    NO AUTOMATIC VERDICT: matrices and diagonals are printed.
    """
    L = A.shape[1]
    offsets = [int(x) for x in a.frame_offsets.split(",")]
    righelli = [P + o for o in offsets if 0 <= P + o < L]
    scan = list(range(max(1, a.frame_scan_start if a.frame_scan_start >= 0 else P - 4),
                      min(P + 6, L - 1) + 1))
    print("\n[task frames] genuine alignment or circular ruler?")
    print("  rulers at blocks %s   POST = block output (contains f_b)"
          % righelli)
    print("  PRE = h_b + a_b (f_b excluded).  scanned contribution across blocks %d-%d"
          % (scan[0], scan[-1]))
    if getattr(a, "_parallelo", False):
        print("  WARNING: parallel blocks. The state h_b + a_b is not a state")
        print("  that the model traverses: attention and FFN read from the same")
        print("  input. The PRE ruler remains computable but is not a frame")
        print("  that exists in the model, and should be read as a construct.")

    fuori = {}
    for comp_nome, Hc in (("ffn", Fq), ("attn", A)):
        for tipo in ("post", "pre"):
            print("\n=========  RULER %s   component %s  ========="
                  % (tipo.upper(), comp_nome.upper()))
            assi = {}
            for b in righelli:
                base = R[:, b + 1, :] if tipo == "post" else R[:, b, :] + A[:, b, :]
                assi[b] = assi_per_fold(base, pidx, a.folds, a.seed)
            print("%5s |" % "bloc" + "".join(" v1@%3d" % b for b in righelli))
            print("-" * (7 + 7 * len(righelli)))
            mat = {}
            for Lb in scan:
                riga = [gap_su_asse(Hc[:, Lb, :], pidx, assi[b])[1] for b in righelli]
                mat[Lb] = riga
                m = "<- p" if Lb == P else ("<- p+1" if Lb == P + 1 else "")
                print("%5d |" % Lb + "".join(" %+6.2f" % v for v in riga) + "  " + m)
            print("\n  diagonale: per ogni righello b, il d' della componente A b e A b+1")
            print("  %8s | %9s | %10s | %s" % ("righello", "d(b)", "d(b+1)", "nota"))
            for j, b in enumerate(righelli):
                db = mat[b][j] if b in mat else float("nan")
                db1 = mat[b + 1][j] if (b + 1) in mat else float("nan")
                nota = ("contained" if tipo == "post" else "pulito") + \
                    (", same block" if comp_nome == "ffn" else "")
                print("  %8d | %+9.2f | %+10.2f | %s" % (b, db, db1, nota))
            fuori["%s_%s" % (comp_nome, tipo)] = {str(k): v for k, v in mat.items()}

    print("\n  IThe POST ruler contains f_b by construction, so its")
    print("  diagonal is not clean. The PRE ruler is. Compare the two:")
    print("  if the term at b survives under PRE, it is genuine. The interpretation is yours.")
    return fuori


# =====================================================================
#  ablazione: azzerare un modulo sulla banda e leggere a valle
# =====================================================================
class Azzera:
    """Zeroes a module's output on the given blocks.
 
    Hooks the module that WRITES into the residual: on sandwich and post-norm
    architectures that is the post-norm, elsewhere the module itself. Zeroing
    the module's output before the norm would not zero the contribution,
    because the norm of zero is not zero when there is an additive gain.
    """

    def __init__(self, model, arch, blocchi, cosa):
        self.strati = _layers(model)
        self.arch, self.blocchi, self.cosa = arch, blocchi, cosa
        self.h = []

    def _bersaglio(self, L, cosa):
        if cosa == "attn":
            if getattr(self.arch, "sandwich", False):
                t = getattr(L, "post_attention_layernorm", None)
                if t is not None:
                    return t
            return getattr(L, "self_attn", None) or getattr(L, "attention", None)
        if getattr(self.arch, "sandwich", False):
            t = getattr(L, "post_feedforward_layernorm", None)
            if t is not None:
                return t
        for n in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
            t = getattr(L, n, None)
            if t is not None:
                return t
        return None

    def __enter__(self):
        for b in self.blocchi:
            t = self._bersaglio(self.strati[b], self.cosa)
            if t is None:
                raise RuntimeError("nessun modulo %s nel blocco %d" % (self.cosa, b))

            def hook(_m, _i, out):
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + tuple(out[1:])
                return torch.zeros_like(out)

            self.h.append(t.register_forward_hook(hook))
        return self

    def __exit__(self, *x):
        for h in self.h:
            h.remove()
        self.h = []


def auc_refit(H, pidx, folds, seed):
    """AUC with the axis REFIT on the states of the same condition."""
    a = []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        ax = fit_axis(H, [pidx[p] for p in tr])
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]
            I += [it, iff]; Y += [1, 0]
        a.append(float(auc_score(project_and_score(H[I], ax), torch.tensor(Y))))
    return sum(a) / len(a)


def auc_asse_fisso(H_asse, H_val, pidx, folds, seed):
    """Axis fit on the training folds of H_asse, read on the test folds of
    H_val. Same folds throughout."""
    a = []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        ax = fit_axis(H_asse, [pidx[p] for p in tr])
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]
            I += [it, iff]; Y += [1, 0]
        a.append(float(auc_score(project_and_score(H_val[I], ax), torch.tensor(Y))))
    return sum(a) / len(a)


def stage_ablazione(a, model, tok, items, pidx, R, arch, dev, P):
    """Intact, FFN off, attention off, and the FROZEN control.
 
    THE FROZEN CONTROL COSTS NO EXTRA PASS. With attention and FFN zeroed
    across the whole band, the state at the readout is identically the state
    ENTERING the band. It is therefore read off the intact run at the entry
    level, with no additional forward. It is the right reference: it says what
    readability would be if the band did nothing.
 
    If FFN-off resembles frozen, the band without its FFN adds nothing. If
    FFN-off sits ABOVE frozen, attention in the band is adding signal that the
    FFN normally destroys.
 
    Two protocols are reported: axis REFIT on the condition ("is anything still
    readable") and axis FIXED from the intact run ("does the original direction
    survive"). They are different questions.
    """
    banda = list(range(P + 1, min(P + 1 + a.banda, arch.n_blocks)))
    ingr = P + 1                      # livello di ingresso alla banda
    ro = min(P + a.banda, arch.n_blocks - 1)
    liv_ro = ro + 1
    print("\n[task ablation] intact, FFN OFF, ATTENTION OFF, FREEZE")
    print("  band blocks %s   entry at level %d   readout at level %d"
          % (banda, ingr, liv_ro))

    H_ingresso = R[:, ingr, :]        # = congelato, senza corse aggiuntive
    H_intatto = R[:, liv_ro, :]
    stati = {"intatto": H_intatto, "congelato": H_ingresso}
    for nome, cosa in (("FFN spenta", "ffn"), ("attenzione spenta", "attn")):
        with Azzera(model, arch, banda, cosa):
            stati[nome] = leggi_livello(model, tok, items, dev, a.batch, liv_ro)

    print("\n%-22s %9s %14s %13s" % ("condizione", "rifit", "asse@lettura",
                                      "asse@ingresso"))
    fuori = {}
    for nome in ("congelato", "intatto", "FFN spenta", "attenzione spenta"):
        H = stati[nome]
        r = auc_refit(H, pidx, a.folds, a.seed)
        if nome == "congelato":
            print("%-22s %9.3f %14s %13s" % (nome, r, "-", "-"))
            fuori[nome] = dict(rifit=r)
            continue
        x = auc_asse_fisso(H_intatto, H, pidx, a.folds, a.seed)
        p = auc_asse_fisso(H_ingresso, H, pidx, a.folds, a.seed)
        print("%-22s %9.3f %14.3f %13.3f" % (nome, r, x, p))
        fuori[nome] = dict(rifit=r, asse_lettura=x, asse_ingresso=p)
    print("\n  congelato (freeze); intatto (intact); FFN spenta (ffn-off); attenzione spenta (attention-off)")

    return fuori


@torch.no_grad()
def leggi_livello(model, tok, items, device, batch, livello):
    out = []
    for s in range(0, len(items), batch):
        enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(device)
        o = model(**enc, output_hidden_states=True)
        out.append(o.hidden_states[livello][:, -1, :].float().cpu())
    return torch.cat(out, 0)


def stage_geometria(a, R, pidx, P):
    """How many dimensions the states occupy, and how many truth occupies.
 
    It answers a question the AUC alone does not settle: an axis that reads
    little may mean the signal is weak, or that it is strong but spread over
    many directions. These are two different worlds and they must be separated
    by a measurement, not by an intuition about the architecture.
 
    FOUR NUMBERS, all dimensionless or normalised, and therefore comparable
    across models with different d.
 
    anisotropy    ||mean|| / mean spread around the mean. How much of a state's
                  typical norm is an offset shared by all of them. A residual
                  stream dominated by a common direction scores high; a centred
                  cloud scores low. It is the quantity the absence of bias
                  reduces. NOTE, however: on planted data, changing the shared
                  offset by a factor of 25 moves anisotropy from 4.25 to 0.17
                  while the axis AUC does not move (0.729 against 0.743). The
                  reason is algebraic: the axis is fit on pair DIFFERENCES, and
                  any component shared by both members, bias included, cancels
                  exactly. The number is printed because it is interesting in
                  itself, not because it explains readability.
 
    state rank    effective rank (exponential of the von Neumann entropy of the
                  spectrum) of the centred states, divided by the number of
                  available dimensions. How wide the cloud is.
 
    diff rank     the same on pair DIFFERENCES, which are the object the SVD
                  fits the axis on. This is the number that speaks to the
                  dimensionality of TRUTH rather than of states in general, and
                  is therefore the closest to the Part I claim.
 
    s1 share      fraction of the differences' energy captured by the first
                  singular component. If truth is concentrated on one axis this
                  is high; if it is spread out, low. A weak axis with a high
                  share is a small but concentrated signal; a weak axis with a
                  low share is a diffuse one.
 
    On the same planted data the s1 share is the only one of the four that
    tracks readability: 0.038 and 0.037 where the axis reads 0.73 and 0.74,
    against 0.016 where it reads 0.506, with the control fixed at 0.015 in every
    case. The effective rank barely moves because it is dominated by the
    variance of the topic. Read the share first, then the rest.
 
    The early block serves as a control: before the blocks have done any work,
    the s1 share on the differences has no reason to be high.
"""
    print("\n[task geometry] the cloud: how wide it is, and how wide is the truth")

    def misura(H, pidx):
        X = H.float()
        mu = X.mean(0)
        disp = float((X - mu).norm(dim=1).mean())
        aniso = float(mu.norm()) / max(disp, 1e-8)
        Xc = X - mu
        # rango effettivo sulla Gram delle osservazioni, normalizzato dal
        # minimo fra numero di righe e dimensione: e' il tetto raggiungibile
        er_s = effective_rank(Xc, from_gram=False)
        tetto_s = min(Xc.shape)
        D = torch.stack([X[it] - X[iff] for it, iff in pidx])
        er_d = effective_rank(D, from_gram=False)
        tetto_d = min(D.shape)
        sv = torch.linalg.svdvals(D)
        quota = float(sv[0] ** 2 / (sv ** 2).sum())
        return dict(aniso=aniso,
                    rs=er_s["effective_rank"], rs_n=er_s["effective_rank"] / tetto_s,
                    rd=er_d["effective_rank"], rd_n=er_d["effective_rank"] / tetto_d,
                    quota=quota)

    righe = [("block 0 ", misura(R[:, 1, :], pidx)),
             ("peak, block %d" % P, misura(R[:, P + 1, :], pidx))]
    print("  %-22s %10s %10s %10s %10s" % ("layer", "anisotropy",
                                           "rank st.", "rank dif.", "quota s1"))
    for nome, m in righe:
        print("  %-22s %10.2f %10.3f %10.3f %10.3f"
              % (nome, m["aniso"], m["rs_n"], m["rd_n"], m["quota"]))
    m = righe[1][1]
    print("\n  in absolute terms at peak: effective rank of states %.1f su %d available, "
          "differences %.1f" % (m["rs"], min(R.shape[0], R.shape[2]), m["rd"]))
    print("  The four numbers have no fixed thresholds: they are read AGAINST the exact same")
    print("  numbers from other models, using the same protocol and size n.")
    return righe[1][1]


def stage_categories(a, Rg, Fg, pset, P, early):
    print("\n[task categories] axis is a mixture of categories??")
    per_cat = pset.by_category()
    Hp = Rg[:, P + 1, :]
    He = Rg[:, early + 1, :]
    cats, Mp, assi = matrice_coseni(Hp, per_cat)
    _, Me, _ = matrice_coseni(He, per_cat)
    for nome, M in (("PICCO", Mp), ("EARLY (blocco %d)" % early, Me)):
        print("\n=== SIGNED cosines between axes per category @ %s ===" % nome)
        print("      " + "".join("%8s" % c for c in cats))
        for i, c in enumerate(cats):
            print("%6s" % c + "".join("%8.2f" % M[i, j] for j in range(len(cats))))
        off = [float(M[i, j]) for i in range(len(cats)) for j in range(len(cats)) if i < j]
        print("  off diagonal mean %+.2f   negative: %d/%d"
              % (sum(off) / len(off), sum(1 for x in off if x < 0), len(off)))
    _, T = matrice_trasferimento(Hp, per_cat, a.folds, a.seed)
    print("\n=== AUC transfer @ PICCO (asse riga -> dati colonna) ===")
    print("      " + "".join("%8s" % c for c in cats))
    for i, c in enumerate(cats):
        print("%6s" % c + "".join("%8.3f" % T[i, j] for j in range(len(cats))))
    dentro = [float(T[i, i]) for i in range(len(cats))]
    fuori = [float(T[i, j]) for i in range(len(cats)) for j in range(len(cats)) if i != j]
    print("  media dentro %.3f   media fuori %.3f"
          % (sum(dentro) / len(dentro), sum(fuori) / len(fuori)))

    pidx = pset.pidx
    etich = [p.category for p in pset.pairs]
    Df = torch.stack([Fg[it, P + 1, :] - Fg[iff, P + 1, :] for it, iff in pidx])
    De = torch.stack([Rg[it, early + 1, :] - Rg[iff, early + 1, :] for it, iff in pidx])
    print("\n=== categories decoding (nearest centroid, %d fold, %d perm) ==="
          % (a.folds, a.perm_cat))
    for nome, D in (("write Delta f @ layer %d" % (P + 1), Df),
                    ("residuo EARLY Delta h @ blocco %d (lexical baseline)" % early, De)):
        acc, nm, n95, p = decodifica_con_null(D, etich, a.folds, a.seed, a.perm_cat)
        print("  %s:" % nome)
        print("    accuracy %.2f%%   null media %.2f%%  null 95pct %.2f%%  p=%.4f  (chance %.2f%%)"
              % (100 * acc, 100 * nm, 100 * n95, p, 100 / len(cats)))
    return cats, Mp, Me, T, assi, Df


def stage_dizionario(a, out, cats, Mp, Me, T, assi, Df, pset, P, early, med,
                     t_global):
    Hall = torch.stack([unit(assi[c]) for c in cats], 0)
    t_global = unit(t_global)
    meta = dict(model=a.model, component="resid",
                peak_block=P, write_layer=P + 1, early_block=early,
                sentence_suffix=pset.protocol.suffix,
                join=pset.protocol.join, pool=pset.protocol.pool,
                dataset=pset.protocol.dataset, revision=pset.protocol.revision,
                seed=a.seed, folds=a.folds,
                k_relations=len(cats), pairs_per_relation=a.pairs_per_relation,
                identity_check_median=med, truthprobe_version=__version__)
    bundle = dict(cats=cats, axes=Hall, t_global=t_global,
                  cos_peak=Mp, cos_early=Me, transfer=T, meta=meta)
    d = os.path.join(out, "dizionari")
    os.makedirs(d, exist_ok=True)
    nome = "truthdict_%s_K%d_n%d_seed%d.pt" % (
        os.path.basename(a.model.rstrip("/\\")).replace("/", "_"),
        len(cats), a.pairs_per_relation, a.seed)
    p = os.path.join(d, nome)
    if os.path.exists(p) and not a.force:
        sys.exit("[stop] %s It already exists'. Use --force only if you know you want "
                 "to overwrite it: with re-write a bundle, the previous one will be replaced." % p)
    torch.save(bundle, p)
    with open(p.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, cats=cats,
                       cos_peak=[[round(float(x), 3) for x in r] for r in Mp],
                       cos_early=[[round(float(x), 3) for x in r] for r in Me],
                       transfer=[[round(float(x), 3) for x in r] for r in T]), f, indent=2)
    print("\n[saved] %s  (+ gemello .json)" % p)
    print("  cats: %s" % cats)
    print("  axes [K,d] = %s   t_global [d] = %s"
          % (tuple(Hall.shape), tuple(t_global.shape)))
    return p


def stage_gauge(a, out, thin=0.10):
    """Il gauge di consenso spettrale sul bundle appena scritto.

    L'orientamento di una categoria deciso dal suo stesso margine e' instabile
    quando il margine e' sottile: lo studio sui semi mostrava tempeste di segno.
    L'autovettore principale della matrice dei coseni e' un riferimento che non
    fabbrica accordo (due modelli indipendenti gaugiati separatamente danno
    Mantel +0.003) e che ha una condizione a priori: dove l'eigengap relativo e'
    piccolo, il segno non e' identificabile e va riportato come tale.

    Legge dal disco invece che dalla memoria, cosi' lo stage gira anche da solo
    su bundle prodotti in corse precedenti.
    """
    d = os.path.join(out, "dizionari")
    if not os.path.isdir(d):
        sys.exit("[gauge] nessuna cartella %s: lancia prima lo stage dizionario" % d)
    pts = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    if not pts:
        sys.exit("[gauge] nessun .pt in %s" % d)
    for nome in pts:
        p = os.path.join(d, nome)
        b = torch.load(p, map_location="cpu")
        meta = b.get("meta", {}) if isinstance(b.get("meta"), dict) else {}
        cats = list(b["cats"])
        A0 = torch.stack([unit(b["axes"][i].float()) for i in range(len(cats))])
        C = A0 @ A0.T
        ev, evec = torch.linalg.eigh(C)
        u = evec[:, -1]
        if u.sum() < 0:
            u = -u                      # convenzione: maggioranza positiva
        u = u / u.abs().max()
        # eigengap relativo: la condizione a priori di identificabilita'
        gap = float((ev[-1] - ev[-2]) / ev[-1]) if len(ev) > 1 else float("nan")
        print("\n=== %s   K=%d   gauge: consenso spettrale ===" % (p, len(cats)))
        print("  [provenance] model %s  block %s  n=%s  suffix %r  seed %s"
              % (meta.get("model", "?"), meta.get("peak_block", "?"),
                 meta.get("pairs_per_relation", "?"),
                 meta.get("sentence_suffix"), meta.get("seed", "?")))
        print("  relative eigengap %.3f  (thin = unidentifiable sign)" % gap)
        segni = []
        for i, c in enumerate(cats):
            m = float(u[i])
            s_ = 1.0 if m >= 0 else -1.0
            segni.append(s_)
            print("  %-6s margin = %+.3f%s%s"
                  % (c, m, "  [flip]" if s_ < 0 else "",
                     "  <-- thin, riportare senza segno" if abs(m) < thin else ""))
        S = torch.tensor(segni)
        Ag = torch.stack([unit((b["axes"].float() * S.unsqueeze(1))[i])
                          for i in range(len(cats))])
        M = (Ag @ Ag.T).tolist()
        fuori = dict(meta,
                     gauge="consenso spettrale (autovettore principale)",
                     eigengap_relativo=gap, thin_threshold=thin,
                     cats=cats,
                     gauge_margins=[round(float(x), 4) for x in u],
                     thin_categories=[c for c, x in zip(cats, u) if abs(float(x)) < thin],
                     flipped_vs_original=[c for c, x in zip(cats, segni) if x < 0],
                     cos_peak=[[round(float(x), 4) for x in r] for r in M])
        for k in ("cos_early", "transfer"):
            if k in b:
                v = b[k]
                fuori[k] = v.tolist() if hasattr(v, "tolist") else v
        fuori["cos_early_note"] = "gauge VECCHIO: il blocco iniziale non ne ha uno stabile"
        dst = os.path.splitext(p)[0] + "_gauge.json"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(fuori, f, indent=1)
        print("  flip: %s   thin: %s"
              % (fuori["flipped_vs_original"] or "nessuna",
                 fuori["thin_categories"] or "nessuna"))
        print("  [saved] %s" % dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stages", default="behav,signal",
                    help="comma-separated, run in this order: "
                         "behav (know-rate) and signal (finds the peak) come "
                         "first and need no landmarks; anatomy, flip, frames, "
                         "ablazione, geometria, categories, dizionario, gauge "
                         "all require peak_block in landmarks.json; signal prints the exact value to write."
                         "'all' runs everything, 'rest' everything after signal")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--k-relations", type=int, default=8,
                    help="number of relation categories")
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--early-block", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5,
                    help="cross-validation folds ")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds to average over")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed for pair selection")
    ap.add_argument("--perm", type=int, default=200,
                    help="permutation null (p-value) ")
    ap.add_argument("--perm-cat", type=int, default=100)
    ap.add_argument("--n-behav", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-vram", default=None,
                    help="tetto di VRAM, es. 9GiB. Activate offload of "
                         "accelerate instead silent offload of driver")
    ap.add_argument("--truncate-layers", type=int, default=None,
                    help="load only the first N blocks. Exact for anything read below N, unusable for the know-rate")
    ap.add_argument("--offload-folder", default=None,
                    help="offload folder in disk")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--frame_offsets", default="-4,-2,0,2",
                    help="Offset frames table of the paper PRE/POST: p-4, p-2, p, p+2")
    ap.add_argument("--frame_scan_start", type=int, default=-1)
    ap.add_argument("--banda", type=int, default=3,
                    help="ablation band width: blocks peak+1 to peak+banda, read at the end")
    # ----------------------------------------------------
    # ----------------------------------------------------

    a = ap.parse_args()

    a.device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    short = a.model.rstrip("/\\").replace("\\", "/").split("/")[-1]
    out = a.outdir or os.path.join("campagne", short)
    os.makedirs(out, exist_ok=True)
    stages = (STAGES if a.stages == "all"
              else STAGES[2:] if a.stages == "rest"
              else [s.strip() for s in a.stages.split(",")])
    for s in stages:
        if s not in STAGES:
            sys.exit("unknow stage: %s (chose %s)" % (s, STAGES))

    print("=" * 66)
    print("CAMPAGNA   truthprobe %s" % __version__)
    print("=" * 66)
    print("[model] %s" % a.model)
    if a.truncate_layers and "behav" in stages:
        sys.exit("[stop] --truncate-layers toglie i blocchi finali, quindi i "
                 "logit non hanno senso e il know-rate sarebbe inventato.\n"
                 "  Misura behav a parte sul modello intero, poi tronca per la "
                 "geometria.")
    if a.truncate_layers:
        print("[tronca] %d blocchi caricati. Il picco e tutta la banda del flip "
              "devono stare sotto: tieni almeno picco+8." % a.truncate_layers)
    gate_predizioni(out, a.model)
    P = gate_landmarks(out, stages)

    proto = Protocol(suffix=a.suffix, seed=a.seed,
                     k_relations=a.k_relations,
                     pairs_per_relation=a.pairs_per_relation,
                     max_pairs=a.max_pairs)
    print("[protocol] suffix %r  join %s  pool %s  seed %d"
          % (proto.suffix, proto.join, proto.pool, proto.seed))

    if "behav" in stages:
        stage_behav(a, out)

    piatto = [s for s in stages if s in ("signal", "anatomy", "flip",
                                     "frames", "ablazione", "geometria")]
    raggr = [s for s in stages if s in ("categories", "dizionario")]

    if piatto:
        ps = counterfact_flat(proto, max_pairs=a.max_pairs,
                              local_file=a.file_counterfact)
        print("[data] %d frasi = %d coppie (campione piatto)"
              % (len(ps.items), len(ps.pidx)))
        avvisa_taglia(a.model, a.device, a.max_vram)
        tok, model = carica(a.model, a.device, a.max_vram, a.offload_folder, a.truncate_layers)
        dev_exec = getattr(model, "device", a.device)
        arch = describe(model)
        for r in arch.summary():
            print("  " + r)
        R, A, Fq = estrai(model, tok, ps.items, arch, dev_exec, a.batch)
        serve_modello = "ablazione" in stages
        if not serve_modello:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rel = errore_identita(R, A, Fq)
        med = float(rel.median())
        print("[identity gate] attn + ffn = residual delta, median %.2e" % med)
        if med > 1e-3:
            sys.exit("[stop] decomposition not valid: niente sotto questa riga "
                     "avrebbe significato.")
        if "signal" in stages:
            stage_signal(a, R, ps.pidx)
        if "anatomy" in stages:
            stage_anatomy(a, R, A, Fq, ps.pidx)
        if "flip" in stages:
            stage_flip(a, R, A, Fq, ps.pidx, P)
        if "frames" in stages:
            a._parallelo = getattr(arch, "parallel", False)
            METRICHE["frames"] = stage_frames(a, R, A, Fq, ps.pidx, P)
        if "ablazione" in stages:
            METRICHE["ablazione"] = stage_ablazione(
                a, model, tok, ps.items, ps.pidx, R, arch, dev_exec, P)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if "geometria" in stages:
            stage_geometria(a, R, ps.pidx, P)
        del R, A, Fq

    if raggr:
        pg = counterfact_by_relation(proto, k=a.k_relations,
                                     n_per=a.pairs_per_relation,
                                     whitelist=WHITELIST_8[:a.k_relations]
                                     if a.k_relations == 8 else None,
                                     local_file=a.file_counterfact)
        print("[data] %d phrase = %d pairs, %d categories"
              % (len(pg.items), len(pg.pidx), len(pg.categories)))
        avvisa_taglia(a.model, a.device, a.max_vram)
        tok, model = carica(a.model, a.device, a.max_vram, a.offload_folder, a.truncate_layers)
        dev_exec = getattr(model, "device", a.device)
        arch = describe(model)
        Rg, Ag, Fg = estrai(model, tok, pg.items, arch, dev_exec, a.batch)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        medg = float(errore_identita(Rg, Ag, Fg).median())
        print("[gate] median %.2e" % medg)
        if medg > 1e-3:
            sys.exit("[stop] decomposition not valid.")
        res = stage_categories(a, Rg, Fg, pg, P, a.early_block)
        if "dizionario" in stages:
            # l'asse globale: un asse solo su TUTTE le coppie insieme, che e'
            # il riferimento con margine stabile contro cui il gauge orienta
            # le categorie sottili
            tg = fit_axis(Rg[:, P + 1, :], pg.pidx)["v1"]
            stage_dizionario(a, out, *res, pg, P, a.early_block, medg, tg)

    if "gauge" in stages:
        stage_gauge(a, out)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(modello=a.model, picco=P, protocollo=proto.key()
                           if hasattr(proto, "key") else None,
                           metriche=METRICHE), f, indent=1, default=float)
        print("\n[json] Data write in %s" % a.json)

    print("\n[end] la lettura appartiene al ricercatore.")


if __name__ == "__main__":
    main()
