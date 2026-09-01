#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gatevalue.py  --  un compito solo: congelare il gate o il valore.

Replica `swiglu.py gatefreeze` del corpus canonico, con le stesse scelte, su
architetture che il canonico non copre (post-norm, e in generale qualunque
collocazione della norma). Non fa altro: nessuno stage, nessun cancello di
predizioni, nessun bundle. Meno pezzi mobili possibile.

LE QUATTRO SCELTE CHE DEVONO COINCIDERE COL CANONICO

  1. DOVE si legge il gate. All'uscita dell'ATTIVAZIONE (mlp.act_fn), non alla
     proiezione. Silu non e' lineare, quindi silu(media) non e' media(silu):
     congelare prima o dopo sono due operazioni diverse, e agganciare
     gate_proj inverte il segno del risultato.

  2. QUANTE posizioni. Solo l'ULTIMO token, perche' e' li' che si legge lo
     stato. Congelare tutta la sequenza e' un intervento diverso e piu' ampio.

  3. QUALE media. Quella della corsa INTATTA, memorizzata prima. Intervenendo
     su piu' blocchi, la media calcolata al volo al secondo blocco verrebbe da
     uno stato che il primo ha gia' alterato.

  4. COME si legge l'AUC. Asse RIFITTATO sugli stati della stessa condizione,
     con gli stessi fold. E' il protocollo di ffn_erosion ablate.

IL RIFERIMENTO CONGELATO non costa una corsa: azzerando attenzione e FFN su
tutta la banda, lo stato al readout e' identicamente quello che entra nella
banda, quindi si legge dall'intatto al livello di ingresso.

NON SI APPLICA dove l'FFN non ha un gate: Phi usa fc1/fc2 con GELU, i MoE
hanno l'FFN instradato senza gate_proj a livello di strato. In quei casi il
programma lo dice e si ferma, invece di agganciare la cosa sbagliata.

    python gatevalue.py --model Qwen/Qwen2.5-1.5B --peak 15
    python gatevalue.py --model modelli\\OLMo-2-1B --peak 8 --json olmo_gv.json

Compagno di arXiv:2607.16741. Licenza CC BY 4.0.
"""

import argparse
import json
import sys

import torch

from truthprobe import __version__
from truthprobe.data import counterfact_flat
from truthprobe.geometry import fit_axis
from truthprobe.hooks import describe, _layers
from truthprobe.protocol import Protocol
from truthprobe.stats import auc_score, kfold_pairs, project_and_score


def bersaglio(L, quale):
    """Il modulo da agganciare, o None se il blocco non ha un FFN a gate."""
    mlp = getattr(L, "mlp", None) or getattr(L, "feed_forward", None)
    if mlp is None:
        return None
    if quale == "gate":
        return getattr(mlp, "act_fn", None)          # DOPO l'attivazione
    return getattr(mlp, "up_proj", None) or getattr(mlp, "w3", None)


def auc_refit(H, pidx, folds, seed):
    """Asse rifittato sui train della stessa condizione, letto sui test."""
    out = []
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        ax = fit_axis(H, [pidx[p] for p in tr])
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]
            I += [it, iff]
            Y += [1, 0]
        out.append(float(auc_score(project_and_score(H[I], ax), torch.tensor(Y))))
    return sum(out) / len(out)


@torch.no_grad()
def passa(model, tok, items, dev, batch, livelli, ganci=None):
    """Un passaggio in avanti, restituendo gli hidden state ai livelli chiesti.

    `ganci` e' una lista di handle gia' registrati: qui si esegue e basta.
    """
    out = {k: [] for k in livelli}
    for s in range(0, len(items), batch):
        enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for k in livelli:
            out[k].append(hs[k][:, -1, :].float().cpu())
    return {k: torch.cat(v, 0) for k, v in out.items()}


@torch.no_grad()
def cache_media(model, tok, items, dev, batch, banda, quale):
    """La media di coppia di g (o u) all'ultimo token, dalla corsa INTATTA.

    Le due righe di ogni coppia escono gia' con la loro media, cosi' il gancio
    deve solo leggerla.
    """
    strati = _layers(model)
    buf, h = {}, []
    for b in banda:
        t = bersaglio(strati[b], quale)
        if t is None:
            for x in h:
                x.remove()
            return None

        def hook(_m, _i, out, b=b):
            buf.setdefault(b, []).append(out[:, -1, :].detach().float().cpu())

        h.append(t.register_forward_hook(hook))
    try:
        for s in range(0, len(items), batch):
            enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(dev)
            model(**enc)
    finally:
        for x in h:
            x.remove()
    fuori = {}
    for b, v in buf.items():
        X = torch.cat(v, 0)
        if X.shape[0] % 2:
            sys.exit("[stop] numero dispari di frasi")
        m = X.view(X.shape[0] // 2, 2, -1).mean(1, keepdim=True)
        fuori[b] = m.expand(-1, 2, -1).reshape(X.shape)
    return fuori


class Congela:
    """Sostituisce l'ultimo token di g (o u) con la media di coppia intatta."""

    def __init__(self, model, banda, quale, cache):
        self.strati = _layers(model)
        self.banda, self.quale, self.cache = banda, quale, cache
        self.n = 0
        self.h = []

    def __enter__(self):
        for b in self.banda:
            t = bersaglio(self.strati[b], self.quale)

            def hook(_m, _i, out, b=b):
                m = self.cache[b][self.n:self.n + out.shape[0]]
                o = out.clone()
                o[:, -1, :] = m.to(out.device, out.dtype)
                return o

            self.h.append(t.register_forward_hook(hook))
        return self

    def avanza(self, k):
        self.n += k

    def __exit__(self, *a):
        for h in self.h:
            h.remove()
        self.h = []


@torch.no_grad()
def leggi_con(model, tok, items, dev, batch, livello, ctx):
    out = []
    for s in range(0, len(items), batch):
        enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(hs[livello][:, -1, :].float().cpu())
        ctx.avanza(enc["input_ids"].shape[0])
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--peak", type=int, required=True, help="BLOCCO, non livello")
    ap.add_argument("--banda", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-vram", default=None)
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--json", dest="uscita_json", default=None)
    a = ap.parse_args()

    if a.batch % 2:
        sys.exit("[stop] --batch deve essere pari: le coppie sono righe adiacenti")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    print("=" * 66)
    print("GATE E VALORE   truthprobe %s" % __version__)
    print("=" * 66)
    proto = Protocol(suffix=a.suffix, seed=a.seed, max_pairs=a.max_pairs)
    ps = counterfact_flat(proto, max_pairs=a.max_pairs, local_file=a.file_counterfact)
    pidx = list(ps.pidx)
    print("[data] %d frasi = %d coppie   suffisso %r"
          % (len(ps.items), len(pidx), proto.suffix))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.float32, use_safetensors=True, trust_remote_code=False)
    if a.max_vram:
        kw.update(device_map="auto", max_memory={0: a.max_vram, "cpu": "24GiB"})
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
        dev = getattr(model, "device", dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw).to(dev)
    model.eval()
    arch = describe(model)
    for r in arch.summary():
        print("  " + r)

    strati = _layers(model)
    if bersaglio(strati[a.peak], "gate") is None:
        sys.exit("\n[stop] questa architettura non ha un FFN a gate. Phi usa "
                 "fc1/fc2 con GELU, i MoE hanno l'FFN instradato senza "
                 "gate_proj a livello di strato. La domanda gate contro valore "
                 "non si pone qui.")

    P = a.peak
    banda = list(range(P + 1, min(P + 1 + a.banda, arch.n_blocks)))
    ingr = P + 1
    liv = min(P + a.banda, arch.n_blocks - 1) + 1
    print("[banda] blocchi %s   ingresso al livello %d   lettura al livello %d"
          % (banda, ingr, liv))

    stati = passa(model, tok, ps.items, dev, a.batch, [ingr, liv])
    base = auc_refit(stati[liv], pidx, a.folds, a.seed)
    congelato = auc_refit(stati[ingr], pidx, a.folds, a.seed)
    print("\n  %-34s %8.3f" % ("intatto al readout", base))
    print("  %-34s %8.3f" % ("intatto all'ingresso (= congelato)", congelato))

    ris = dict(intatto=base, congelato=congelato)
    for etichetta, quale in (("GATE congelato  (Dg spento)", "gate"),
                             ("VALORE congelato (Du spento)", "up")):
        cache = cache_media(model, tok, ps.items, dev, a.batch, banda, quale)
        if cache is None:
            print("  %-34s   non applicabile" % etichetta)
            continue
        ctx = Congela(model, banda, quale, cache)
        with ctx:
            H = leggi_con(model, tok, ps.items, dev, a.batch, liv, ctx)
        v = auc_refit(H, pidx, a.folds, a.seed)
        print("  %-34s %8.3f   %+.3f rispetto all'intatto" % (etichetta, v, v - base))
        ris["gate" if quale == "gate" else "valore"] = v

    print("\nCOME SI LEGGE")
    print("  Delta POSITIVO: congelare quella componente MIGLIORA la")
    print("  leggibilita' a valle, quindi quella componente spingeva contro")
    print("  l'asse. Delta negativo: portava verita' e toglierla costa.")
    print("  Il congelato dice quanto varrebbe se la banda non facesse nulla.")

    if a.uscita_json:
        with open(a.uscita_json, "w", encoding="utf-8") as f:
            json.dump(dict(modello=a.model, picco=P, banda=banda,
                           lettura=liv, suffisso=proto.suffix,
                           coppie=len(pidx), fold=a.folds, seme=a.seed,
                           risultati=ris), f, indent=1)
        print("\n[json] %s" % a.uscita_json)


if __name__ == "__main__":
    main()
