#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""localizza_asse.py  --  dove e' stato letto l'asse di un dizionario?

Due pipeline producono per lo stesso modello un t_global a coseno -0.121, cioe'
quasi ortogonale. La convenzione della frase da sola darebbe +0.52, misurato.
Quindi la differenza non e' nella frase: e' in DOVE viene agganciata la lettura.

Non serve indovinare. t_global e' un vettore in d dimensioni, e se lo si
confronta con l'asse globale fittato a ogni livello e su ogni flusso di una
nuova estrazione, il massimo del coseno dice dove quel vettore e' nato.

  massimo al livello picco+1 sul residuo -> i due leggono nello stesso posto,
      e la divergenza sta altrove (materiale, semi, orientamento)
  massimo su attn o ffn -> il dizionario e' stato costruito su una componente
      diversa da resid
  massimo a un livello diverso -> off-by-one nella convenzione degli indici

SI USA IL VALORE ASSOLUTO del coseno, perche' l'orientamento di un asse e'
convenzionale: fit_axis lo fissa con la sua regola, e due esecuzioni possono
uscire con segno opposto senza che niente sia diverso.

IL MATERIALE DEVE COINCIDERE. L'asse dipende dalle coppie su cui e' fittato,
quindi K, n e seme vanno passati uguali a quelli del bundle da localizzare.
Se il bundle non li registra, si prova la combinazione dichiarata e si legge
il coseno massimo raggiunto: un massimo basso ovunque significa che nemmeno il
materiale coincide, e il confronto non e' localizzante.

    python localizza_asse.py --bundle campagne\\X\\dizionari\\truthdict_X.pt \\
        --model <path> --k-relations 8 --pairs-per-relation 60 --seed 0

Compagno di arXiv:2607.16741. Licenza CC BY 4.0.
"""

import argparse
import sys

import torch

from truthprobe import __version__
from truthprobe.data import counterfact_by_relation, counterfact_flat
from truthprobe.geometry import unit, fit_axis
from truthprobe.hooks import describe, BlockCapture
from truthprobe.protocol import Protocol


@torch.no_grad()
def estrai(model, tok, items, arch, device, batch):
    caps = [BlockCapture(model, arch, b) for b in range(arch.n_blocks)]
    for c in caps:
        c.__enter__()
    R, A, F = [], [], []
    try:
        for s in range(0, len(items), batch):
            enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(device)
            o = model(**enc, output_hidden_states=True)
            R.append(torch.stack([h[:, -1, :].float().cpu() for h in o.hidden_states], 1))
            A.append(torch.stack([c.attn() for c in caps], 1))
            F.append(torch.stack([c.ffn() for c in caps], 1))
            print("\r[extract] %d/%d" % (min(s + batch, len(items)), len(items)),
                  end="", flush=True)
        print()
    finally:
        for c in caps:
            c.__exit__(None, None, None)
    return torch.cat(R, 0), torch.cat(A, 0), torch.cat(F, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help=".pt di cui localizzare l'asse")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k-relations", type=int, default=8)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="se dato, usa il campione PIATTO invece che raggruppato")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-vram", default=None)
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--top", type=int, default=8, help="quante righe stampare")
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    b = torch.load(a.bundle, map_location="cpu", weights_only=False)
    if "t_global" not in b:
        sys.exit("[stop] il bundle non contiene t_global")
    tg = unit(b["t_global"].float())
    meta = b.get("meta") if isinstance(b.get("meta"), dict) else {}

    print("=" * 70)
    print("LOCALIZZA ASSE   truthprobe %s" % __version__)
    print("=" * 70)
    print("[bundle] %s" % a.bundle)
    print("  dichiara: modello %s  blocco %s  K %s  n %s  suffisso %r  seed %s"
          % (meta.get("model", b.get("model", "?")),
             meta.get("peak_block", b.get("peak_block", "?")),
             meta.get("k_relations", len(b.get("cats", []))),
             meta.get("pairs_per_relation", "?"),
             meta.get("sentence_suffix"), meta.get("seed", "?")))
    print("  t_global: %d dimensioni" % tg.shape[0])

    proto = Protocol(suffix=a.suffix, seed=a.seed, k_relations=a.k_relations,
                     pairs_per_relation=a.pairs_per_relation)
    if a.max_pairs:
        ps = counterfact_flat(proto, max_pairs=a.max_pairs, local_file=a.file_counterfact)
        print("[data] campione PIATTO, %d coppie" % len(ps.pidx))
    else:
        ps = counterfact_by_relation(proto, k=a.k_relations,
                                     n_per=a.pairs_per_relation,
                                     local_file=a.file_counterfact)
        print("[data] campione RAGGRUPPATO, %d coppie, %d categorie"
              % (len(ps.pidx), len(ps.categories)))

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
    if arch.d_model != tg.shape[0]:
        sys.exit("[stop] il bundle ha d=%d, il modello d=%d: non sono lo stesso "
                 "modello." % (tg.shape[0], arch.d_model))

    R, A, F = estrai(model, tok, ps.items, arch, dev, a.batch)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pidx = list(ps.pidx)
    righe = []
    for nome, H in (("resid", R), ("attn", A), ("ffn", F)):
        for lev in range(H.shape[1]):
            v = unit(fit_axis(H[:, lev, :], pidx)["v1"])
            c = float(torch.dot(v, tg))
            # il segno di un asse e' convenzionale: conta il modulo
            righe.append((abs(c), c, nome, lev))
    righe.sort(reverse=True)

    print("\n--- DOVE NASCE QUEL VETTORE ---------------------------------")
    print("  %-8s %6s %10s %10s" % ("flusso", "indice", "|cos|", "cos"))
    for m, c, nome, lev in righe[:a.top]:
        eti = "livello %d" % lev if nome == "resid" else "blocco %d" % lev
        print("  %-8s %6s %10.3f %+10.3f" % (nome, eti.split()[-1], m, c))

    m, c, nome, lev = righe[0]
    print("\n  massimo: %s, indice %d, |cos| = %.3f" % (nome, lev, m))
    pk = meta.get("peak_block")
    if pk is not None and nome == "resid" and lev == int(pk) + 1:
        print("  Coincide con il residuo a picco+1, cioe' con la convenzione")
        print("  dichiarata dal bundle. La divergenza NON e' nel punto di lettura.")
    elif nome != "resid":
        print("  Il dizionario e' stato costruito su %s, non sul residuo." % nome)
    elif pk is not None:
        print("  Atteso il livello %d (picco %s piu' uno), trovato %d: scarto di %d."
              % (int(pk) + 1, pk, lev, lev - int(pk) - 1))
    if m < 0.5:
        print("\n  ATTENZIONE: il massimo e' basso ovunque. Non e' localizzazione:")
        print("  significa che nemmeno il MATERIALE coincide. Prima di leggere")
        print("  la riga sopra, controlla K, n, seme e convenzione della frase.")


if __name__ == "__main__":
    main()
