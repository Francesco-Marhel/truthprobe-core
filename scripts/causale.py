#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""causale.py  --  what the FFN actually does after the peak: rotate or erode?

The rotation-versus-erosion table in campagna.py is OBSERVATIONAL: it shows
where the FFN's mean difference points and leaves the reading to you. This tool
INTERVENES: it removes a component during the forward pass and measures what
survives downstream. They are different questions, and the second does not
follow from the first.

THE ALGEBRA THAT SEPARATES THE TWO HYPOTHESES

Let v be the truth axis and f the FFN's write. Decompose f = f_par + f_ort,
with f_par = (f.v)v.

  PURE ROTATION. If the FFN only adds orthogonal components, the cosine between
  the residual and v decays toward zero because the norm grows while the
  projection does not change. But it can NEVER cross zero: (x+y).x = ||x||^2 > 0
  for every y orthogonal to x. An anti-parallel component is required to change
  the sign.

  EROSION. f_par points against v and subtracts projection. The sign inverts.

Observationally the two are told apart by whether the "along" column crosses
zero or merely approaches it. Causally they are told apart like this: if you
remove f_ort and downstream readability is preserved, rotation was not doing the
damage; if you remove f_par and it is preserved, rotation was.

THE TWO MODES

  direzionale   replaces the FFN's write across the band with one of its
                projections. Three variants plus a control that removes a
                random direction of the same construction, because removing a
                large component takes the model off distribution and part of
                the damage is disturbance rather than direction.

  gatevalue     freezes one of the two SwiGLU components by replacing it with
                the intra-pair mean, so it can no longer carry the true/false
                contrast while its scale remains. It answers which of the two
                carries truth and which pushes against it. Requires a gated
                FFN: it does not apply to MoE models nor to Phi.

THE AXIS IS ALWAYS THE INTACT ONE, fit on the training pairs of the run without
interventions, and read on the held-out pairs. Refitting after the intervention
would measure whether a direction STILL exists, not whether the earlier one
survived.

    python causale.py --model <path> --peak 16 --modo direzionale
    python causale.py --model <path> --peak 16 --modo gatevalue --per-blocco

Companion to arXiv:2607.16741. Licensed CC BY 4.0.
"""

import argparse
import math
import statistics as st
import sys
import time

import torch

from truthprobe import CANONICAL, __version__
from truthprobe.data import counterfact_flat
from truthprobe.geometry import unit, fit_axis
from truthprobe.hooks import describe, _layers
from truthprobe.protocol import Protocol
from truthprobe.stats import auc_score, kfold_pairs, project_and_score


# =====================================================================
#  intervento
# =====================================================================
class Intervento:
    """Sostituisce la scrittura di un modulo durante il forward.

    Aggancia il modulo che SCRIVE nel residuo, che su pre-norm e' l'MLP e su
    sandwich o post-norm e' la post-norma. Il moltiplicatore residuo non entra
    qui: viene applicato dal modello dopo, quindi una proiezione fatta prima
    resta una proiezione anche dopo.
    """

    def __init__(self, model, arch, blocchi, fn):
        self.model, self.arch, self.blocchi, self.fn = model, arch, blocchi, fn
        self.strati = _layers(model)
        self.h = []

    def _target(self, b):
        L = self.strati[b]
        if getattr(self.arch, "sandwich", False):
            t = getattr(L, "post_feedforward_layernorm", None)
            if t is not None:
                return t
        for nome in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
            t = getattr(L, nome, None)
            if t is not None:
                return t
        raise RuntimeError("nessun modulo FFN nel blocco %d" % b)

    def __enter__(self):
        for b in self.blocchi:
            t = self._target(b)

            def hook(_m, _i, out, b=b):
                if isinstance(out, tuple):
                    return (self.fn(out[0], b),) + out[1:]
                return self.fn(out, b)

            self.h.append(t.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()
        self.h = []


class CongelaSwiGLU:
    """Congela gate o valore sostituendoli con la media della coppia.

    TRE SCELTE CHE DEVONO COINCIDERE COL CANONICO, e che nella prima versione
    di questo file erano tutte e tre sbagliate.

    DOVE. Il gate si aggancia all'uscita dell'ATTIVAZIONE, non alla proiezione.
    Il canonico aggancia mlp.act_fn e la libreria fa lo stesso, con la
    motivazione scritta in hooks.py: applicare silu a mano assumerebbe quale
    non linearita' il modello usa. E c'e' di piu': siccome silu non e' lineare,
    silu(media(g)) non e' media(silu(g)), quindi congelare prima o dopo
    l'attivazione sono due operazioni diverse. Agganciare gate_proj dava alla
    colonna gate un segno opposto a quello canonico.

    QUANDO. Solo l'ULTIMO token. Lo stato che si legge e' quello del token
    finale; congelare tutte le posizioni e' un intervento molto piu' ampio e
    non e' quello che il canonico misura.

    QUALE MEDIA. Quella della corsa INTATTA, memorizzata prima. Intervenendo su
    una banda di piu' blocchi, la media calcolata al volo al secondo blocco
    verrebbe da uno stato che il primo ha gia' alterato.

    NON SI APPLICA a Phi (fc1/fc2 con GELU, non c'e' gate) ne' ai MoE come
    Granite (l'FFN e' instradato e non ha gate_proj a livello di strato).
    """

    def __init__(self, model, arch, blocchi, quale, cache):
        self.model, self.arch, self.blocchi = model, arch, blocchi
        self.quale, self.cache = quale, cache
        self.strati = _layers(model)
        self.n = 0
        self.h = []

    @staticmethod
    def bersaglio(L, quale):
        """Il modulo da agganciare, o None se l'architettura non ha un gate."""
        mlp = getattr(L, "mlp", None) or getattr(L, "feed_forward", None)
        if mlp is None:
            return None
        if quale == "gate":
            return getattr(mlp, "act_fn", None)     # DOPO l'attivazione
        return getattr(mlp, "up_proj", None) or getattr(mlp, "w3", None)

    def __enter__(self):
        for b in self.blocchi:
            t = self.bersaglio(self.strati[b], self.quale)
            if t is None:
                raise SenzaGate(self.quale)

            def hook(_m, _i, out, b=b):
                # sostituisce SOLO l'ultimo token con la media di coppia
                # presa dalla corsa intatta
                m = self.cache[b][self.n:self.n + out.shape[0]]
                o = out.clone()
                o[:, -1, :] = m.to(out.device, out.dtype)
                return o

            self.h.append(t.register_forward_hook(hook))
        return self

    def avanza(self, k):
        """Il gancio deve sapere a quali frasi corrisponde il lotto corrente."""
        self.n += k

    def __exit__(self, *a):
        for h in self.h:
            h.remove()
        self.h = []


class SenzaGate(RuntimeError):
    def __init__(self, quale):
        super().__init__(
            "questa architettura non ha un FFN a gate, quindi non c'e' nessun "
            "%s da congelare. Phi usa fc1/fc2 con GELU; i MoE come Granite "
            "hanno l'FFN instradato senza gate_proj a livello di strato. "
            "La domanda gate contro valore non si pone su questi modelli."
            % ("gate" if quale == "gate" else "valore"))


@torch.no_grad()
def cache_media_coppia(model, tok, items, arch, blocchi, quale, device, batch):
    """La media di coppia di g (o u) all'ultimo token, dalla corsa INTATTA.

    Restituisce {blocco: [N, d_i]} dove le due righe di ogni coppia portano
    gia' la loro media, cosi' il gancio deve solo leggerla.
    """
    strati = _layers(model)
    buf, h = {}, []
    for b in blocchi:
        t = CongelaSwiGLU.bersaglio(strati[b], quale)
        if t is None:
            raise SenzaGate(quale)

        def hook(_m, _i, out, b=b):
            buf.setdefault(b, []).append(out[:, -1, :].detach().float().cpu())

        h.append(t.register_forward_hook(hook))
    try:
        for s in range(0, len(items), batch):
            enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(device)
            model(**enc)
    finally:
        for x in h:
            x.remove()
    out = {}
    for b, v in buf.items():
        X = torch.cat(v, 0)
        if X.shape[0] % 2:
            raise RuntimeError("numero dispari di frasi: le coppie devono "
                               "essere righe adiacenti")
        m = X.view(X.shape[0] // 2, 2, -1).mean(1, keepdim=True)
        out[b] = m.expand(-1, 2, -1).reshape(X.shape)
    return out


# =====================================================================
#  proiezioni
# =====================================================================
def solo_parallela(v):
    v = unit(v)
    def f(t, _b):
        w = v.to(t.device, t.dtype)
        return (t @ w).unsqueeze(-1) * w
    return f


def solo_ortogonale(v):
    v = unit(v)
    def f(t, _b):
        w = v.to(t.device, t.dtype)
        return t - (t @ w).unsqueeze(-1) * w
    return f


def togli_casuale(v, seme):
    """Controllo: rimuove una direzione casuale invece di v.

    Serve perche' rimuovere una componente e' un intervento massiccio e una
    parte del danno e' disturbo generico. Se il danno qui e' pari a quello di
    solo_parallela, la direzione non conta e conta solo l'aver tolto qualcosa.
    """
    g = torch.Generator().manual_seed(seme)
    r = torch.randn(v.shape[0], generator=g)
    r = unit(r - (r @ unit(v)) * unit(v))       # ortogonale a v, per non ripetere il test
    def f(t, _b):
        w = r.to(t.device, t.dtype)
        return t - (t @ w).unsqueeze(-1) * w
    return f


# =====================================================================
#  estrazione con lettura a un livello solo
# =====================================================================
@torch.no_grad()
def leggi(model, tok, items, device, batch, livello, ctx=None):
    """Legge il residuo a un livello. Se c'e' un intervento con stato, gli dice
    a che punto della lista di frasi siamo: la media di coppia memorizzata e'
    indicizzata per frase, non per lotto."""
    out = []
    for s in range(0, len(items), batch):
        enc = tok(items[s:s + batch], return_tensors="pt", padding=True).to(device)
        o = model(**enc, output_hidden_states=True)
        out.append(o.hidden_states[livello][:, -1, :].float().cpu())
        if ctx is not None and hasattr(ctx, "avanza"):
            ctx.avanza(enc["input_ids"].shape[0])
    return torch.cat(out, 0)


def auc_su_assi(H, pidx, assi):
    """Assi PREFITTATI sull'intatto: si legge, non si rifitta."""
    a = []
    for ax, te in assi:
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]
            I += [it, iff]; Y += [1, 0]
        a.append(float(auc_score(project_and_score(H[I], ax), torch.tensor(Y))))
    return st.mean(a), (st.pstdev(a) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--peak", type=int, required=True, help="BLOCCO, non livello")
    ap.add_argument("--modo", choices=["direzionale", "gatevalue"], required=True)
    ap.add_argument("--banda", type=int, default=3,
                    help="quanti blocchi dopo il picco intervenire")
    ap.add_argument("--per-blocco", action="store_true",
                    help="un blocco alla volta invece che tutta la banda")
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--suffix", default=".")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-vram", default=None)
    ap.add_argument("--file-counterfact", default=None)
    a = ap.parse_args()

    if a.batch % 2:
        sys.exit("[stop] --batch deve essere pari: le coppie sono righe "
                 "adiacenti e non devono cadere a cavallo di due lotti.")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    proto = Protocol(suffix=a.suffix, seed=a.seed, max_pairs=a.max_pairs)

    print("=" * 68)
    print("CAUSALE   truthprobe %s   modo %s" % (__version__, a.modo))
    print("=" * 68)
    ps = counterfact_flat(proto, max_pairs=a.max_pairs, local_file=a.file_counterfact)
    pidx = list(ps.pidx)
    print("[data] %d frasi = %d coppie" % (len(ps.items), len(pidx)))

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

    P = a.peak
    lettura = min(P + a.banda + 1, arch.n_blocks)      # livello di hidden state
    banda = list(range(P + 1, min(P + 1 + a.banda, arch.n_blocks)))
    print("[banda] blocchi %s   lettura al livello %d" % (banda, lettura))

    # ---- intatto: assi per fold e riferimento -------------------------
    t0 = time.time()
    H_pk = leggi(model, tok, ps.items, dev, a.batch, P + 1)
    assi = [(fit_axis(H_pk, [pidx[p] for p in tr]), te)
            for tr, te in kfold_pairs(len(pidx), a.folds, a.seed)]
    H_int = leggi(model, tok, ps.items, dev, a.batch, lettura)
    base, sd_b = auc_su_assi(H_int, pidx, assi)
    print("\n[intatto] AUC al livello %d con asse del picco: %.3f (sd %.3f)"
          " (%.1f s)" % (lettura, base, sd_b, time.time() - t0))

    v = unit(fit_axis(H_pk, pidx)["v1"])

    # ---- configurazioni ----------------------------------------------
    if a.modo == "direzionale":
        conf = [("solo parallela  (tolgo l'ortogonale)", solo_parallela(v)),
                ("solo ortogonale (tolgo la parallela)", solo_ortogonale(v)),
                ("controllo: tolgo una direzione casuale", togli_casuale(v, a.seed))]
    else:
        conf = [("congelo il GATE", "gate"), ("congelo il VALORE", "up")]
        # la media di coppia si prende dalla corsa INTATTA, una volta sola
        cache = {}
        for _, q in conf:
            try:
                cache[q] = cache_media_coppia(model, tok, ps.items, arch, banda,
                                              q, dev, a.batch)
            except SenzaGate as e:
                print("\n[stop] %s" % e)
                return

    gruppi = ([[b] for b in banda] if a.per_blocco else [banda])

    print("\n%-40s %6s %8s %8s" % ("intervento", "blocchi", "AUC", "delta"))
    print("%-40s %6s %8.3f %8s" % ("intatto", "-", base, "-"))
    for g in gruppi:
        for nome, spec in conf:
            try:
                if a.modo == "direzionale":
                    ctx = Intervento(model, arch, g, spec)
                    with ctx:
                        H = leggi(model, tok, ps.items, dev, a.batch, lettura)
                else:
                    ctx = CongelaSwiGLU(model, arch, g, spec,
                                        {b: cache[spec][b] for b in g})
                    with ctx:
                        H = leggi(model, tok, ps.items, dev, a.batch, lettura, ctx)
            except RuntimeError as e:
                print("%-40s %6s   %s" % (nome, str(g), e))
                continue
            m, s = auc_su_assi(H, pidx, assi)
            print("%-40s %6s %8.3f %+8.3f" % (nome, str(g), m, m - base))

    print("\nCOME SI LEGGE")
    if a.modo == "direzionale":
        print("  Se togliere l'ORTOGONALE non danneggia, la rotazione non era")
        print("  la causa. Se togliere la PARALLELA non danneggia, non lo era")
        print("  l'erosione. Confronta sempre col controllo casuale: se il danno")
        print("  e' lo stesso, stai misurando il disturbo e non la direzione.")
    else:
        print("  Un delta POSITIVO significa che congelare quella componente")
        print("  MIGLIORA la leggibilita' a valle: quella componente stava")
        print("  spingendo contro l'asse. Un delta negativo significa che")
        print("  portava verita' e togliergliela costa.")
    print("  Il verdetto e' tuo.")


if __name__ == "__main__":
    main()
