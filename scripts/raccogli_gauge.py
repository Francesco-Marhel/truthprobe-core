#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""raccogli_gauge.py  --  eigengap, frustration and margins from gauge bundles.

Reads the files produced by the gauge stage and reports, for each bundle, the
relative gap between the first and second eigenvalue, how many categories were
flipped, and how many fall below the identifiability threshold.

THE EIGENGAP IS AN A PRIORI CONDITION: it is computed from the cosine matrix
alone, before any sign is assigned and without reference to any result. That is
why it is reported alongside every bundle rather than after seeing whether the
gauge worked. It is also invariant under the gauge itself: S C S with S diagonal
of plus and minus one is a similarity transformation, so the spectrum does not
change.

FRUSTRATION is the fraction of triangles whose cycle product is negative. It is
gauge-invariant for the same reason the cycle products are: every index appears
twice in the product, so the signs cancel. It therefore measures an obstruction
in the geometry rather than a choice of orientation.

If a bundle does not carry the field (older bundles, or a gauge not yet run),
the eigengap is recomputed from the matrix, which is all that is needed.

    python raccogli_gauge.py campagne_k33
    python raccogli_gauge.py campagne_k33 --filtro granite --latex
"""

import argparse
import json
import math
import os
import statistics as st
import sys


def autovalori(M):
    """I due autovalori maggiori per iterazione della potenza con deflazione.

    Bastano i primi due, e la matrice e' piccola (K per K con K <= 33), quindi
    non vale la pena tirarsi dietro numpy per questo. La deflazione e' esatta
    perche' una matrice di coseni e' simmetrica e i suoi autovettori sono
    ortogonali.
    """
    K = len(M)

    def potenza(A, iters=500):
        v = [1.0 / math.sqrt(K)] * K
        lam = 0.0
        for _ in range(iters):
            w = [sum(A[i][j] * v[j] for j in range(K)) for i in range(K)]
            n = math.sqrt(sum(x * x for x in w))
            if n < 1e-12:
                return 0.0, v
            v = [x / n for x in w]
            lam = sum(v[i] * sum(A[i][j] * v[j] for j in range(K))
                      for i in range(K))
        return lam, v

    l1, u1 = potenza([r[:] for r in M])
    D = [[M[i][j] - l1 * u1[i] * u1[j] for j in range(K)] for i in range(K)]
    l2, _ = potenza(D)
    return l1, abs(l2), u1


def frustrazione(M):
    """Frazione di triangoli sbilanciati, nel senso di Harary.

    Un triangolo e' BILANCIATO quando il prodotto dei tre coseni e' positivo,
    e frustrato altrimenti. La quantita' e' INVARIANTE DI GAUGE: cambiando i
    segni degli assi ogni indice compare due volte nel prodotto, quindi
    s_j^2 = 1 e il prodotto non cambia. Misura quindi un'ostruzione della
    geometria, non una scelta di orientamento.

    E' il complemento naturale dell'eigengap: quello dice quando il segno e'
    identificabile, questa dice quanta parte della struttura non e'
    orientabile affatto. Un insieme con frustrazione nulla ammette
    un'assegnazione che rende tutti i coseni positivi; con frustrazione
    positiva nessuna assegnazione ci riesce.
    """
    K = len(M)
    tot = frus = 0
    for i in range(K):
        for j in range(i + 1, K):
            for k in range(j + 1, K):
                p = M[i][j] * M[j][k] * M[i][k]
                tot += 1
                if p < 0:
                    frus += 1
    return (frus / tot) if tot else float("nan")


def leggi(p):
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    M = d.get("cos_peak") or d.get("cos")
    if M is None and "matrice" in d:
        M = d["matrice"]
    if M is None:
        return None
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else d
    K = len(M)
    gap = d.get("eigengap")
    if gap is None:
        l1, l2, u = autovalori(M)
        gap = (l1 - l2) / l1 if l1 > 1e-12 else float("nan")
    else:
        _, _, u = autovalori(M)
    marg = d.get("margini")
    if isinstance(marg, dict):
        val = list(marg.values())
    else:
        val = list(u)
        s = 1.0 if sum(1 for x in val if x > 0) * 2 >= K else -1.0
        m = max(abs(x) for x in val) or 1.0
        val = [s * x / m for x in val]
    girate = sum(1 for x in val if x < 0)
    sottili = sum(1 for x in val if abs(x) < 0.10)
    return dict(
        file=os.path.basename(p),
        frus=frustrazione(M),
        modello=str(meta.get("model", meta.get("modello", "?")))
        .rstrip("/\\").replace("\\", "/").split("/")[-1],
        K=K, gap=float(gap), girate=girate, sottili=sottili,
        n=meta.get("pairs_per_relation", meta.get("n", "?")),
        sfx=meta.get("sentence_suffix"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cartella")
    ap.add_argument("--filtro", default=None)
    ap.add_argument("--latex", action="store_true",
                    help="stampa le righe pronte per una tabella LaTeX")
    a = ap.parse_args()

    righe = []
    for radice, _, nomi in os.walk(a.cartella):
        for n in sorted(nomi):
            if not n.endswith("_gauge.json"):
                continue
            if a.filtro and a.filtro.lower() not in n.lower():
                continue
            r = leggi(os.path.join(radice, n))
            if r:
                righe.append(r)
    if not righe:
        sys.exit("[stop] nessun file *_gauge.json in %s" % a.cartella)

    # una riga per modello: media sui semi
    per = {}
    for r in righe:
        per.setdefault((r["modello"], r["K"]), []).append(r)

    if a.latex:
        print("%% modello & K & semi & eigengap & sd & frustrazione")
        for (m, K), v in sorted(per.items()):
            g = [x["gap"] for x in v]
            print("%s & %d & %d & $%.3f$ & $%.3f$ & $%.3f$ \\\\"
                  % (m.replace("_", r"\_"), K, len(v), st.mean(g),
                     st.pstdev(g) if len(g) > 1 else 0.0,
                     st.mean([x["frus"] for x in v])))
        return

    print("=" * 74)
    print("GAUGE: eigengap relativo e margini")
    print("=" * 74)
    print("  %-22s %3s %5s %8s %8s %9s %7s %7s"
          % ("modello", "K", "semi", "gap", "ds", "frustraz.", "girate", "sottili"))
    for (m, K), v in sorted(per.items()):
        g = [x["gap"] for x in v]
        f = [x["frus"] for x in v]
        print("  %-22s %3d %5d %8.3f %8.3f %9.3f %7.1f %7.1f"
              % (m, K, len(v), st.mean(g),
                 st.pstdev(g) if len(g) > 1 else 0.0, st.mean(f),
                 st.mean([x["girate"] for x in v]),
                 st.mean([x["sottili"] for x in v])))
    tutti = [x["gap"] for x in righe]
    fr = [x["frus"] for x in righe]
    print("\n  su %d bundle: gap medio %.3f (da %.3f a %.3f), "
          "frustrazione media %.3f (da %.3f a %.3f)"
          % (len(righe), st.mean(tutti), min(tutti), max(tutti),
             st.mean(fr), min(fr), max(fr)))
    if len(righe) > 2:
        mg, mf = st.mean(tutti), st.mean(fr)
        sg = math.sqrt(sum((x - mg) ** 2 for x in tutti))
        sf = math.sqrt(sum((x - mf) ** 2 for x in fr))
        if sg > 1e-9 and sf > 1e-9:
            r = sum((x - mg) * (y - mf) for x, y in zip(tutti, fr)) / (sg * sf)
            print("  correlazione fra eigengap e frustrazione: %+.3f su %d viste"
                  % (r, len(righe)))
    sfx = {r["sfx"] for r in righe}
    if len(sfx) > 1:
        print("  ATTENZIONE: convenzioni della frase diverse fra i bundle (%s)."
              " I loro coefficienti non sono confrontabili." % sorted(map(repr, sfx)))
    print("\n  Un gap piccolo significa che l'autovettore principale ruota sotto")
    print("  piccole perturbazioni della matrice, quindi i segni che dichiara")
    print("  non sono stabili. E' una condizione calcolata PRIMA di assegnare")
    print("  qualsiasi segno, non una diagnosi dopo il fatto.")
    print("\n  La frustrazione e' invariante di gauge: nessuna assegnazione di")
    print("  segni la cambia. Misura quanta parte della struttura non e'")
    print("  orientabile, ed e' quindi una proprieta' della geometria.")


if __name__ == "__main__":
    main()
