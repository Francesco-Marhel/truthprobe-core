#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mantel_per_livello.py  --  a che profondita' nasce l'accordo fra due modelli?

crea_dizionario --all-layers salva nel .pt la matrice dei coseni fra categorie
a OGNI livello, sotto la chiave cos_by_layer. Questo tool ne fa il Mantel
livello per livello fra due bundle, e stampa la curva.

A COSA SERVE. Il controllo di superficie a un blocco fisso dice quanto
dell'accordo e' gia' presente "presto". La curva completa dice qualcosa di
piu': DOVE l'accordo nasce, se cresce, e se una parte era gia' li' prima che
il modello calcolasse qualsiasi cosa.

IL LIVELLO 0 E' IL PUNTO CHE DECIDE. Sono gli embedding, prima di ogni blocco.

  senza suffisso: l'ultimo token e' il target, che DIFFERISCE fra i due membri
      della coppia. Un accordo alto al livello 0 e' identita' di token
      condivisa fra i due modelli, non geometria: due tokenizer che segmentano
      i target in modo simile producono differenze simili.

  con suffisso '.': l'ultimo token e' il punto in ENTRAMBE le frasi, quindi la
      differenza di coppia e' esattamente zero e l'asse non esiste. Il livello
      0 e' degenere per costruzione e va letto come non definito, non come
      accordo nullo.

Per questo la curva ha senso soprattutto sui bundle SENZA suffisso: e' l'unico
modo di misurare quanto della disposizione condivisa e' lessicale.

    python mantel_per_livello.py --a bundle1.pt --b bundle2.pt
    python mantel_per_livello.py --a b1.pt --b b2.pt --perms 999 --soglia 0.6
"""

import argparse
import sys

try:
    import arrangement_stress_test as A
except ImportError:
    sys.exit("serve arrangement_stress_test.py nella stessa cartella")

import torch


def carica(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if "cos_by_layer" not in d:
        sys.exit("[stop] %s non contiene cos_by_layer.\n"
                 "  Va rigenerato con --all-layers." % p)
    cbl = d["cos_by_layer"]
    if hasattr(cbl, "tolist"):
        cbl = cbl.tolist()
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    tr = d.get("transfer")
    return dict(cats=list(d["cats"]), cbl=cbl, meta=meta,
                transfer=tr.tolist() if hasattr(tr, "tolist") else tr,
                nome=(meta.get("model") or d.get("model") or p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--perms", type=int, default=999)
    ap.add_argument("--soglia", type=float, default=None,
                    help="se dato, calcola anche la versione ristretta alle "
                         "categorie sopra soglia su ENTRAMBI (serve transfer)")
    x = ap.parse_args()

    A.SENZA_SEGNO = False
    P, Q = carica(x.a), carica(x.b)
    if P["cats"] != Q["cats"]:
        sys.exit("[stop] insiemi di categorie diversi")
    K = len(P["cats"])
    nP, nQ = len(P["cbl"]), len(Q["cbl"])
    if nP != nQ:
        print("[nota] profondita' diverse: %d contro %d livelli. Si confrontano"
              " i livelli per indice, il che ha senso solo se i due modelli"
              " hanno lo stesso numero di blocchi." % (nP, nQ))
    n = min(nP, nQ)

    print("=" * 62)
    print("MANTEL PER LIVELLO   K=%d   %d livelli" % (K, n))
    print("=" * 62)
    for e, b in (("A", P), ("B", Q)):
        m = b["meta"]
        sfx = m.get("sentence_suffix")
        print("[%s] %s   suffisso %s   blocco %s   n %s" %
              (e, b["nome"],
               ("%r" % sfx) if sfx is not None else "NON REGISTRATO",
               m.get("peak_block", "?"), m.get("pairs_per_relation", "?")))

    idx = list(range(K))
    keep = None
    if x.soglia is not None and P["transfer"] and Q["transfer"]:
        keep = [i for i in idx
                if P["transfer"][i][i] >= x.soglia and Q["transfer"][i][i] >= x.soglia]
        print("[ristretto] %d categorie sopra %.2f su entrambi" % (len(keep), x.soglia))
        if len(keep) < 4:
            keep = None
            print("  troppo poche: la colonna ristretta non viene calcolata")

    print("\n  %-8s %9s %9s %s" % ("livello", "Mantel", "p",
                                  "ristretto" if keep else ""))
    for lev in range(n):
        Ma, Mb = P["cbl"][lev], Q["cbl"][lev]
        try:
            r, p = A.mantel(Ma, Mb, idx, x.perms)
        except Exception as e:
            print("  %-8d   non calcolabile (%s)" % (lev, type(e).__name__))
            continue
        extra = ""
        if keep:
            rr, _ = A.mantel(Ma, Mb, keep, x.perms)
            extra = "%+9.3f" % rr
        marca = "  <- embedding" if lev == 0 else ""
        print("  %-8d %+9.3f %9.4f %s%s" % (lev, r, p, extra, marca))

    print("\nCOME SI LEGGE")
    print("  Livello 0 alto SENZA suffisso: parte dell'accordo e' identita' di")
    print("  token condivisa fra i due tokenizer, non geometria appresa.")
    print("  Livello 0 con suffisso '.': la differenza di coppia e' zero per")
    print("  costruzione, quindi il valore non e' definito e va ignorato.")
    print("  La quantita' che porta il claim e' il MARGINE fra il picco e i")
    print("  livelli iniziali, non il valore al picco da solo.")


if __name__ == "__main__":
    main()
