#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""riepilogo_mantel.py  --  reliability and law, from a folder of dictionaries.

With four models and four seeds the comparisons are 120, and reading them one by
one is not work a person should do. This tool runs them all and prints only the
summaries.

It REIMPLEMENTS NOTHING: it imports `mantel`, `pearson` and `offdiag` from
arrangement_stress_test.py, so the numbers here and those of a single comparison
run by hand are the same by construction. If that file changes, this one changes
with it.

THREE THINGS, IN THIS ORDER

RELIABILITY. The same model under different seeds. It is the noise floor: it
says how repeatable a measurement is before asking whether two models agree.
Without this number no agreement between models is interpretable.

LAW. Two different models. Split into two columns: same seed, that is the same
sentences, and crossed seeds, that is disjoint samples. The difference between
the two is the inflation due to a shared sample, measured rather than assumed.

CEILING. The square root of the product of the two reliabilities: the largest
agreement observable between two imperfect measures of the same thing. An
observed value close to the ceiling means nothing is left to explain; one
clearly below means the two do not measure entirely the same thing. The
corrected value is observed divided by ceiling, and it must ALWAYS be reported
next to the raw one: dividing by a low ceiling amplifies noise, and above 1 the
formula is outside its domain.

FULL AND RESTRICTED. Every row comes in two versions: over all categories, and
over only those whose within-category AUC clears the threshold on BOTH sides.
Where the model does not know the fact there is no truth direction, so that axis
is estimated on noise and its row in the matrix is not repeatable. The distance
between the two versions measures how much of the matrix the model cannot read.

    python riepilogo_mantel.py campagne_k33
    python riepilogo_mantel.py campagne_k33 --perms 0       (solo r, istantaneo)
    python riepilogo_mantel.py campagne_k33 --soglia 0.6 --perms 999
"""

import argparse
import itertools
import json
import os
import statistics as st
import sys

try:
    import arrangement_stress_test as A
except ImportError:
    sys.exit("need arrangement_stress_test.py in the same folder: both "
             "they must use the same arithmetic")


def carica(cartella):
    """Each *_gauge.json file in the folder, indexed by model and seed."""
    out = []
    for radice, _, nomi in os.walk(cartella):
        for n in sorted(nomi):
            if not n.endswith("_gauge.json"):
                continue
            p = os.path.join(radice, n)
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception as e:
                print("[salto] %s (%s)" % (p, type(e).__name__))
                continue
            mod = d.get("model") or "?"
            corto = str(mod).rstrip("/\\").replace("\\", "/").split("/")[-1]
            d["_file"] = p
            d["_modello"] = corto
            d["_seme"] = d.get("seed")
            d["_firma"] = (d.get("k_relations"), d.get("pairs_per_relation"),
                           d.get("sentence_suffix"))
            out.append(d)
    return out


def tieni(a, b, soglia):
    """Gli indici delle categorie che superano la soglia su ENTRAMBI i lati."""
    ta, tb = a.get("transfer"), b.get("transfer")
    if not ta or not tb:
        return None
    K = len(a["cats"])
    return [i for i in range(K) if ta[i][i] >= soglia and tb[i][i] >= soglia]


def confronta(a, b, perms, soglia):
    """Tutto quello che serve su una coppia di bundle.

    Restituisce un dizionario con r e p per il picco pieno, il picco ristretto
    e il CONTROLLO DI SUPERFICIE al blocco iniziale. Quest'ultimo e' la difesa
    contro l'obiezione che la disposizione sia struttura lessicale invece che
    semantica: al blocco iniziale il residuo e' ancora in gran parte identita'
    dei token, quindi li' l'accordo deve essere vicino a zero. Il numero che
    porta il claim non e' il Mantel al picco ma il MARGINE fra picco ed early:
    un picco a 0.93 con early a 0.65 vale meno di un picco a 0.79 con early a
    0.05.
    """
    if a["cats"] != b["cats"]:
        return None
    idx = list(range(len(a["cats"])))
    o = {}
    o["r"], o["p"] = A.mantel(a["cos_peak"], b["cos_peak"], idx, perms)
    if "cos_early" in a and "cos_early" in b:
        o["re"], o["pe"] = A.mantel(a["cos_early"], b["cos_early"], idx, perms)
    else:
        o["re"] = o["pe"] = None
    keep = tieni(a, b, soglia)
    o["K"] = len(keep) if keep is not None else None
    if keep is not None and len(keep) >= 4:
        o["rR"], o["pR"] = A.mantel(a["cos_peak"], b["cos_peak"], keep, perms)
    else:
        o["rR"] = o["pR"] = None
    return o


def pmax(v):
    """Il p PEGGIORE del gruppo. Mediare dei p non ha senso; quello che serve
    sapere e' se anche il confronto meno convincente regge."""
    v = [x for x in v if x is not None]
    return ("%.4f" % max(v)) if v else "n/d"


def ms(v):
    if not v:
        return "n/d", ""
    if len(v) == 1:
        return "%.3f" % v[0], ""
    return "%.3f" % st.mean(v), "±%.3f" % st.pstdev(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cartella")
    ap.add_argument("--soglia", type=float, default=0.6,
                    help="AUC dentro categoria minima perche' una categoria "
                         "entri nel set ristretto")
    ap.add_argument("--perms", type=int, default=199,
                    help="permutazioni per il p. Con 0 calcola solo r, ed e' "
                         "istantaneo")
    ap.add_argument("--senza-segno", action="store_true")
    a = ap.parse_args()

    A.SENZA_SEGNO = a.senza_segno
    B = carica(a.cartella)
    if not B:
        sys.exit("nessun *_gauge.json in %s" % a.cartella)

    firme = {d["_firma"] for d in B}
    print("=" * 74)
    print("RIEPILOGO   %d bundle   soglia %.2f   permutazioni %d%s"
          % (len(B), a.soglia, a.perms, "   SENZA SEGNO" if a.senza_segno else ""))
    print("=" * 74)
    if len(firme) > 1:
        print("[stop] protocolli diversi nella stessa cartella:")
        for f in sorted(firme, key=str):
            print("   K=%s n=%s suffisso=%r" % f)
        sys.exit("  separa le cartelle prima di confrontare.")
    K, n, sfx = firme.pop()
    print("protocollo: K=%s  n=%s  suffisso=%r" % (K, n, sfx))

    per_mod = {}
    for d in B:
        per_mod.setdefault(d["_modello"], []).append(d)
    for m in per_mod:
        per_mod[m].sort(key=lambda x: (x["_seme"] is None, x["_seme"]))
    print("modelli: " + ", ".join("%s (%d semi)" % (m, len(v))
                                  for m, v in sorted(per_mod.items())))

    # ---- affidabilita' ----------------------------------------------
    print("\n--- AFFIDABILITA' (stesso modello, semi diversi) -------------")
    print("  %-20s %5s %13s %13s %13s %8s"
          % ("modello", "cop.", "piena", "ristretta", "early", "p peggio"))
    aff = {}
    for m, v in sorted(per_mod.items()):
        pieni, ristr, ks, early, ps = [], [], [], [], []
        for x, y in itertools.combinations(v, 2):
            o = confronta(x, y, a.perms, a.soglia)
            if o is None:
                continue
            pieni.append(o["r"]); ps.append(o["p"])
            if o["rR"] is not None:
                ristr.append(o["rR"]); ks.append(o["K"])
            if o["re"] is not None:
                early.append(o["re"])
        if not pieni:
            print("  %-20s %5s   (un solo seme)" % (m, len(v)))
            continue
        mp, sp = ms(pieni); mr, sr = ms(ristr); me, se_ = ms(early)
        print("  %-20s %5d %8s%-5s %8s%-5s %8s%-5s %8s%s"
              % (m, len(pieni), mp, sp, mr, sr, me, se_, pmax(ps),
                 "   K'=%d" % round(st.mean(ks)) if ks else ""))
        aff[m] = dict(piena=st.mean(pieni),
                      ristretta=st.mean(ristr) if ristr else None)

    # ---- legge ------------------------------------------------------
    print("\n--- LEGGE (modelli diversi) ----------------------------------")
    print("  %-24s %13s %13s %13s %8s %8s"
          % ("coppia", "stesso seme", "incrociati", "early incr.", "margine",
             "p peggio"))
    incrociati = {}
    for m1, m2 in itertools.combinations(sorted(per_mod), 2):
        st_seme, cr_seme = [], []
        st_r, cr_r, cr_e, ps = [], [], [], []
        for x in per_mod[m1]:
            for y in per_mod[m2]:
                o = confronta(x, y, a.perms, a.soglia)
                if o is None:
                    continue
                stesso = (x["_seme"] is not None and x["_seme"] == y["_seme"])
                (st_seme if stesso else cr_seme).append(o["r"])
                if o["rR"] is not None:
                    (st_r if stesso else cr_r).append(o["rR"])
                if not stesso:
                    ps.append(o["p"])
                    if o["re"] is not None:
                        cr_e.append(o["re"])
        if not cr_seme and not st_seme:
            continue
        a1, b1 = ms(st_seme); a2, b2 = ms(cr_seme); a3, b3 = ms(cr_e)
        marg = (st.mean(cr_seme) - st.mean(cr_e)) if (cr_seme and cr_e) else None
        print("  %-24s %8s%-5s %8s%-5s %8s%-5s %8s %8s"
              % (m1 + " / " + m2, a1, b1, a2, b2, a3, b3,
                 ("%+.3f" % marg) if marg is not None else "n/d", pmax(ps)))
        if cr_seme:
            incrociati[(m1, m2)] = dict(piena=st.mean(cr_seme),
                                        ristretta=st.mean(cr_r) if cr_r else None)

    # ---- soffitto ---------------------------------------------------
    print("\n--- SOFFITTO E CORRETTO (su semi incrociati) -----------------")
    print("  %-26s %8s %8s %9s   %8s %8s %9s"
          % ("coppia", "oss.", "soff.", "corr.", "oss.R", "soff.R", "corr.R"))
    for (m1, m2), o in sorted(incrociati.items()):
        riga = "  %-26s" % (m1 + " / " + m2)
        for chiave, val in (("piena", o["piena"]), ("ristretta", o["ristretta"])):
            ra, rb = aff.get(m1, {}).get(chiave), aff.get(m2, {}).get(chiave)
            if val is None or ra is None or rb is None or ra <= 0 or rb <= 0:
                riga += " %8s %8s %9s" % ("n/d", "n/d", "n/d")
                continue
            s = (ra * rb) ** 0.5
            riga += " %8.3f %8.3f %9.3f" % (val, s, val / s)
            if val / s > 1.02:
                riga += "!"
        print(riga)
    print("\n  MARGINE = accordo incrociato al picco meno accordo al blocco")
    print("  iniziale. E' la quantita' che regge il claim: dice quanto")
    print("  dell'accordo NON e' spiegabile dalla struttura lessicale condivisa.")
    print("\n  Un corretto sopra 1 non e' un accordo migliore del possibile:")
    print("  e' il segno che la formula e' fuori dal suo dominio, di solito")
    print("  perche' un'affidabilita' e' troppo bassa per fare da denominatore.")


if __name__ == "__main__":
    main()
