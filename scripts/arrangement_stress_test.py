"""
arrangement_stress_test.py -- does the arrangement law survive at scale?

The registered law (R5) is CROSS-FAMILY AGREEMENT at matched protocol,
not stability of one family's matrix across sample sizes. This tool
compares two dictionaries (one per family, same pairs-per-relation) and
prints the Mantel test on the signed peak cosines, the early-block
surface control, and a restricted variant on the categories whose
within-AUC clears a pre-declared threshold on BOTH models. It also
reports, against optional reference dictionaries (the 60-pair ones),
which categories look sign-flipped. No GPU, no model: pure arithmetic
on files already produced.

Examples (from the folder with the JSONs):
  python arrangement_stress_test.py --a dizionari/qwen_888.json --b dizionari/llama_888.json \
      --ref-a dizionari/qwen_60.json --ref-b dizionari/llama_60.json --auc-threshold 0.6
  python arrangement_stress_test.py --a qwen_888.json --b llama_888.json --exclude P176
"""
import argparse, json, math, os, random

PROVENIENZA = ("model", "peak_block", "write_layer", "early_block",
               "k_relations", "pairs_per_relation", "sentence_suffix",
               "seed", "revision", "dataset_revision")

def load_dict(path):
    if path.endswith(".pt") or path.endswith(".pts"):
        import torch
        d = torch.load(path, map_location="cpu")
        meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        out = {}
        for k in ("cats", "cos_peak", "cos_early", "transfer"):
            v = d.get(k)
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        # la provenienza sta dentro meta nei bundle di crea_dizionario e in
        # cima in quelli piu' vecchi: si guarda in entrambi i posti
        for k in PROVENIENZA:
            v = d.get(k)
            out[k] = meta.get(k) if v is None else v
        out["k_relations"] = out.get("k_relations") or len(out["cats"])
        return out
    with open(path) as f:
        j = json.load(f)
    j["k_relations"] = j.get("k_relations") or len(j["cats"])
    return j


def confronta_protocollo(A, B, forza=False):
    """I due bundle sono stati costruiti nello stesso modo?

    Il Mantel confronta due matrici di coseni voce per voce. Se i due bundle
    nascono da protocolli diversi il numero esce lo stesso, plausibile, e non
    significa niente. Tre campi lo decidono.

    k_relations   matrici di dimensione diversa: il confronto non esiste.
    pairs_per_relation  stesso K ma assi stimati con precisione diversa: il
                  soffitto di attenuazione cambia, e un Mantel piu' basso
                  puo' essere solo rumore in piu'.
    sentence_suffix  con e senza punto finale gli assi stanno a coseno +0.52,
                  mentre la disposizione sopravvive a Mantel +0.775: un
                  confronto misto mescola due geometrie.

    Un campo assente non blocca (i bundle vecchi non lo scrivevano) ma viene
    detto, perche' non verificabile non vuol dire uguale.
    """
    problemi, ignoti = [], []
    for campo, perche in (
            ("k_relations", "le matrici hanno dimensioni diverse"),
            ("pairs_per_relation", "gli assi hanno precisione diversa"),
            ("sentence_suffix", "gli assi stanno a coseno +0.52 fra le due convenzioni")):
        va, vb = A.get(campo), B.get(campo)
        if va is None or vb is None:
            ignoti.append(campo)
        elif va != vb:
            problemi.append("  %-20s A=%r  B=%r   -> %s" % (campo, va, vb, perche))
    if ignoti:
        print("[provenienza] NON VERIFICABILE: %s assente in almeno uno dei due "
              "bundle. Non verificabile non vuol dire uguale." % ", ".join(ignoti))
    if problemi:
        print("\n[protocollo] i due bundle non sono confrontabili:")
        for p in problemi:
            print(p)
        if not forza:
            raise SystemExit("[stop] rigenera uno dei due allo stesso protocollo, "
                             "oppure passa --forza se sai perche' lo stai facendo.")
        print("[forza] si procede lo stesso: il numero sotto non e' confrontabile "
              "con la tabella della tesi.")

SENZA_SEGNO = False

def offdiag(M, idx):
    """Le voci fuori diagonale, con o senza segno.

    Senza segno il test misura se le categorie sono VICINE o LONTANE allo
    stesso modo nei due bundle, ignorando da che parte punta ciascun asse.
    Serve a separare due diagnosi che il Mantel firmato confonde:

      firmato basso, non firmato alto  -> la disposizione e' stabile ma il
        gauge non identifica l'orientamento a questo K e a questo n. Una sola
        categoria girata inverte una riga e una colonna intere.

      firmato basso, non firmato basso -> la disposizione stessa non e'
        ripetibile: gli assi per categoria sono stimati troppo male.

    NON e' un rimpiazzo del test firmato: perde l'informazione di polarita',
    che e' parte della legge. E' una diagnosi.
    """
    v = [M[i][j] for i in idx for j in idx if i < j]
    return [abs(x) for x in v] if SENZA_SEGNO else v

def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    if sx == 0 or sy == 0: return float("nan")
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy)

def mantel(Ma, Mb, idx, perms=9999, seed=0):
    x = offdiag(Ma, idx); y = offdiag(Mb, idx)
    r_obs = pearson(x, y)
    rng = random.Random(seed); ge = 1
    for _ in range(perms):
        p = list(idx); rng.shuffle(p)
        yp = [Mb[i][j] if i < j else Mb[j][i] for a, i in enumerate(p) for b, j in enumerate(p) if a < b]
        if SENZA_SEGNO:
            yp = [abs(x) for x in yp]
        if pearson(x, yp) >= r_obs: ge += 1
    return r_obs, ge/(perms+1)

def spearman(x, y):
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for k, i in enumerate(s): r[i] = k
        return r
    return pearson(ranks(x), ranks(y))

def best_sign_flips(M_new, M_ref, cats):
    """Greedy: flip category signs in M_new to maximize agreement with M_ref.
    Reports which flips help -> suspected orientation flips."""
    K = len(cats); signs = [1]*K
    def corr(sg):
        x = [sg[i]*sg[j]*M_new[i][j] for i in range(K) for j in range(K) if i < j]
        y = [M_ref[i][j] for i in range(K) for j in range(K) if i < j]
        return pearson(x, y)
    improved = True
    while improved:
        improved = False
        for i in range(K):
            base = corr(signs); signs[i] *= -1
            if corr(signs) > base + 1e-9: improved = True
            else: signs[i] *= -1
    return [c for c, s in zip(cats, signs) if s < 0], corr(signs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dictionary of family A (e.g. Qwen @888)")
    ap.add_argument("--b", required=True, help="dictionary of family B (e.g. Llama @888)")
    ap.add_argument("--ref-a", default=None, help="reference for A (e.g. Qwen @60)")
    ap.add_argument("--ref-b", default=None, help="reference for B (e.g. Llama @60)")
    ap.add_argument("--auc-threshold", type=float, default=0.6)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--perms", type=int, default=9999)
    ap.add_argument("--senza-segno", action="store_true",
                    help="Mantel sul VALORE ASSOLUTO dei coseni: diagnosi che "
                         "separa instabilita' del gauge da instabilita' della "
                         "disposizione")
    ap.add_argument("--forza", action="store_true",
                    help="calcola anche se i protocolli divergono. Il numero "
                         "che esce non e' confrontabile con la tabella")
    a, b = None, None
    args = ap.parse_args()
    global SENZA_SEGNO
    SENZA_SEGNO = args.senza_segno
    if SENZA_SEGNO:
        print("[modo] coseni SENZA SEGNO: si misura vicino/lontano, non la polarita'")
    A, B = load_dict(args.a), load_dict(args.b)
    assert A["cats"] == B["cats"], "category sets differ"
    cats = A["cats"]; K = len(cats)
    print(f"[A] {A.get('model') or '?'}   [B] {B.get('model') or '?'}   K={K}")
    print("[A] blocco %s  n=%s  suffisso %r  seed %s"
          % (A.get("peak_block"), A.get("pairs_per_relation"),
             A.get("sentence_suffix"), A.get("seed")))
    print("[B] blocco %s  n=%s  suffisso %r  seed %s"
          % (B.get("peak_block"), B.get("pairs_per_relation"),
             B.get("sentence_suffix"), B.get("seed")))
    confronta_protocollo(A, B, args.forza)

    idx_all = [i for i, c in enumerate(cats) if c not in args.exclude]
    if args.exclude: print(f"[excluded by flag] {args.exclude}")

    r, p = mantel(A["cos_peak"], B["cos_peak"], idx_all, args.perms)
    x, y = offdiag(A["cos_peak"], idx_all), offdiag(B["cos_peak"], idx_all)
    print(f"\nTHE LAW @ this protocol   Mantel r = {r:+.3f}   p = {p:.4f}   "
          f"Spearman = {spearman(x, y):+.3f}   (K'={len(idx_all)})")
    re, pe = mantel(A["cos_early"], B["cos_early"], idx_all, args.perms)
    print(f"surface control (early)   Mantel r = {re:+.3f}   p = {pe:.4f}")

    diagA = [A["transfer"][i][i] for i in range(K)]
    diagB = [B["transfer"][i][i] for i in range(K)]
    keep = [i for i in idx_all if diagA[i] >= args.auc_threshold and diagB[i] >= args.auc_threshold]
    print(f"\nwithin-AUC (A|B): " + "  ".join(f"{cats[i]} {diagA[i]:.2f}|{diagB[i]:.2f}" for i in range(K)))
    print(f"restricted set (within-AUC >= {args.auc_threshold} on BOTH): {[cats[i] for i in keep]}")
    if len(keep) >= 4:
        rr, rp = mantel(A["cos_peak"], B["cos_peak"], keep, args.perms)
        print(f"THE LAW, restricted       Mantel r = {rr:+.3f}   p = {rp:.4f}   (K'={len(keep)}; "
              f"note: with K'<6 the permutation floor limits p)")
    else:
        print("restricted set too small for a meaningful Mantel (need >= 4)")

    # --- symbiosis: per-category adherence vs knowledge proxy ---
    def row_adherence(i):
        x = [A["cos_peak"][i][j] for j in idx_all if j != i]
        y = [B["cos_peak"][i][j] for j in idx_all if j != i]
        return pearson(x, y)
    adh = {cats[i]: row_adherence(i) for i in idx_all}
    know = {cats[i]: min(diagA[i], diagB[i]) for i in idx_all}
    order = sorted(adh, key=lambda c: -adh[c])
    print("\nSYMBIOSIS  per-category adherence corr(row_A, row_B) vs knowledge proxy min(within-AUC):")
    for c in order:
        print(f"  {c:6s} adherence {adh[c]:+.3f}   knowledge {know[c]:.2f}")
    ks = [know[c] for c in order]; av = [adh[c] for c in order]
    print(f"  rank-correlation (Spearman) adherence ~ knowledge: {spearman(av, ks):+.3f}"
          f"   Pearson: {pearson(av, ks):+.3f}   (registered prediction: positive)")

    for tag, new, ref in (("A", args.a, args.ref_a), ("B", args.b, args.ref_b)):
        if not ref: continue
        R = load_dict(ref); assert R["cats"] == cats
        M_new = (A if tag == "A" else B)["cos_peak"]
        flips, c_after = best_sign_flips(M_new, R["cos_peak"], cats)
        c_before = pearson(offdiag(M_new, range(K)), offdiag(R["cos_peak"], range(K)))
        print(f"\n[{tag} vs its reference] corr before sign repair {c_before:+.3f} -> after {c_after:+.3f}"
              f"   suspected orientation flips: {flips if flips else 'none'}")

if __name__ == "__main__":
    main()
