# -*- coding: utf-8 -*-
"""
truthprobe.geometry

L'asse e le sue proiezioni. Le funzioni qui sono portate da truth_probe.py
mantenendo la semantica esatta, incluse le scelte che sembrano dettagli e non
lo sono, elencate sotto. Il test di regressione verifica che i numeri
coincidano con quelli canonici.

TRE SCELTE CHE VANNO CONSERVATE

1. L'ORIENTAMENTO USA LE MEDIE DELLE PROIEZIONI, non la somma delle
   differenze. Sono quasi sempre d'accordo, ma non sempre: con distribuzioni
   asimmetriche possono divergere, e il bit di segno cambierebbe.

2. LA CALIBRAZIONE E' ROBUSTA. med e scala vengono dalla mediana e dalla
   deviazione assoluta mediana per 1.4826, non da media e deviazione standard.
   Serve perche' pochi stati a norma alta sposterebbero la calibrazione.

3. v2 E' ORTOGONALIZZATO CONTRO v1 esplicitamente. La SVD lo darebbe gia'
   ortogonale, ma dopo il possibile cambio di segno di v1 la rimozione della
   componente e' fatta comunque, e la si conserva.

Il Lemma di identificazione senza etichette vale su fit_axis: la matrice di
Gram delle righe di differenza non dipende da quale elemento della coppia si
designa vero, quindi lo span di v1 e' invariante. L'argomento orient serve
solo al null di permutazione e all'orientamento finale.
"""

import torch


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def ang_diff(a, b):
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def robust_ms(c):
    """mediana e scala robusta (MAD per 1.4826), come in truth_probe."""
    med = c.median()
    scale = (1.4826 * (c - med).abs().median()).clamp_min(1e-8)
    return float(med), float(scale)


def fit_axis(Hl, pidx, orient=None):
    """Asse di verita' a un livello, dalla SVD delle differenze intra-coppia.

    Hl    [N, d] stati a un livello
    pidx  lista di (indice_vero, indice_falso)
    orient  opzionale, +1 o -1 per coppia: scambia i due lati. Serve al null di
            permutazione. Per il Lemma lo span di v1 non ne dipende.

    Restituisce il dizionario dell'asse con v1, v2 e la calibrazione, nella
    stessa forma che project_fields si aspetta.
    """
    if orient is None:
        orient = [1] * len(pidx)
    Hl = Hl.float()
    t_idx, f_idx = [], []
    for (it, iff), o in zip(pidx, orient):
        (t_idx if o >= 0 else f_idx).append(it)
        (f_idx if o >= 0 else t_idx).append(iff)

    D = Hl[t_idx] - Hl[f_idx]
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    v1 = Vh[0].clone()
    v2 = Vh[1] - torch.dot(Vh[1], v1) * v1
    v2 = v2 / v2.norm().clamp_min(1e-8)

    proj_t, proj_f = Hl[t_idx] @ v1, Hl[f_idx] @ v1
    if proj_t.mean() < proj_f.mean():          # orientamento: un bit di segno
        v1 = -v1
        proj_t, proj_f = -proj_t, -proj_f

    states = torch.cat([Hl[t_idx], Hl[f_idx]], 0)
    c1, c2 = states @ v1, states @ v2
    med1, s1 = robust_ms(c1)
    med2, s2 = robust_ms(c2)
    Re = (c1 - med1) / s1
    Im = (c2 - med2) / s2
    r = float(torch.sqrt(Re ** 2 + Im ** 2).median().clamp_min(1e-8))

    Re_t = (proj_t - med1) / s1
    Im_t = ((Hl[t_idx] @ v2) - med2) / s2
    th = torch.atan2(Im_t, Re_t)
    th_true = float(torch.atan2(torch.sin(th).mean(), torch.cos(th).mean()))

    return dict(v1=v1, v2=v2, med1=med1, s1=s1, med2=med2, s2=s2,
                r=r, th_true=th_true)


def project_fields(Hl, ax):
    """Le coordinate calibrate di uno stato rispetto a un asse.

    Re    la posizione sull'asse, l'unica coordinata che porta verita' secondo
          i sette esperimenti di falsificazione
    Im    la seconda direzione, varianza residua sulle coppie a polarita' unica
    b     Re passato per una sigmoide
    m     il modulo, cioe' l'energia
    theta l'argomento, cioe' la fase
    """
    Hl = Hl.float()
    Re = (Hl @ ax["v1"] - ax["med1"]) / ax["s1"]
    Im = (Hl @ ax["v2"] - ax["med2"]) / ax["s2"]
    b = torch.sigmoid(Re)
    m = torch.sqrt(Re ** 2 + Im ** 2)
    theta = torch.atan2(Im, Re)
    risk = (0.5 - b) * torch.tanh(m / max(ax["r"], 1e-8))
    pdev = ang_diff(theta, ax["th_true"]).abs()
    return dict(Re=Re, Im=Im, b=b, m=m, theta=theta, risk=risk, phase_dev=pdev)


def axis_vector(ax):
    """v1 unitario, per chi vuole solo la direzione."""
    return unit(ax["v1"])


def cosine_matrix(axes):
    """Matrice KxK dei coseni con SEGNO fra assi orientati ciascuno dentro la
    propria categoria. Il segno e' informazione: negativo significa che la
    direzione condivisa legge la verita' dell'altra categoria al contrario.
    Ma dipende dall'orientamento di ENTRAMBE le categorie coinvolte: solo i
    prodotti sui cicli chiusi sono invarianti (vedi stats.frustration)."""
    A = torch.stack([unit(a["v1"] if isinstance(a, dict) else a) for a in axes], 0)
    return A @ A.T, A


def arrangement_by_layer(H_all, pidx, cat_pairs, layers=None, verbose=True):
    """The category arrangement at EVERY layer, in one call.

    H_all is [N, L+1, d], the whole residual stack from a single forward pass,
    so thirty maps cost one extraction and no extra passes: the states are
    already there.

    Returns {layer: (axes [K, d], cosine matrix [K, K])}.

    A caution that is not optional. At shallow layers the axes are weak and
    their orientation has no stable reference, so the SIGNS of those matrices
    are not comparable with the peak's: the early matrix is a surface control,
    to be read for whether structure exists at all, not for which structure.
    Use the unsigned or cycle-product forms when comparing across depth, and
    gauge only where the eigengap says the signs are identifiable."""
    L = H_all.shape[1]
    layers = list(range(L)) if layers is None else list(layers)
    cats = sorted(cat_pairs)
    out = {}
    for n, l in enumerate(layers):
        H = H_all[:, l, :]
        ax = torch.stack([unit(fit_axis(H, cat_pairs[c])["v1"]) for c in cats], 0)
        out[l] = (ax, ax @ ax.T)
        if verbose:
            print("\r  [arrangement] layer %d/%d" % (n + 1, len(layers)),
                  end="", flush=True)
    if verbose:
        print()
    return out


def subspace_fraction(D, A):
    """Frazione della norma quadrata di ogni riga di D che cade nello span
    delle righe di A. Il valore atteso a caso e' circa righe(A) / colonne(A):
    va sempre riportato accanto, altrimenti il numero non si legge."""
    Q, _ = torch.linalg.qr(A.T)
    n2 = D.pow(2).sum(1).clamp_min(1e-12)
    return (D @ Q).pow(2).sum(1) / n2
