# -*- coding: utf-8 -*-
"""
truthprobe.subspace

Confronti fra SOTTOSPAZI e misure spettrali.

Tutto il resto della libreria confronta direzioni singole, con i coseni, oppure
matrici di direzioni, con Mantel. Qui l'oggetto e' un altro: due sottospazi, e
la domanda e' quante dimensioni condividono.

E' una domanda che il lavoro sulla serie pone gia' senza avere lo strumento
per rispondere direttamente. L'affermazione che l'arrangement dell'attenzione
copre circa meta' delle dimensioni effettive di quello dell'FFN dopo correzione
per affidabilita' e' una frase sui sottospazi, stimata per via indiretta. Gli
angoli principali la misurano.

QUATTRO STRUMENTI, E COSA RISPONDONO

  principal_angles   dati due insiemi di direzioni, quanti angoli sono piccoli.
                     Uno spettro, non un numero: due sottospazi possono
                     condividere tre dimensioni su otto ed essere ortogonali
                     nelle altre cinque, e un solo scalare lo nasconderebbe.

  effective_rank     quante dimensioni un insieme di direzioni occupa DAVVERO,
                     cioe' l'esponenziale dell'entropia spettrale. Non conta
                     quante direzioni ci sono ma quanto sono indipendenti.

  spectral_entropy   l'entropia di von Neumann della matrice di Gram
                     normalizzata a traccia uno. Invariante di gauge, perche'
                     dipende solo dallo spettro. E' l'eigengap generalizzato a
                     tutto lo spettro: dove l'eigengap guarda il primo salto,
                     questa guarda l'intera distribuzione.

  cka                somiglianza fra due insiemi di RAPPRESENTAZIONI, invariante
                     per rotazione e scala. E' per confrontare stati, non assi:
                     sugli assi il coseno e Mantel dicono di piu' perche' il
                     segno e' informazione, mentre CKA lo butta via.

UNA CAUTELA CHE VALE PER TUTTI
Con d dimensioni e sottospazi di dimensione k, due sottospazi CASUALI hanno
gia' angoli piccoli quando k non e' trascurabile rispetto a d. Il valore atteso
a caso va sempre riportato accanto al risultato, ed e' per questo che ogni
funzione qui lo restituisce insieme al numero, invece di lasciarlo calcolare a
chi legge.
"""

import math

import torch


def _orthonormal(A):
    """Base ortonormale dello span delle RIGHE di A [k, d]. Restituisce [d, r]
    con r il rango effettivo: se le righe sono dipendenti, r e' minore di k, e
    quel fatto e' esso stesso un dato."""
    A = torch.as_tensor(A).float()
    Q, R = torch.linalg.qr(A.T)
    tol = 1e-6 * max(A.shape) * float(R.diagonal().abs().max().clamp_min(1e-30))
    keep = R.diagonal().abs() > tol
    return Q[:, keep]


def _angles(QA, QB):
    """Angoli principali fra due basi ORTONORMALI, in radianti crescenti.

    Il coseno da solo non basta. Calcolare l'angolo come arccos del valore
    singolare e' instabile quando l'angolo e' piccolo: li' il coseno vale quasi
    uno, e una perdita di precisione di 1e-7 sul coseno diventa 0.03 gradi
    sull'angolo. Per un sottospazio condiviso esattamente si vorrebbe zero, e
    quel residuo si scambierebbe per struttura.

    La cura e' quella classica di Bjorck e Golub: sopra 45 gradi si usa il
    coseno, sotto si usa il SENO, cioe' la norma della parte di QB che sporge
    dallo span di QA. Il seno e' accurato dove il coseno non lo e', e viceversa.
    """
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    ang = torch.arccos(s)
    piccoli = s > (1.0 / math.sqrt(2.0))
    if bool(piccoli.any()):
        # componente di QB ortogonale a QA: i suoi valori singolari sono i seni
        P = QB - QA @ (QA.T @ QB)
        sin = torch.linalg.svdvals(P).clamp(0.0, 1.0)
        sin = torch.sort(sin).values                 # crescente, come i coseni
        n = int(piccoli.sum())
        ang = ang.clone()
        ang[:n] = torch.arcsin(sin[:n])
    return s, ang


def principal_angles(A, B, degrees=True):
    """Angoli principali fra lo span delle righe di A e quello di B.

    Restituisce gli angoli in ordine crescente, i coseni corrispondenti, e il
    valore atteso per due sottospazi casuali delle stesse dimensioni.

    Il primo angolo e' zero quando i due sottospazi condividono una direzione
    esatta. Il numero di angoli sotto una soglia e' la dimensione condivisa. Se
    tutti gli angoli sono grandi i due sottospazi sono ortogonali, che con d
    grande e k piccolo e' il caso generico: per questo c'e' il riferimento.
    """
    QA, QB = _orthonormal(A), _orthonormal(B)
    d = QA.shape[0]
    ka, kb = QA.shape[1], QB.shape[1]
    s, ang = _angles(QA, QB)
    if degrees:
        ang = ang * 180.0 / math.pi

    # riferimento: due sottospazi casuali della stessa dimensione
    g = torch.Generator().manual_seed(0)
    ra = _orthonormal(torch.randn(ka, d, generator=g))
    rb = _orthonormal(torch.randn(kb, d, generator=g))
    _, ar = _angles(ra, rb)
    if degrees:
        ar = ar * 180.0 / math.pi

    return dict(angles=ang, cosines=s, dim_a=ka, dim_b=kb, d=d,
                random_angles=ar,
                shared_at_45=int((ang < (45.0 if degrees else math.pi / 4)).sum()),
                random_shared_at_45=int((ar < (45.0 if degrees else math.pi / 4)).sum()))


def subspace_overlap(A, B):
    """Frazione dell'energia dello span di A che cade dentro lo span di B.

    E' la media dei coseni al quadrato degli angoli principali, cioe' la
    generalizzazione a un sottospazio della frazione 'dentro' usata altrove per
    un singolo vettore. Il valore atteso a caso e' circa dim(B)/d."""
    r = principal_angles(A, B, degrees=False)
    return dict(overlap=float((r["cosines"] ** 2).mean()),
                chance=r["dim_b"] / r["d"],
                dim_a=r["dim_a"], dim_b=r["dim_b"])


def spectral_entropy(M, base="e", from_gram=True):
    """Entropia di von Neumann dello spettro.

    M puo' essere una matrice di Gram gia' pronta (from_gram=True) oppure un
    insieme di vettori riga, nel qual caso la Gram viene costruita.

    La matrice viene normalizzata a traccia uno, cosi' gli autovalori sono
    probabilita' e l'entropia e' quella di von Neumann in senso proprio. Gli
    autovalori negativi da errore numerico vengono azzerati, non ignorati in
    silenzio: se ce ne sono di grandi la matrice non e' semidefinita positiva e
    il numero non ha senso, quindi la funzione lo dichiara.

    Invariante di gauge: dipende solo dallo spettro, e cambiare i segni delle
    categorie e' una trasformazione di similarita' che lo lascia intatto."""
    M = torch.as_tensor(M).float()
    if not from_gram:
        M = M @ M.T
    K = M.shape[0]
    w = torch.linalg.eigvalsh((M + M.T) / 2)
    neg = float(w[w < 0].abs().max()) if (w < 0).any() else 0.0
    w = w.clamp_min(0.0)
    tot = float(w.sum())
    if tot <= 0:
        return dict(entropy=float("nan"), effective_rank=float("nan"),
                    max_entropy=math.log(K), negative_eig=neg)
    p = w / tot
    nz = p[p > 1e-12]
    H = float(-(nz * nz.log()).sum())
    if base == "2":
        H = H / math.log(2)
    return dict(entropy=H, effective_rank=math.exp(H if base == "e" else H * math.log(2)),
                max_entropy=math.log(K) if base == "e" else math.log2(K),
                n_dims=K, negative_eig=neg,
                warning=("spettro con autovalori negativi non trascurabili: la "
                         "matrice non e' semidefinita positiva e l'entropia non "
                         "e' interpretabile") if neg > 1e-4 * tot else "")


def effective_rank(A, from_gram=False):
    """Quante dimensioni un insieme di direzioni occupa davvero.

    K assi perfettamente ortogonali danno rango effettivo K; K assi tutti
    paralleli danno 1. E' la quantita' che serve quando si dice che un
    arrangement copre meno dimensioni di un altro."""
    r = spectral_entropy(A, from_gram=from_gram)
    return dict(effective_rank=r["effective_rank"], entropy=r["entropy"],
                n_dims=r["n_dims"], max_possible=r["n_dims"])


def cka(X, Y, unbiased=False):
    """Centered Kernel Alignment lineare fra due insiemi di rappresentazioni.

    X e Y sono [n, d1] e [n, d2] sulle STESSE n osservazioni, nello stesso
    ordine. Invariante per rotazione e scala isotropa, non per riscalatura per
    coordinata.

    Serve a confrontare STATI, non assi. Sugli assi il coseno con segno e
    Mantel dicono di piu', perche' nel lavoro di questa serie il segno e'
    informazione e CKA lo scarta per costruzione."""
    X = torch.as_tensor(X).float()
    Y = torch.as_tensor(Y).float()
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA vuole le stesse osservazioni: %d contro %d"
                         % (X.shape[0], Y.shape[0]))
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = float((X.T @ Y).pow(2).sum())
    xx = float((X.T @ X).pow(2).sum())
    yy = float((Y.T @ Y).pow(2).sum())
    den = math.sqrt(xx * yy)
    return float(xty / den) if den > 0 else float("nan")


# =====================================================================
#  valutazione di feature esterne, per esempio da un autoencoder sparso
# =====================================================================
def feature_alignment(features, axes, perms=1000, seed=0):
    """Quanto un insieme di feature esterne si allinea agli assi di categoria.

    features: [m, d] direzioni prodotte da chiunque, tipicamente le righe del
              decoder di un autoencoder sparso
    axes:     [K, d] gli assi di categoria del dizionario misurato

    Per ogni asse riporta il massimo |coseno| su tutte le feature, e lo confronta
    con un null di direzioni casuali della stessa dimensione. Il null e'
    necessario e non decorativo: con m feature in d dimensioni il massimo su m
    tentativi cresce con m anche se le feature sono casuali, quindi un valore
    alto senza null non dice nulla.

    QUESTA NON E' UNA IMPLEMENTAZIONE DI AUTOENCODER, ed e' una scelta. Questa
    libreria fornisce il riferimento misurato e il criterio; addestrare il
    modello concorrente e' compito di chi lo propone. La valutazione e' misura,
    l'addestramento e' un metodo, e tenerli separati e' cio' che rende il
    criterio dichiarabile in anticipo."""
    F = torch.as_tensor(features).float()
    A = torch.as_tensor(axes).float()
    F = F / F.norm(dim=1, keepdim=True).clamp_min(1e-12)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    m, d = F.shape
    K = A.shape[0]

    best = (A @ F.T).abs().max(dim=1)
    obs = best.values
    which = best.indices

    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(perms):
        R = torch.randn(m, d, generator=g)
        R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)
        null.append(float((A @ R.T).abs().max(dim=1).values.mean()))
    nt = torch.tensor(null)
    return dict(per_axis=obs, best_feature=which,
                mean=float(obs.mean()),
                null_mean=float(nt.mean()), null_p95=float(nt.quantile(0.95)),
                p=(1 + int((nt >= float(obs.mean())).sum())) / (perms + 1),
                n_features=m, n_axes=K, d=d)
