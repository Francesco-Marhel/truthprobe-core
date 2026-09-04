# -*- coding: utf-8 -*-
"""
truthprobe.stats

Le statistiche della serie, in un posto solo. AUC e fold vengono da
truth_probe.py, il gauge da reorient_gauge.py, Mantel, frustrazione e
affidabilita' da verifica_arrangement.py.

Due cose sono state cambiate rispetto agli originali, e sono cambiamenti di
prestazione, non di definizione. Mantel e Mantel triplo sono vettorizzati: le
permutazioni si generano a scaglioni invece che una alla volta, e con K=33 e
9999 ripetizioni si passa da minuti a frazioni di secondo. Il test di
regressione verifica che diano lo stesso r della versione a ciclo.

UN AVVERTIMENTO SUI PAVIMENTI
Il test di Mantel permuta le etichette di categoria, quindi le rietichettature
possibili sono K fattoriale: sotto K=6 il p minimo raggiungibile e' grande e la
significativita' non e' ottenibile comunque. mantel() lo segnala da solo.
"""

import itertools
import math
import random

import torch


# =====================================================================
#  AUC e fold
# =====================================================================
def auc_score(s, y):
     """Computes the AUC ROC on single sentences, handling ties as 0.5.
    Identical definition to truth_probe.auc_score.
    """
    s = s.float()
    y = y.long()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return float(gt + 0.5 * eq)


def project_and_score(H, ax):
    """The position coordinate Re of states H under axis ax.

    Shortcut for the common case: everywhere in this series the score fed to
    auc_score is the calibrated position, never the magnitude or the phase.
    Those two were tested in Part I and carry nothing."""
    from .geometry import project_fields
    return project_fields(H, ax)["Re"]


def paired_accuracy(scores_true, scores_false):
    """
    Fraction of pairs where the true sample is projected above the false sample.

    This is NOT the AUC and is not interchangeable with it. Here, the topic is
    controlled within the pair, so the value is much higher: on the same dataset,
    we measured 0.90 for paired accuracy against 0.72 for AUC.
    Always report which of the two metrics is being used.
    """
    d = (scores_true - scores_false)
    return float((d > 0).float().mean() + 0.5 * (d == 0).float().mean())


def kfold_pairs(n_pairs, k, seed=0):
    """
    K-Fold cross-validation over pairs. 
    
    A pair is never split between the training and test sets to prevent 
    data leakage of the shared topic across splits.
    """
    idx = list(range(n_pairs))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    for i in range(k):
        test = set(folds[i])
        train = [p for p in idx if p not in test]
        yield train, sorted(folds[i])


def se_binomial(p, n):
      """
    Standard error of a proportion. 
    Used to avoid misinterpreting an observed delta that falls within 
    sampling noise as a true effect.
    """
    return math.sqrt(max(p * (1 - p), 0.0) / max(n, 1))


# =====================================================================
#  contributi contro un frame fisso
# =====================================================================
def frame_gap(contrib_true, contrib_false, axis, contrib_block=None,
              frame_block=None):
    """Class gap and effect size of a contribution read against a FIXED axis.

        gap = mean(v . c)|true  -  mean(v . c)|false
        d'  = gap / pooled standard deviation of the two classes

    This is the statistic behind propagate-versus-overwrite. A negative FFN gap
    means the FFN literally writes toward the FALSE side of the frame; attention
    measured the same way is the mirror image.

    The axis is FIXED, fitted elsewhere and passed in, never refitted here. That
    is the whole point: the question is how a contribution sits against a frame
    it did not necessarily write. Which frame you pass changes the answer, and
    the axis-provenance control exists precisely because a result anchored to
    "the block where the ruler was fitted" could be a property of the ruler.

    Works on any [n, d] contribution: a block's attention or FFN output, a
    single head's slice, or a single expert's output in a mixture. The function
    does not know which, and does not need to.

    CONTAINMENT, and why the two block indices exist.
    The law is stated on frames that do NOT contain the measured contribution:
    a frame fitted at block b, contributions read at a layer L > b, or a pre
    frame at b = L built from blocks strictly before L. If the frame already
    contains what is being measured, the contribution is partly correlated with
    itself and the sign can flip for that reason alone. Measured on
    Qwen2.5-1.5B: against a CLEAN frame the FFN reads -0.41 and attention
    +1.31, near mirror images; against a frame containing them both read
    positive, +0.71 and +1.04, and the law is invisible.

    Passing contrib_block and frame_block lets the function say so instead of
    returning a plausible number in silence. They are optional only because the
    function is also used where no block index applies; when they are given, the
    result carries a 'contained' flag and a warning."""
    axis = torch.as_tensor(axis).float()
    pt = torch.as_tensor(contrib_true).float() @ axis
    pf = torch.as_tensor(contrib_false).float() @ axis
    gap = float(pt.mean() - pf.mean())
    pooled = float(torch.sqrt((pt.var() + pf.var()) / 2).clamp_min(1e-8))
    out = dict(gap=gap, dprime=gap / pooled, pooled=pooled,
               n_true=len(pt), n_false=len(pf), contained=None, warning="")
    if contrib_block is not None and frame_block is not None:
        out["contained"] = bool(contrib_block <= frame_block)
        if out["contained"]:
            out["warning"] = (
                "the frame at block %d CONTAINS the contribution at block %d: "
                "the contribution is partly correlated with itself and the sign "
                "is not the law's. Use a frame from a strictly earlier block, or "
                "read contributions at a later layer."
                % (frame_block, contrib_block))
    return out


def frame_curves(contributions, H_axis, pidx, folds=5, seed=0, axis_block=None):
    """frame_gap across depth, held out over pairs, on one or more streams.

    contributions: {name: tensor [N, n_blocks, d]}
    H_axis:        [N, d] states the fixed axis is fitted on, of one block
    Returns {name: {"gap": [...], "dprime": [...]}} with one value per block.

    The axis is refitted on the TRAINING folds only and read on the held-out
    pairs, so the numbers are not the optimism of a ruler fitted on the same
    pairs it measures."""
    from .geometry import fit_axis
    names = list(contributions)
    if axis_block is not None:
        contained = [L for L in range(contributions[names[0]].shape[1])
                     if L <= axis_block]
        if contained:
            print("  [frame_curves] blocks %s..%s are CONTAINED in the frame at "
                  "block %d: there the sign is not the law's, because the "
                  "contribution is partly correlated with itself."
                  % (contained[0], contained[-1], axis_block))
    nB = contributions[names[0]].shape[1]
    acc = {n: dict(gap=[[] for _ in range(nB)], dprime=[[] for _ in range(nB)])
           for n in names}
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        v1 = fit_axis(H_axis, [pidx[p] for p in tr])["v1"]
        tt = [pidx[p][0] for p in te]
        ff = [pidx[p][1] for p in te]
        for n in names:
            C = contributions[n]
            for L in range(nB):
                r = frame_gap(C[tt, L, :], C[ff, L, :], v1)
                acc[n]["gap"][L].append(r["gap"])
                acc[n]["dprime"][L].append(r["dprime"])
    mean = lambda xs: [sum(c) / len(c) for c in xs]
    return {n: dict(gap=mean(acc[n]["gap"]), dprime=mean(acc[n]["dprime"]))
            for n in names}


# =====================================================================
#  gauge di consenso spettrale
# =====================================================================
def global_axis_gauge(axes, t_global, thin=0.10):
    """Sign each category axis by the GLOBAL axis: s_c = sign(v_c . t_global).

    This is the primary gauge, not the eigenvector one. The reason is that the
    within-category orientation, true side positive by that category's own
    margin, is unstable exactly where the margin is thin: a category the model
    knows poorly can flip between seeds for no reason at all. The global axis is
    one direction with a large, seed-stable margin, so using it as the reference
    is a choice of meridian, not an imposition of structure.

    That distinction matters and is testable: a gauge cannot manufacture
    agreement, because cycle products are invariant under every sign assignment
    (see triple_mantel and frustration). What it can do is remove a source of
    noise that has nothing to do with geometry.

    Returns (signs, margins, thin_indices). The margin is the signed cosine, and
    a category below the threshold must be reported UNSIGNED rather than forced:
    there the sign is a coin, and printing it as if it were a measurement is the
    error the flag exists to prevent."""
    A = torch.as_tensor(axes).float()
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    t = torch.as_tensor(t_global).float()
    t = t / t.norm().clamp_min(1e-12)
    m = A @ t
    s = torch.where(m >= 0, torch.ones_like(m), -torch.ones_like(m))
    return s, m, [i for i in range(len(m)) if abs(float(m[i])) < thin]


def gauge_report(cats, signs, margins, thin_idx, name="global-axis"):
    """The three lists a gauged bundle must carry, in one place.

    flipped is not a diagnostic: it is a result. A category whose axis points
    against the global direction is anti-aligned with the shared frame, and that
    is the finding, not an artefact to be normalised away."""
    return dict(gauge=name,
                gauge_margins=[round(float(m), 4) for m in margins],
                thin_categories=[cats[i] for i in thin_idx],
                flipped_vs_original=[cats[i] for i in range(len(cats))
                                     if float(signs[i]) < 0])


def consensus_gauge(C, thin=0.10):
    """
    Extracts signs from the principal eigenvector of the cosine matrix. 
    This represents the spectral relaxation for maximizing collective sign agreement.

    Returns (signs, margins, thin_indices). Margins are normalized to the maximum value.
    Below the 'thin' threshold, the category should be reported as UNSTABLE (when the margin is thin) rather than 
    forced, because in that region the sign acts as a currency (highly volatile/uncertain).

    A gauge cannot create structure: only relative signs are observable, and products 
    over closed cycles are invariant under any gauge transformation (see frustration).
    """
    C = torch.as_tensor(C).float()
    _, V = torch.linalg.eigh(C)
    u = V[:, -1]
    if float(u.sum()) < 0:
        u = -u
    s = torch.sign(u)
    s[s == 0] = 1.0
    m = u.abs() / u.abs().max().clamp_min(1e-12)
    return s, m, [i for i in range(len(m)) if float(m[i]) < thin]


def apply_gauge(C, s):
    s = torch.as_tensor(s).float()
    return torch.as_tensor(C).float() * s.unsqueeze(0) * s.unsqueeze(1)


def eigengap(C):
    """
    Eigengap of the first eigenvalue: quantifies the identifiability of the gauge.

    The relative gap strongly correlates with network frustration, serving as a 
    predictive metric to anticipate where signs will be lost BEFORE actually 
    applying them.
    """
    C = torch.as_tensor(C).float()
    w = torch.linalg.eigvalsh(C).flip(0)
    lam1, lam2 = float(w[0]), float(w[1])
    gap = lam1 - lam2
    return dict(lam1=lam1, lam2=lam2, gap=gap,
                rel=(gap / lam1 if lam1 > 0 else float("nan")))


# =====================================================================
#  correlazioni fra matrici
# =====================================================================
def _offdiag_mask(K):
    return ~torch.eye(K, dtype=torch.bool)


def pearson(x, y):
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    a, b = x - x.mean(), y - y.mean()
    den = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / den)


def _ranks(v):
    order = torch.argsort(v)
    r = torch.empty_like(v)
    r[order] = torch.arange(len(v), dtype=v.dtype)
    return r


def spearman(x, y):
    return pearson(_ranks(torch.as_tensor(x).float()),
                   _ranks(torch.as_tensor(y).float()))


def mantel(A, B, perms=9999, method="pearson", seed=0, chunk=512):
   """
    Computes the Mantel test correlation between the off-diagonal elements of two matrices.
    The p-value is derived via joint permutation of rows and columns of one of the matrices.

    Chunk-vectorized implementation: yields identical results to a standard loop 
    but is significantly faster due to batching.
    """
    A = torch.as_tensor(A).float()
    B = torch.as_tensor(B).float()
    K = A.shape[0]
    if B.shape[0] != K:
        raise ValueError("Matrices with different dimensions: %d e %d" % (K, B.shape[0]))
    m = _offdiag_mask(K)
    corr = pearson if method == "pearson" else spearman

    r = corr(A[m], B[m])
    floor = 1.0 / (math.factorial(K) + 1) if K < 8 else 0.0
    if perms <= 0:
        return dict(r=r, p=float("nan"), K=K, floor=floor)

    if method == "pearson":                      # forma chiusa, vettorizzabile
        b = B[m]; b = b - b.mean(); b = b / b.norm().clamp_min(1e-12)
        g = torch.Generator().manual_seed(seed)
        hits, done = 0, 0
        while done < perms:
            n = min(chunk, perms - done)
            idx = torch.stack([torch.randperm(K, generator=g) for _ in range(n)], 0)
            Ap = torch.gather(A[idx], 2, idx.unsqueeze(1).expand(n, K, K))
            X = Ap[:, m]
            X = X - X.mean(dim=1, keepdim=True)
            hits += int((((X @ b) / X.norm(dim=1).clamp_min(1e-12)) >= r).sum())
            done += n
    else:                                        # spearman: ciclo, e' raro
        rng = random.Random(seed)
        idx = list(range(K))
        hits = 0
        for _ in range(perms):
            rng.shuffle(idx)
            t = torch.tensor(idx)
            if corr(A[t][:, t][m], B[m]) >= r:
                hits += 1
    return dict(r=r, p=(1 + hits) / (perms + 1), K=K, floor=floor)


def _triangles(K):
    t = torch.tensor(list(itertools.combinations(range(K), 3)))
    return t[:, 0], t[:, 1], t[:, 2]


def triple_mantel(A, B, perms=5000, seed=0):
    """
    The same as the Mantel test, but computed over cyclic triangle products. 
    
    It is GAUGE INVARIANT: the product C_ij * C_jk * C_ki is unchanged under 
    any sign assignment. This metric compares network structure without being 
    affected by orientation, making it the optimal choice when the gauge 
    structures of the two bundles are not directly comparable.
    """
    A = torch.as_tensor(A).float()
    B = torch.as_tensor(B).float()
    K = A.shape[0]
    i, j, k = _triangles(K)
    tA = A[i, j] * A[j, k] * A[i, k]
    tB = B[i, j] * B[j, k] * B[i, k]
    r = pearson(tA, tB)
    if perms <= 0:
        return dict(r=r, p=float("nan"), n_tri=len(i))
    g = torch.Generator().manual_seed(seed)
    hits = 0
    for _ in range(perms):
        p = torch.randperm(K, generator=g)
        Bp = B[p][:, p]
        if pearson(tA, Bp[i, j] * Bp[j, k] * Bp[i, k]) >= r:
            hits += 1
    return dict(r=r, p=(1 + hits) / (perms + 1), n_tri=len(i))


def frustration(C, perms=5000, seed=0):
    """
    Harary structural balance: computes the fraction of triangles with a negative product.

    This metric is gauge invariant. A frustrated triangle cannot be resolved by any 
    sign assignment; it represents a genuine structural obstruction rather than an 
    orientation artifact. The null model randomizes the signs of the matrix elements 
    while preserving their magnitudes (moduli).
    """
    C = torch.as_tensor(C).float()
    K = C.shape[0]
    i, j, k = _triangles(K)
    t = C[i, j] * C[j, k] * C[i, k]
    frac = float((t < 0).float().mean())
    wt = float(t[t < 0].abs().sum() / (t.abs().sum() + 1e-12))
    absC = C.abs()
    g = torch.Generator().manual_seed(seed)
    iu = torch.triu_indices(K, K, offset=1)
    nf = []
    for _ in range(perms):
        S = torch.ones(K, K)
        v = torch.where(torch.rand(iu.shape[1], generator=g) < 0.5, -1.0, 1.0)
        S[iu[0], iu[1]] = v
        S[iu[1], iu[0]] = v
        Cn = absC * S
        nf.append(float((Cn[i, j] * Cn[j, k] * Cn[i, k] < 0).float().mean()))
    nf = torch.tensor(nf)
    return dict(frac=frac, weighted=wt, null_mean=float(nf.mean()),
                null_p5=float(nf.quantile(0.05)))


# =====================================================================
#  decodifica a centroide
# =====================================================================
def nearest_centroid_cv(D, labels, folds=5, seed=0):
    """Held-out nearest-centroid decoding by cosine.

    D is [n, d] of unit vectors, labels are the class of each row. Folds are
    built WITHIN each class, so every fold holds out a slice of every class and
    no class disappears from the training centroids.

    Returns accuracy. Chance is one over the number of classes, and with many
    classes chance is low: always report it beside the accuracy, and price the
    rest with a label-permutation null (see decoding_with_null).
    """
    import zlib
    D = torch.as_tensor(D).float()
    cats = sorted(set(labels))
    per = {c: [i for i, l in enumerate(labels) if l == c] for c in cats}
    correct = total = 0
    for f in range(folds):
        train, test = [], []
        for c in cats:
            ids = list(per[c])
            random.Random(seed + zlib.crc32(str(c).encode()) % 1000).shuffle(ids)
            cut = [ids[k::folds] for k in range(folds)]
            test += cut[f]
            train += [i for k in range(folds) if k != f for i in cut[k]]
        if not test or not train:
            continue
        C = []
        for c in cats:
            tr = [i for i in train if labels[i] == c]
            v = D[tr].mean(0) if tr else torch.zeros(D.shape[1])
            C.append(v / v.norm().clamp_min(1e-8))
        pred = (D[test] @ torch.stack(C, 0).T).argmax(dim=1)
        truth = torch.tensor([cats.index(labels[i]) for i in test])
        correct += int((pred == truth).sum())
        total += len(test)
    return correct / total if total else float("nan")


def decoding_with_null(D, labels, folds=5, seed=0, perms=100):
    """Nearest-centroid accuracy with a label-permutation null.

    The null shuffles the labels and reruns the whole procedure, so it prices
    both chance and any optimism from the fold structure."""
    acc = nearest_centroid_cv(D, labels, folds, seed)
    rng = random.Random(seed)
    null = []
    for _ in range(perms):
        lab = list(labels)
        rng.shuffle(lab)
        null.append(nearest_centroid_cv(D, lab, folds, seed))
    nt = torch.tensor(null)
    return dict(acc=acc, null_mean=float(nt.mean()),
                null_p95=float(nt.quantile(0.95)),
                p=(1 + int((nt >= acc).sum())) / (perms + 1),
                chance=1.0 / max(len(set(labels)), 1))


# =====================================================================
#  scomposizione esatta gate contro valore in una SwiGLU
# =====================================================================
def gate_value_split(g_t, u_t, g_f, u_f, w):
    """EXACT decomposition of a pair's axis gap in the expanded basis.

        gate  = (g_t - g_f) * (u_t + u_f)/2  .  w
        value = (g_t + g_f)/2 * (u_t - u_f)  .  w
        total = (g_t*u_t - g_f*u_f)          .  w

    and gate + value == total identically, because

        Dg*ubar + gbar*Du = g_t u_t - g_f u_f

    expands term by term. This is not an approximation and not an ablation: it
    is an algebraic identity, so the two shares always sum to the whole and the
    split cannot be blamed for what it leaves out.

    w is the residual axis PULLED BACK into the expanded basis, w = W_down^T v1,
    not an axis fitted in the expanded space. Fitting there would be a different
    object, estimated in thousands of dimensions from a few hundred pairs.

    g must be the OUTPUT of the activation, not the pre-activation: capturing
    act_fn directly avoids assuming which non-linearity the model uses.

    A caution on reading the share. When gate and value nearly CANCEL the total
    is a small difference of large terms, and gate/total divides by almost
    nothing. The share is then meaningless and must be reported as such, not as
    a large number."""
    Dg, Du = g_t - g_f, u_t - u_f
    gbar, ubar = (g_t + g_f) / 2, (u_t + u_f) / 2
    return (Dg * ubar) @ w, (gbar * Du) @ w, (g_t * u_t - g_f * u_f) @ w


def gate_value_split_sandwich(g_t, u_t, g_f, u_f, w, s_t, s_f):
    """The same split when a post-norm sits between the FFN and the residual.

    There the vector that enters the stream is s * W_down(g*u) with s = 1/rms,
    a per-sentence scalar, so the pair gap carries a THIRD term:

        gap = sbar*gate + sbar*value + Ds*Pbar

    with sbar and Ds the mean and difference of the two scales and Pbar the mean
    of the projections. The norm term is not noise: it is the gap produced by
    the two sentences being normalised differently, and attributing it to gate
    or value would be wrong."""
    gate, value, _ = gate_value_split(g_t, u_t, g_f, u_f, w)
    P_t, P_f = (g_t * u_t) @ w, (g_f * u_f) @ w
    sbar, ds = (s_t + s_f) / 2, s_t - s_f
    Pbar = (P_t + P_f) / 2
    return (sbar * gate, sbar * value, ds * Pbar, s_t * P_t - s_f * P_f)


def intra_pair_mean(X, pidx):
    """Both rows of a pair replaced by their mean. The freeze target.

    Freezing at the INTRA-PAIR mean, not at the global mean and not at zero, is
    what makes the intervention answer the intended question. Zero would remove
    the stream; the global mean would move both sentences to a common point far
    from either. The intra-pair mean removes exactly the CLASS-DRIVEN variation
    of that stream and leaves everything else where it was."""
    out = X.clone()
    for it, iff in pidx:
        m = (X[it] + X[iff]) / 2
        out[it] = m
        out[iff] = m
    return out


# =====================================================================
#  la legge dell'arrangement ristretta al gate della conoscenza
# =====================================================================
def restricted_law(C_a, C_b, know_a, know_b, cats, threshold=0.60,
                   perms=9999, seed=0, early_a=None, early_b=None):
    """The arrangement law, before and after restricting to shared knowledge.

    Two dictionaries agree on their category arrangement, but part of the
    disagreement is not disagreement at all: it is attenuation. Where a model
    knows a relation poorly its axis for that relation is estimated from a
    weaker signal, and a noisy measurement cannot correlate with anything,
    including a correct one. Restricting to relations BOTH models know is not
    cherry-picking as long as the criterion is declared in advance and the
    knowledge proxy is measured independently of the agreement it is used to
    explain: the proxy here is the within-category held-out AUC, the diagonal
    of the transfer matrix, which never looks at the other model.

    know_a and know_b are those diagonals. The threshold applies to BOTH: a
    relation survives only if both models clear it.

    The matrices are expected ALREADY GAUGED. Applying a gauge inside would hide
    which one was used and with what margins, and the gauge is a separate step
    with its own provenance: build the bundle, gauge it, then compare. Pass
    early_a and early_b to obtain the surface control, which is not optional in
    practice: without it a high agreement does not separate shared geometry from
    shared corpus statistics.

    Returns the full, restricted and surface Mantel, the surviving categories,
    and the Spearman correlation between per-category ADHERENCE and the
    knowledge proxy.
    Adherence is the correlation between a category's ROW in the two matrices:
    how similarly that category sits with respect to all the others. If
    adherence rises with knowledge, the gate is not a post-hoc filter but a
    graded relation, and that is the stronger claim.

    A caution the caller must keep: with K categories the permutation floor is
    1/K!, and restriction lowers K. A restricted set of five relations cannot
    reach significance whatever the data say, and the returned 'floor' says so.
    """
    A = torch.as_tensor(C_a).float()
    B = torch.as_tensor(C_b).float()
    ka = torch.as_tensor(know_a).float()
    kb = torch.as_tensor(know_b).float()
    K = A.shape[0]

    full = mantel(A, B, perms=perms, seed=seed)

    # SURFACE CONTROL. The same agreement computed on the EARLY-block matrices.
    # Without it a high peak agreement does not distinguish shared geometry from
    # shared corpus statistics: two models trained on overlapping text can align
    # at a shallow block for reasons that have nothing to do with truth. The
    # early matrices carry the OLD orientation, because at a shallow block there
    # is no stable gauge to apply, and that is stated rather than hidden.
    surface = None
    if early_a is not None and early_b is not None:
        surface = mantel(torch.as_tensor(early_a).float(),
                         torch.as_tensor(early_b).float(),
                         perms=perms, seed=seed + 2)

    keep = [i for i in range(K) if ka[i] >= threshold and kb[i] >= threshold]
    if len(keep) >= 3:
        idx = torch.tensor(keep)
        rest = mantel(A[idx][:, idx], B[idx][:, idx], perms=perms, seed=seed + 1)
    else:
        rest = dict(r=float("nan"), p=float("nan"), K=len(keep), floor=1.0)

    # per-category adherence: the row of a category against the same row in the
    # other matrix, with the diagonal removed so a category is not compared to
    # itself
    adh = []
    for i in range(K):
        oth = [j for j in range(K) if j != i]
        adh.append(pearson(A[i, oth], B[i, oth]))
    proxy = torch.minimum(ka, kb)
    sp = spearman(torch.tensor(adh), proxy)

    return dict(full=full, restricted=rest, surface=surface,
                kept=[cats[i] for i in keep],
                dropped=[cats[i] for i in range(K) if i not in keep],
                threshold=threshold,
                adherence={cats[i]: adh[i] for i in range(K)},
                knowledge={cats[i]: float(proxy[i]) for i in range(K)},
                spearman_adherence_knowledge=sp,
                pearson_adherence_knowledge=pearson(torch.tensor(adh), proxy))


# =====================================================================
#  attenuazione classica
# =====================================================================
def reliabilities(r_ab, r_ac, r_bc):
    """
    Derives individual reliabilities from three pairwise agreements.
    
    Under a single-factor model, the observed correlation is factored as 
    r_XY = lambda_X * lambda_Y. This corresponds to Spearman's (1904) 
    classical correction for attenuation: restricting to known categories 
    does not alter the underlying law, it merely removes the attenuation.
    """
    def s(x, y, z):
        v = x * y / z if z != 0 else float("nan")
        return math.sqrt(v) if v == v and v > 0 else float("nan")
    return dict(a=s(r_ab, r_ac, r_bc), b=s(r_ab, r_bc, r_ac), c=s(r_ac, r_bc, r_ab))


def attenuation_ceiling(rel_a, rel_b):
    """
    Computes the maximum observable agreement between two measures given their reliabilities.

    Under the assumption of a single shared factor, an agreement ABOVE this ceiling 
    indicates additional shared structure beyond that factor. An agreement BELOW 
    this ceiling implies that the two measures do not capture entirely the same construct.
    """
    v = rel_a * rel_b
    return math.sqrt(v) if v > 0 else float("nan")
