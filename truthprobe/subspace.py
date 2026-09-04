# -*- coding: utf-8 -*-
"""

truthprobe.subspace

Subspace comparisons and spectral metrics.

The rest of this library focuses on comparing individual directions (via cosines) 
or matrices of directions (via the Mantel test). This module shifts the focus 
entirely: it compares two full subspaces, answering the fundamental question: 
how many dimensions do they actually share?

This question was already posed in our series before we had the direct tool 
to answer it. The claim that 'the attention arrangement covers approximately 
half the effective dimensions of the FFN arrangement after reliability correction' 
is a statement about subspaces, initially estimated indirectly. The principal 
angles implemented here provide the direct measurement.

FOUR TOOLS, AND WHAT THEY ANSWER:

    principal_angles: Given two sets of directions, it determines how many 
                      angles are small. It returns a full spectrum rather than 
                      a single scalar: two subspaces can share three dimensions 
                      out of eight while being perfectly orthogonal in the 
                      remaining five, a nuance that a single scalar would hide.

    effective_rank:   Quantifies the continuous dimensionality effectively spanned 
                      by a set of directions. It is the exponential of the spectral 
                      entropy, measuring not just the raw count of directions, 
                      but how many are linearly independent.

    spectral_entropy: The von Neumann entropy of the normalized Gram matrix 
                      (trace of one). It is gauge invariant, depending exclusively 
                      on the spectrum. It can be viewed as a generalized eigengap 
                      across the entire spectrum: while the standard eigengap 
                      looks only at the first spectral jump, this metric analyzes 
                      the full distribution.

    cka:              Measures the global similarity between two sets of REPRESENTATIONS. 
                      It is invariant to rotation and isotropic scaling, designed to 
                      compare full states rather than individual axes. Along the axes, 
                      the signed cosine and Mantel test are more informative because 
                      the sign itself carries meaning, whereas CKA discards it.

A WARNING APPLICABLE TO ALL METRICS:
In d dimensions and with subspaces of dimension k, two RANDOM subspaces will 
inherently exhibit small angles when k is non-negligible relative to d. 
The expected random baseline must always be reported alongside the empirical result. 
For this reason, every relevant function in this module returns the raw numerical 
baseline directly instead of leaving the computation to the reader.
"""


import math

import torch


def _orthonormal(A):
    """
    Computes an orthonormal basis for the row space of A [k, d].
    
    Returns [d, r], where r is the effective rank. If the rows are linearly 
    dependent, r will be strictly less than k. This numerical rank reduction 
    is itself a valuable data point.
    """
    A = torch.as_tensor(A).float()
    Q, R = torch.linalg.qr(A.T)
    tol = 1e-6 * max(A.shape) * float(R.diagonal().abs().max().clamp_min(1e-30))
    keep = R.diagonal().abs() > tol
    return Q[:, keep]


def _angles(QA, QB):
   """
    Computes the principal angles between two ORTHONORMAL bases, 
    sorted in ascending order (in radians).

    Using the cosine alone is insufficient. Computing the angle solely as the 
    arccos of the singular values becomes numerically unstable for small angles. 
    When the angle is tiny, the cosine approaches 1; a minor loss of precision 
    of 1e-7 in the cosine translates to an error of 0.03 degrees in the angle. 
    For a shared subspace, where the true angle should be exactly zero, this 
    residual error would be misinterpreted as structural divergence.

    The remedy follows the classical Björck & Golub (1973) algorithm: for angles 
    below 45 degrees, the cosine is substituted with the sine. The sine is derived 
    from the norm of the projection of QB orthogonal to the span of QA. 
    The sine function yields high precision where the cosine fails, and vice versa.
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
    """
    Computes the principal angles between the row space of A and B.

    Returns the angles sorted in ascending order, their corresponding cosines, 
    and the expected baseline values for two random subspaces of the identical dimensions.

    The first principal angle is exactly zero if and only if the two subspaces share 
    an exact direction. The number of angles falling below a given threshold quantifies 
    the shared dimensionality between the spaces. If all angles are large, the two 
    subspaces are mutually orthogonal. In high-dimensional settings (large d, small k), 
    orthogonality is the generic case; hence, the random baseline is provided for 
    proper statistical calibration.
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
     """
    Fraction of the energy from the span of A that falls within the span of B.

    It is computed as the mean of the squared cosines of the principal angles, 
    generalizing the 'inside' fraction used elsewhere for a single vector to a 
    full subspace. The expected baseline value at random is approximately dim(B)/d.
    """
    r = principal_angles(A, B, degrees=False)
    return dict(overlap=float((r["cosines"] ** 2).mean()),
                chance=r["dim_b"] / r["d"],
                dim_a=r["dim_a"], dim_b=r["dim_b"])


def spectral_entropy(M, base="e", from_gram=True):
     """
    Von Neumann entropy of the spectrum.

    M can either be a precomputed Gram matrix (from_gram=True) or a set of row 
    vectors, in which case the Gram matrix will be constructed.

    The matrix is normalized to have a trace of one, ensuring the eigenvalues 
    act as a probability distribution and the resulting value is a true von Neumann 
    entropy. Negative eigenvalues arising from numerical precision errors are zeroed 
    out rather than silently ignored; if large negative eigenvalues are present, 
    the matrix is not positive semi-definite and the metric becomes meaningless, 
    which this function explicitly tracks and declares.

    Gauge invariant: depends exclusively on the spectrum. Changing category signs 
    amounts to a similarity transformation, leaving the eigenvalues intact.
    """
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
                warning=("Spectrum contains non-negligible negative eigenvalues: the "
                         "matrix is not positive semi-definite and the entropy is not "
                         "interpretable.") if neg > 1e-4 * tot else "")


def effective_rank(A, from_gram=False):
    """
    The continuous dimensionality effectively spanned by a set of directions.

    K perfectly orthogonal axes yield an effective rank of K; K perfectly parallel 
    axes yield 1. This is the precise metric required to quantify when one 
    arrangement covers fewer dimensions than another.
    """
    r = spectral_entropy(A, from_gram=from_gram)
    return dict(effective_rank=r["effective_rank"], entropy=r["entropy"],
                n_dims=r["n_dims"], max_possible=r["n_dims"])


def cka(X, Y, unbiased=False):
    """
    Linear Centered Kernel Alignment between two sets of representations.

    X and Y are tensors of shape [n, d1] and [n, d2] computed over the EXACT 
    SAME n observations, in the identical order. It is invariant to rotation 
    and isotropic scaling, but not to coordinate-wise rescaling.

    This metric is used to compare STATES, not individual axes. Along the axes, 
    the signed cosine and the Mantel test are more informative because, 
    in the context of this work series, the sign itself carries information, 
    whereas CKA discards signs by design.
    """
    X = torch.as_tensor(X).float()
    Y = torch.as_tensor(Y).float()
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA requires the same number of observations: %d contro %d"
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
    """
    Quantifies how closely a set of external features aligns with category axes.

    features: [m, d] directions produced from any source, typically the rows 
              of the decoder of a sparse autoencoder.
    axes:     [K, d] the measured dictionary category axes.

    For each axis, it reports the maximum absolute cosine |cosine| across all features 
    and compares it against a null model of random directions of the same dimension. 
    The null model is necessary and not decorative: with m features in d dimensions, 
    the maximum over m random trials increases inherently with m. Therefore, a high 
    observed value is meaningless without baseline validation.

    THIS IS NOT AN AUTOENCODER IMPLEMENTATION, and that is a deliberate design choice. 
    This library provides the baseline measurement and the evaluation criterion; training 
    a competing model is the responsibility of whoever proposes it. Evaluation is a metric, 
    training is a method; keeping them strictly decoupled is precisely what 
    makes the evaluation criterion pre-declarable and unbiased.
    """
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
