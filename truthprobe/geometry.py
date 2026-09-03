# -*- coding: utf-8 -*-
"""
truthprobe.geometry

The axis and its projections. The functions here are ported from truth_probe.py 
preserving the exact semantics, including design choices that may seem like minor 
details but are not. The regression test verifies that the numbers match 
the canonical codebase exactly.


THREE DESIGN CHOICES THAT MUST BE PRESERVED

1. ORIENTATION RELIES ON THE MEANS OF THE PROJECTIONS, not on the sum of 
   the differences. They almost always agree, but not always: under skewed 
   distributions they can diverge, which would flip the sign bit.

2. CALIBRATION IS ROBUST. 'med' and 'scale' are derived from the median and 
   the median absolute deviation (MAD) scaled by 1.4826, rather than the mean 
   and standard deviation. This is necessary because a few outlier states with 
   high norms would otherwise skew the calibration distortingly.

3. v2 IS EXPLICITLY ORTHOGONALIZED AGAINST v1. SVD would already yield an 
   orthogonal output; however, following a potential sign flip of v1, the 
   component removal is performed explicitly anyway to guarantee and preserve orthogonality.


The label-free identification Lemma holds on `fit_axis`: the Gram matrix of the 
difference rows does not depend on which element of the pair is designated as true, 
hence the span of v1 is invariant. The 'orient' argument is used exclusively 
for the permutation null baseline and the final orientation.
"""

import torch


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def ang_diff(a, b):
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def robust_ms(c):
    """median and robust scale (MAD per 1.4826), like in truth_probe."""
    med = c.median()
    scale = (1.4826 * (c - med).abs().median()).clamp_min(1e-8)
    return float(med), float(scale)


def fit_axis(Hl, pidx, orient=None):
    """Truth axis, from the SVD of the intra-pair differences.

    Hl    [N, d] stati a un livello
    pidx  list of (true_index, false_index)
    orient  optional, +1 o -1 per pair: swap the two side. it used for permutation
            null. By the lemma, the span of v1 does not depend on it.

    Returns the axis dictionary containing v1, v2, and the calibration parameters, 
    structured exactly as expected by project_fields.
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
    """Calibrated coordinates of a state relative to an axis.

    Re     position on the axis, the unique truth-bearing coordinate according 
           to the seven falsification experiments
    Im     the second direction, residual variance on fixed-polarity pairs
    b      Re passed through a sigmoid function
    m      magnitude, i.e., energy
    theta  argument, i.e., phase
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
        """Unit vector v1, for those who only need the direction."""
    return unit(ax["v1"])


def cosine_matrix(axes):
    """KxK SIGNED cosine matrix between oriented axes, each within its own category. 
    The sign is information: negative means that the shared direction reads the 
    truth of the other category in reverse. 
    However, it depends on the orientation of BOTH categories involved: only the 
    products over closed loops are invariant (see `stats.frustration`).
    """
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
    """Fraction of the squared norm of each row in D that falls within the span 
    of the rows of A. The expected random baseline value is approximately rows(A) / columns(A): 
    this baseline must always be reported alongside the result, otherwise the number cannot be interpreted.
    """
    Q, _ = torch.linalg.qr(A.T)
    n2 = D.pow(2).sum(1).clamp_min(1e-12)
    return (D @ Q).pow(2).sum(1) / n2
