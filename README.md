# truthprobe

A measurement apparatus for the geometry of truth in language models.

Companion library to *The Anatomy of a Truth Direction* (arXiv:2607.16741).
The canonical scripts that produced Part I live in the paper's own repository.
This one holds the shared machinery, the campaign that produced Parts II and III,
and the dictionary bundles those numbers were read from.

The canonical scripts predate the sentence convention used here: they build "prompt target" without the final period, and have no parameter to change it.
Bundles from the two repositories are therefore not comparable.
The bridge is measured: the global axes differ at cosine 0.474, while the arrangement between categories survives at Mantel +0.775.

---

## What this measures, in one page

A model, given a factual statement, produces a hidden state. Take a **minimal
pair**: two sentences identical except for one content word, one true and one
false. The difference of their hidden states cancels the shared topic and
leaves the truth signal. The dominant direction of those differences, found by
SVD without any training, is the **truth axis**.

Three commitments hold everything else up.

1. **The axis is identified without labels, up to one global sign.** The Gram
   matrix of the difference rows does not depend on which element of a pair is
   called true, so neither do its eigenvectors.
2. **Every claim is gated.** The additive decomposition of a block is verified
   before any number derived from it is read. A gate that fails stops the run
   instead of warning.
3. **Conventions travel with the data.** How a sentence is built, which dataset
   revision, which seed: all of it lives in a `Protocol` object, written into
   every artifact and checked before any comparison.

The third exists because of a real failure. Two tools in this project built
sentences differently for months without anyone noticing, producing axes at
cosine +0.52 while the arrangement between categories survived at Mantel +0.775.
The convention is now impossible to leave implicit.

### Scope, with a number for each condition

The probe reads **factual truth**, inside a **controlled contrast**, on facts
the model **represents**. All three are required:

| condition | what happens without it |
|---|---|
| controlled contrast | on free-text pairs a surface-feature classifier with no model reaches AUC 0.814, above the best layer of any model tested |
| known fact | separability falls with the behavioural know-rate; the axis-probe gap widens |
| label matches representation | on adversarial labels the paired accuracy sits at 34-43%, below chance |

---

## Install

```bash
pip install -e .
python scripts/campagna.py --model <hf-name-or-path> --stages behav,signal
```

`torch` is the only hard dependency. `transformers` and `datasets` are the
`[models]` extra, needed only when a model is loaded: the pure functions install
and test without them.

---

## Layout

```
truthprobe/     the library: functions, no scripts
  protocol.py     conventions, and the ONE place sentences are built
  data.py         pair loading; PairSet
  geometry.py     fit_axis, project_fields, cosine matrices, per-layer maps
  stats.py        AUC, folds, gauges, Mantel, frustration, attenuation, splits
  subspace.py     principal angles, spectral entropy, CKA
  hooks.py        architecture detection, additive decomposition, gates
  bundle.py       save and load with provenance
tools/          exploratory scripts built on the library
scripts/        the campaign that produced Parts II and III of the paper
data/           the dictionary bundles, with their protocol metadata
```

The line between the two folders: **if it decides what to measure it is a tool,
if it only computes it is the library.** Nothing in `truthprobe/` has a `main`.

---

## Provenance of each function

Not everything here has the same standing, and the difference matters when
deciding how much to trust a number.

**Verified against published canonical code.** `fit_axis` and `project_fields`
with their three quiet choices (orientation on the means, robust calibration,
explicit orthogonalisation of v2); `auc_score`, `kfold_pairs`;
`axis_cosine_matrix`, `transfer_matrix`; the per-head decomposition and the
sandwich-norm handling; `frame_curves`; `gate_value_split` and its sandwich
three-term form; `nearest_centroid_cv` and `decoding_with_null`;
`best_sign_flips`; `global_axis_gauge`. Each is checked in the regression suite
to reproduce the canonical result, in most cases to machine precision.

**From unpublished tooling of the same project.** `mantel`, `triple_mantel`,
`frustration`, `eigengap`, `reliabilities`, `consensus_gauge`. These come from
working code that is not in the paper's repository, so a reader cannot check
them against a published source. They are validated here on planted ground
truth.

**Written for this library, with no canonical counterpart.** The `subspace`
module (principal angles, spectral entropy, effective rank, CKA);
`feature_alignment`; `restricted_law`; `arrangement_by_layer`; and the whole of
`protocol.py`, `data.py`, `bundle.py`. Standard mathematics or infrastructure,
validated on planted ground truth, but reproducing nothing previously published.

---

## The core, with formulas

### The axis

$$D = U S V^{\top}, \qquad v_1 = V_1$$

with rows $d_i = s_i (h_i^{A} - h_i^{B})$. Since $d_i d_i^{\top} =
(-d_i)(-d_i)^{\top}$, the Gram matrix $D^{\top}D$ is invariant to the
within-pair labels, and so is the span of $v_1$.

```python
from truthprobe.geometry import fit_axis, project_fields
ax = fit_axis(H, pidx)                 # H [N, d], pidx [(i_true, i_false), ...]
Re = project_fields(H, ax)["Re"]       # the position coordinate
```

### The additive decomposition, and its gate

$$h_{b+1} = h_b + a_b + f_b$$

```python
from truthprobe.hooks import describe, BlockCapture, identity_gate
arch = describe(model)
with torch.no_grad(), BlockCapture(model, arch, block) as cap:
    out = model(**enc, output_hidden_states=True)
    a, f, per_head = cap.attn(), cap.ffn(), cap.heads()
med, ok, msg = identity_gate(h_before, h_after, a, f)
```

The verdict is the **median** over all (sentence, block) pairs, never the
maximum: where a block adds almost nothing the delta is near zero and the
relative error explodes even when the reconstruction is exact.

### Per head, and inside the feed-forward

$$o(z) = \sum_h W_o[:, \mathrm{slice}_h]\, z_h$$

$$\text{gate} = \Delta g \odot \bar u \cdot w, \quad
\text{value} = \bar g \odot \Delta u \cdot w, \quad
\text{gate} + \text{value} \equiv \text{total}$$

with $w = W_{\text{down}}^{\top} v_1$, the residual axis pulled back into the
expanded basis. The split is an **algebraic identity**, not an ablation: the two
shares always sum to the whole. On sandwich-norm models a third term appears,
from the two sentences being normalised differently.

```python
from truthprobe.stats import gate_value_split, intra_pair_mean
sw = cap.swiglu(axis=v1, pidx=pidx)    # gate_term, value_term, norm_term, total
```

### A contribution against a fixed frame

$$\mathrm{gap} = \overline{(v_1 \cdot c)}\big|_{\text{true}} -
\overline{(v_1 \cdot c)}\big|_{\text{false}}, \qquad
d' = \frac{\mathrm{gap}}{\text{pooled sd}}$$

```python
from truthprobe.stats import frame_gap, frame_curves
r = frame_gap(c_true, c_false, axis, contrib_block=16, frame_block=15)
```

The axis is fixed and passed in, never refitted inside. The two block indices
are not decoration: if the frame **contains** the measured contribution, the
contribution is partly correlated with itself and the sign is not the law's.
`frame_gap` says so instead of returning a plausible number in silence.

### The sign gauge

$$M = D_s C D_s, \qquad s_c = \mathrm{sign}(v_c \cdot t_{\text{global}})$$

```python
from truthprobe.stats import global_axis_gauge, consensus_gauge, eigengap
s, margins, unsigned = global_axis_gauge(axes, t_global)   # primary
s, margins, unsigned = consensus_gauge(C)                  # spectral alternative
```

A gauge is a **choice of meridian, not an imposition of structure**, and that is
checkable: cycle products $C_{ij}C_{jk}C_{ki}$ are invariant under every sign
assignment, so no gauge can manufacture agreement. Categories below the margin
threshold are reported **unsigned** rather than forced: there the sign is a coin.

A bundle carries the axes, not the gauge. Signing is a separate step with its
own provenance.

### Arrangement agreement, and the knowledge gate

$$r = \mathrm{Pearson}\big(\mathcal{O}(M^{A}), \mathcal{O}(M^{B})\big),
\qquad \hat p = \frac{1 + \#\{b : r_b \ge r\}}{B + 1}$$

```python
from truthprobe.stats import mantel, triple_mantel, frustration, restricted_law
mantel(A, B, perms=9999)          # signed cells, needs a common gauge
triple_mantel(A, B)               # cycle products, gauge-FREE
restricted_law(A, B, know_a, know_b, cats, threshold=0.60,
               early_a=Ea, early_b=Eb)      # with the surface control
```

With $K$ categories the permutation floor is $1/K!$: below $K = 6$ significance
is unreachable whatever the data say, and restriction lowers $K$.

### Attenuation, and subspaces

$$r_{AB} = \lambda_A \lambda_B, \qquad
\cos\theta_k = \sigma_k(Q_A^{\top} Q_B), \qquad
\mathrm{rank}_{\mathrm{eff}} = \exp\!\big({-}\textstyle\sum_i p_i \log p_i\big)$$

```python
from truthprobe.stats import reliabilities, attenuation_ceiling
from truthprobe.subspace import principal_angles, effective_rank
```

Principal angles return a **spectrum**, not a scalar: two subspaces can share
three dimensions out of eight and be orthogonal in the other five, and one
number hides that. Small angles are computed from the sine, not from `arccos`,
which loses precision near zero.

---

## The tools

| tool | question |
|---|---|
| `crea_dizionario.py` | build a dictionary bundle, on any stream |
| `teste_dizionario.py` | one dictionary per attention head, in a single pass |
| `categories.py` | is the axis a mixture of category components? |
| `analisi_teste.py` | what each head writes: coverage, geometry, arrangement, subspaces |
| `ablazione_teste.py` | which heads or experts are causally necessary |
| `mappa_dominio.py` | does the dictionary recognise this contrast, and where is the signal? |
| `behav_contrasto.py` | does the model itself prefer the labelled-true target? |
| `baseline_superficie.py` | how much separates from the text alone, no model? |
| `strati_contrasto.py` | what kind of contrast does each item actually pose? |
| `confronta_bundle.py` | are two bundles the same measurement? |
| `inventario.py` | catalogue, deduplicate and file bundles by CONTENT |
| `collaudo.py` | reproduce the published numbers by an independent path |

Two are worth running **before** any geometry: `baseline_superficie.py`, because
if surface alone separates the material the geometry adds nothing; and
`behav_contrasto.py`, because a probe cannot be faulted for failing to read a
belief the model does not hold.

---

## Four traps, all found the hard way

Every one of these produced a plausible, wrong number that nobody noticed at
first. The library now catches three of them; the fourth it cannot.

**1. The sentence convention.** Building `"prompt target"` and
`"prompt target."` are different measurements, not formatting variants. With the
period the last token is identical in both sentences of a pair, so token
identity leaves the measurement; without it, the last token is the target word
and its identity enters. Measured on Gemma-2-2b: axes at cosine +0.52,
arrangement surviving at Mantel +0.775. *Caught by:* `Protocol`, written into
every artifact and checked before comparison.

**2. The frame that contains what it measures.** Reading a block's contribution
against a frame fitted at that same block makes the contribution partly
correlated with itself. Under a post frame the FFN reads +0.59 at its own block;
under a pre frame it reads −0.44. The sign inverts, and the relational law is
visible only in the second. *Caught by:* the block indices in `frame_gap`.

**3. The stream the axes were fitted on.** Reading the residual on axes fitted
on the attention contribution gives a flat 0.038 everywhere: the two live in the
same space and not in the same informative subspace. *Caught by:* the tools read
the stream from the bundle's metadata rather than from a flag.

**4. The composition of the material.** Pairs drawn at random from the whole
dataset and pairs grouped by relation are different populations, and the AUC
scale moves with them: on planted data, the same signal reads 0.732 grouped and
0.696 flat. The library offers both loaders and **chooses neither**. *Not caught
by anything:* this one is the caller's, and a script that leaves the choice
implicit produces numbers that cannot be compared with anything.

The common shape is worth naming: in all four the mathematics was right and the
ingredients were wrong. A library can make conventions explicit and gate what it
can verify; it cannot decide which question you meant to ask.

---

## An architecture the library has never seen

Where a module lives is the only architecture-dependent fact, and it can be
supplied from outside:

```python
from truthprobe.hooks import Wiring, describe, BlockCapture
w = Wiring(ffn_write = lambda L: L.block_sparse_moe,
           router    = lambda L: L.block_sparse_moe.gate,
           experts   = lambda L: L.block_sparse_moe.experts)
arch = describe(model, wiring=w)
with torch.no_grad(), BlockCapture(model, arch, block) as cap:
    model(**enc)
    r = cap.routing()      # {"logits", "experts", "weights"} at the read token
```

The gate then verifies the wiring instead of trusting it. A mixture of experts
keeps the block additive, so the block-level decomposition holds unchanged; what
changes is inside the feed-forward, and attribution per expert is a new object.

One design note before running such an experiment: **the router routes per
token, not per sentence.** In a minimal pair the two sentences differ in one
token, and with the period convention the read token is identical in both. The
question is therefore whether the upstream target change alters which experts
fire at the period. If it does not, a per-expert decomposition has nothing to
separate, and that is knowable before fitting anything.

---

## What is known, and where it stops

Established across families: attention propagates truth frames it did not write
while the feed-forward opposes the frame of its moment; the post-peak erosion is
carried by the value stream on five architectures; per-category axes form a
signed arrangement is largely shared across families, with a measurable family-specific component

Measured boundaries: the arrangement is identifiable only where the eigengap is
large; outside the peak the knowledge gate stops operating; on material without
a controlled contrast no component and no layer recovers a signal.

Open: the near-orthogonality between the attention-fitted and the
residual-fitted global axis, cosine −0.017 on the same pairs at the same block
with both separating above 0.87.

---

Code is MIT. The dictionary bundles in `data/` are CC BY 4.0, like the paper.
The tools print matrices, margins and distances, and never a
verdict: the reading belongs to the researcher.
