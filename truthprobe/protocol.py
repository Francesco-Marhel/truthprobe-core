# -*- coding: utf-8 -*-
"""
truthprobe.protocol

The conventions that MUST be identical across every analytical tool, encapsulated 
within a single object that travels with the data and is stored inside every artifact.


WHY THESE ARE NOT FLAGS

During the development of this series, two tools from the same project independently 
constructed sentences differently for months without anyone noticing: 
one concatenated "prompt target." and the other "prompt target". These two conventions 
produce axes with a cosine similarity of +0.52—meaning completely different directions—yet 
the alignment survives a Mantel test at +0.775. No tool flagged any anomaly, 
because the convention was treated as a minor internal detail within each individual file.

A standard flag with a default value would not have prevented this divergence; 
it would have merely shifted the default configuration across two places. Here, 
the Protocol is strictly mandatory, is written into every bundle, and any comparison 
between two bundles verifies it BEFORE computing anything. A divergence thus becomes 
a highly visible error instead of a silently incorrect number.


THREE CATEGORIES OF PARAMETERS, TREATED DIFFERENTLY

  protocol     suffix, dataset, revision, seed, K, n, precision, pooling. 
               These alter the numerical values and must be strictly shared: 
               they reside here.
               
  analysis     block, component, permutations. These alter the results but 
               are specific to each measurement run: they reside in the 
               artifact's metadata.
               
  execution    batch, device, output_folder. These do not alter the numerical 
               values: they remain free flags and are not logged.

"""

from dataclasses import dataclass, asdict, field, replace
from typing import Optional

COUNTERFACT_REPO = "NeelNanda/counterfact-tracing"
COUNTERFACT_REV = "c945b082ca08d0a8f3ba227fb78404a09614c36e"
TRUTHFULQA_REV = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


@dataclass(frozen=True)
class Protocol:
    """Data construction and state readout conventions.

    suffix: what is appended after the target.
        "."   The final token is identical in both sentences of the pair, 
              hence the token's identity cancels out of the measurement. 
              What is read has reached that point through the attention mechanism. 
              This is also the condition that gives meaning to the 
              "curve at 0.500 at layer 0" check.
        ""    The final token is the target word itself, which differs between 
              the true and false sentence: the measurement therefore includes 
              the token's own identity. Legitimate, but it answers a completely 
              different question, and the two cases are NOT comparable.

    join: how prompt and target are combined.
        "space" prompt.strip() + " " + target.strip(), explicit in the codebase.
        "raw"   prompt.strip() + target, meaning the space comes directly from the dataset. 
                This is the historical convention of `categories.py` and `crea_dizionario.py`. 
                It is preserved exclusively to reproduce legacy bundles: it depends 
                on a property of the dataset rather than the codebase itself.
    """

    suffix: str = "."
    join: str = "space"
    pool: str = "last"
    dtype: str = "float32"
    dataset: str = COUNTERFACT_REPO
    revision: str = COUNTERFACT_REV
    seed: int = 0
    k_relations: Optional[int] = None
    pairs_per_relation: Optional[int] = None
    max_pairs: Optional[int] = None
    notes: str = ""

    def __post_init__(self):
        if self.join not in ("space", "raw"):
            raise ValueError("join must be 'space' or 'raw', not %r" % self.join)
        if self.pool not in ("last", "mean"):
            raise ValueError("pool must be 'last' or 'mean', not %r" % self.pool)
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("unrecognized dtype: %r" % self.dtype)

    # ---- costruzione delle frasi: l'UNICO posto in tutta la libreria ----
    def sentence(self, prompt, target):
        """The single function that concatenates prompt and target. If a change is 
    needed in the future, updating it here changes it everywhere."""
        p = str(prompt).strip()
        if self.join == "space":
            return p + " " + str(target).strip() + self.suffix
        return p + str(target) + self.suffix

    # ---- identita' e confronto ----
    def key(self):
    """The fields that determine whether two artifacts are comparable. 'notes' and 
    dimension fields are excluded: two bundles with different K remain comparable 
    on shared categories, whereas two with different suffixes do not.
    """
        return (self.suffix, self.join, self.pool, self.dataset, self.revision)

    def compatible_with(self, other):
        return self.key() == other.key()

    def diff(self, other):
            """The fields that differ, formatted for readable error messages."""
        a, b = asdict(self), asdict(other)
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    def require_compatible(self, other, what="comparison"):
        if not self.compatible_with(other):
            d = self.diff(other)
            righe = "\n".join("    %-12s %r  versus  %r" % (k, v[0], v[1])
                              for k, v in d.items())
            raise ValueError(
                "%s between incompatible protocol:\n%s\n"
                "  Two artifacts do not measure the same object.. recreates with "
                "same protocol, or declare explicity that the comparison is between "
                "different conventions." % (what, righe))

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        if d is None:
            return None
        campi = {f for f in Protocol.__dataclass_fields__}
        return Protocol(**{k: v for k, v in d.items() if k in campi})

    def with_(self, **kw):
        """Copy with a few fields changed (Protocol e' immutabile)."""
        return replace(self, **kw)

    def label(self):
   """Short string representation for filenames that exposes the chosen convention."""
        parti = ["s%d" % self.seed]
        if self.k_relations:
            parti.append("K%d" % self.k_relations)
        if self.pairs_per_relation:
            parti.append("n%d" % self.pairs_per_relation)
        if self.suffix != ".":
            parti.append("nodot")
        if self.join != "space":
            parti.append("rawjoin")
        if self.pool != "last":
            parti.append(self.pool)
        return "_".join(parti)


# Il protocollo dei numeri pubblicati che vengono da truth_probe e anatomy.
CANONICAL = Protocol(suffix=".", join="space", pool="last", dtype="float32")

# Il protocollo storico della famiglia dizionari (categories.py,
# crea_dizionario.py). Si conserva per riprodurre i bundle gia' prodotti.
LEGACY_DICT = Protocol(suffix="", join="raw", pool="last", dtype="float32")
