# -*- coding: utf-8 -*-
"""
truthprobe.protocol

Le convenzioni che DEVONO essere identiche in ogni strumento, raccolte in un
oggetto solo che viaggia con i dati e finisce dentro ogni artefatto.

PERCHE' NON SONO FLAG
Durante lo sviluppo di questa serie due strumenti dello stesso progetto hanno
costruito le frasi in modo diverso per mesi senza che nessuno se ne accorgesse:
uno concatenava "prompt target." e l'altro "prompt target". Le due convenzioni
producono assi con coseno +0.52, cioe' direzioni diverse, mentre l'arrangement
sopravvive a Mantel +0.775. Nessuno strumento segnalava nulla, perche' la
convenzione era un dettaglio interno a ciascun file.

Un flag con un default non avrebbe impedito quella divergenza: avrebbe solo
spostato il default in due posti. Qui invece il Protocol e' obbligatorio, viene
scritto dentro ogni bundle, e il confronto fra due bundle lo verifica PRIMA di
calcolare qualunque cosa. Una divergenza diventa un errore visibile invece che
un numero sbagliato.

TRE CATEGORIE DI PARAMETRI, TRATTATE DIVERSAMENTE
  protocollo   suffisso, dataset, revisione, seed, K, n, precisione, pooling.
               Cambiano i numeri e devono essere condivise: stanno qui.
  analisi      blocco, componente, permutazioni. Cambiano il risultato ma sono
               proprie di ciascuna misura: stanno nei metadati dell'artefatto.
  esecuzione   batch, dispositivo, cartella di uscita. Non cambiano i numeri:
               restano flag liberi e non vengono registrati.
"""

from dataclasses import dataclass, asdict, field, replace
from typing import Optional

COUNTERFACT_REPO = "NeelNanda/counterfact-tracing"
COUNTERFACT_REV = "c945b082ca08d0a8f3ba227fb78404a09614c36e"
TRUTHFULQA_REV = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


@dataclass(frozen=True)
class Protocol:
    """Le convenzioni di costruzione dei dati e di lettura degli stati.

    suffix: cosa si attacca dopo il target.
        "."  l'ultimo token e' identico nelle due frasi della coppia, quindi
             l'identita' del token esce dalla misura e cio' che si legge e'
             arrivato al punto attraverso l'attenzione. E' anche la condizione
             che rende sensato il controllo "curva a 0.500 al livello 0".
        ""   l'ultimo token e' la parola target, diversa fra vero e falso: la
             misura include l'identita' del token. Legittimo ma e' un'altra
             domanda, e i due casi NON sono confrontabili fra loro.

    join: come si uniscono prompt e target.
        "space"  prompt.strip() + " " + target.strip(), esplicito nel codice
        "raw"    prompt.strip() + target, cioe' lo spazio arriva dal dataset.
                 E' la convenzione storica di categories.py e crea_dizionario.py
                 e si conserva solo per riprodurre bundle vecchi: dipende da una
                 proprieta' del dataset invece che dal codice.
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
            raise ValueError("join deve essere 'space' o 'raw', non %r" % self.join)
        if self.pool not in ("last", "mean"):
            raise ValueError("pool deve essere 'last' o 'mean', non %r" % self.pool)
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype non riconosciuto: %r" % self.dtype)

    # ---- costruzione delle frasi: l'UNICO posto in tutta la libreria ----
    def sentence(self, prompt, target):
        """L'unica funzione che concatena prompt e target. Se un giorno serve
        cambiare, si cambia qui e cambia ovunque."""
        p = str(prompt).strip()
        if self.join == "space":
            return p + " " + str(target).strip() + self.suffix
        return p + str(target) + self.suffix

    # ---- identita' e confronto ----
    def key(self):
        """I campi che rendono due artefatti confrontabili. 'notes' e i campi
        di dimensione non entrano: due bundle con K diverso restano
        confrontabili sulle categorie condivise, due con suffix diverso no."""
        return (self.suffix, self.join, self.pool, self.dataset, self.revision)

    def compatible_with(self, other):
        return self.key() == other.key()

    def diff(self, other):
        """I campi che differiscono, per messaggi d'errore leggibili."""
        a, b = asdict(self), asdict(other)
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    def require_compatible(self, other, what="confronto"):
        if not self.compatible_with(other):
            d = self.diff(other)
            righe = "\n".join("    %-12s %r  contro  %r" % (k, v[0], v[1])
                              for k, v in d.items())
            raise ValueError(
                "%s fra protocolli incompatibili:\n%s\n"
                "  I due artefatti non misurano lo stesso oggetto. Rigenerane uno "
                "con lo stesso protocollo, oppure dichiara esplicitamente che il "
                "confronto e' fra convenzioni diverse." % (what, righe))

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        if d is None:
            return None
        campi = {f for f in Protocol.__dataclass_fields__}
        return Protocol(**{k: v for k, v in d.items() if k in campi})

    def with_(self, **kw):
        """Copia con qualche campo cambiato (Protocol e' immutabile)."""
        return replace(self, **kw)

    def label(self):
        """Etichetta corta per nomi di file, che rende visibile la convenzione."""
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
