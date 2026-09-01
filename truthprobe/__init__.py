# -*- coding: utf-8 -*-
"""
truthprobe

Il nucleo condiviso degli strumenti della serie sulla direzione di verita'.

Non e' un framework: e' il posto in cui stanno le operazioni che ogni
strumento rifaceva per conto suo, piu' l'oggetto Protocol che impedisce a due
strumenti di divergere sulle convenzioni senza che nessuno se ne accorga.

    from truthprobe import Protocol, CANONICAL
    from truthprobe.geometry import fit_axis, project_fields
    from truthprobe.stats import auc_score, mantel, consensus_gauge

La versione finisce dentro ogni artefatto salvato, cosi' un bundle sa da quale
codice e' stato prodotto.
"""

__version__ = "0.1.0"

from .protocol import Protocol, CANONICAL, LEGACY_DICT, COUNTERFACT_REPO, COUNTERFACT_REV

__all__ = ["Protocol", "CANONICAL", "LEGACY_DICT",
           "COUNTERFACT_REPO", "COUNTERFACT_REV", "__version__"]
