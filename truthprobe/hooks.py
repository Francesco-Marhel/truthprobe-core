# -*- coding: utf-8 -*-
"""
truthprobe.hooks

La scomposizione additiva del blocco, e il cancello che dice quando non e'
valida.

IL PRINCIPIO
Tutto il lavoro sulla meccanica poggia su una sola identita':

    h[b+1] = h[b] + a[b] + f[b]

dove a[b] e f[b] sono i vettori che attenzione e FFN AGGIUNGONO al flusso
residuo. Se quell'identita' non tiene, ogni numero a valle e' privo di
significato. Per questo il cancello non e' un avviso: e' una condizione di
esecuzione, e chi lo ignora produce numeri sbagliati senza accorgersene.

Il cancello usa la MEDIANA dell'errore relativo, non il massimo: dove un blocco
aggiunge quasi nulla il denominatore e' minuscolo e il rapporto esplode, quindi
il massimo e' fragile mentre la mediana riporta il caso tipico.

E richiede float32. In bfloat16 l'identita' fallisce per cancellazione
catastrofica: si sottraggono stati grandi per ottenere delta piccoli, e la
precisione ridotta si mangia il risultato.

LE VARIANTI DI ARCHITETTURA, E QUALI ROMPONO COSA

  pre-norm            Llama, Qwen, Mistral. Il modulo scrive direttamente nel
                      residuo. Si aggancia l'uscita del modulo.

  sandwich norm       Gemma-2 e Gemma-3. Il vettore che entra nel residuo e'
                      l'uscita del modulo passata per una post-norma. Si
                      aggancia la POST-NORMA. Rilevamento per presenza
                      congiunta delle due norme del feedforward, mai per il
                      nome post_attention_layernorm, che su Llama e Qwen esiste
                      ma e' la PRE-norma dell'MLP: e' una trappola di
                      denominazione che fa fallire il cancello a 0.94.

  bias su o_proj      GPT-2 e altri. La somma dei contributi per testa NON e'
                      l'uscita di o_proj: manca il bias, che e' un termine
                      unico e non attribuibile a nessuna testa. Viene riportato
                      a parte, come il termine di norma nello split sandwich.
                      Nelle differenze intra-coppia si cancella, ma il cancello
                      lo deve sapere.

  blocchi paralleli   GPT-J, Falcon, PaLM. Attenzione e FFN leggono ENTRAMBI lo
                      stesso stato normalizzato invece che in sequenza.
                      L'identita' additiva regge, quindi la scomposizione e'
                      valida, ma NON esiste uno stato pre-FFN: il controllo di
                      provenienza con il pre frame li' misura un'altra cosa e
                      va reinterpretato.

  post-norm vero      La norma e' applicata alla SOMMA: h = norm(h + attn(h)).
                      Il flusso residuo non e' additivo e la scomposizione non
                      esiste. Va rifiutata, non aggiustata. Il cancello la
                      intercetta comunque.

  MoE                 Il blocco resta additivo, quindi la scomposizione a
                      livello di blocco vale senza modifiche. Cambia DENTRO
                      l'FFN, che diventa una somma pesata di esperti scelti da
                      un router: la scomposizione per esperto e' un oggetto
                      nuovo, non una variante di questa.

  head_dim diverso    Su Gemma head_dim per n_heads non fa hidden_size. La
                      dimensione della testa si ricava sempre da
                      o_proj.in_features, mai dividendo hidden_size.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List

import torch


# =====================================================================
#  descrizione dell'architettura
# =====================================================================
@dataclass
class Wiring:
    """DOVE guardare dentro un blocco. Separato da COSA fare con quello che si
    cattura, che e' il resto della libreria e non cambia mai.

    Serve perche' i modelli chiamano le stesse cose con nomi diversi, e nessun
    rilevamento automatico puo' coprire un'architettura che non esisteva quando
    la libreria e' stata scritta. Invece di aggiungere un caso alla libreria
    ogni volta, si passa il cablaggio dall'esterno:

        w = Wiring(
            attn_write = lambda L: L.post_attention_layernorm,  # cio' che entra nel residuo
            ffn_write  = lambda L: L.post_feedforward_layernorm,
            out_proj   = lambda L: L.self_attn.o_proj,
            pre_norm_in= lambda L: L.post_attention_layernorm,  # solo se sandwich
        )
        arch = describe(model, wiring=w)

    Ogni campo e' una funzione che riceve il blocco e restituisce il modulo da
    agganciare. Quello che non serve resta None.

    Il cancello di identita' verifica comunque il risultato: se il cablaggio e'
    sbagliato, l'errore relativo esplode e la libreria si ferma invece di
    produrre numeri privi di significato. Per questo passare il cablaggio a mano
    non e' pericoloso: e' una proposta che viene verificata.
    """
    attn_write: Optional[object] = None    # modulo la cui uscita entra nel residuo
    ffn_write: Optional[object] = None
    out_proj: Optional[object] = None      # per la scomposizione per testa
    post_norm: Optional[object] = None     # la post-norma, se interposta
    router: Optional[object] = None        # MoE: il modulo che sceglie gli esperti
    experts: Optional[object] = None       # MoE: la lista degli esperti, se lo e'
    expert_pack: Optional[object] = None   # MoE: il modulo che tiene i pesi come
                                           # tensori 3D invece che come moduli
    router_weight: Optional[object] = None # MoE: la matrice del router, [E, d]
    act_fn: Optional[object] = None        # SwiGLU: l'attivazione, dove si legge il gate
    gate_proj: Optional[object] = None     # SwiGLU: il ramo che decide QUANTO passa
    up_proj: Optional[object] = None       # SwiGLU: il ramo che decide COSA passa
    down_proj: Optional[object] = None     # SwiGLU: la proiezione di uscita
    notes: str = ""

    def resolve(self, layer, field):
        fn = getattr(self, field, None)
        return fn(layer) if callable(fn) else fn


@dataclass
class Architecture:
    family: str
    n_blocks: int
    d_model: int
    n_heads: int
    head_dim: int
    sandwich: bool
    o_proj_bias: bool
    parallel: bool
    moe: bool
    n_experts: Optional[int] = None
    top_k: Optional[int] = None            # MoE: esperti attivi per token
    norm_style: str = "pre"                # "pre" | "sandwich" | "post"
    unit_offset: bool = False              # la RMSNorm moltiplica per (1+w)
    residual_multiplier: float = 1.0       # Granite scala l'uscita del blocco
    notes: List[str] = None
    wiring: Optional[Wiring] = None        # cablaggio esplicito, se fornito

    def to_dict(self):
        d = asdict(self)
        d["wiring"] = ("esplicito: " + (self.wiring.notes or "fornito dall'utente")
                       if self.wiring else "rilevato automaticamente")
        return d

    def summary(self):
        r = ["%s: %d blocchi, d=%d, %d teste da %d"
             % (self.family, self.n_blocks, self.d_model, self.n_heads, self.head_dim)]
        if self.residual_multiplier != 1.0:
            r.append("moltiplicatore residuo %.4f applicato ai contributi"
                     % self.residual_multiplier)
        if self.sandwich:
            r.append("norme in uscita (%s): si aggancia la post-norma, "
                     "guadagno %s" % (self.norm_style,
                                      "(1+w)" if self.unit_offset else "w"))
        if self.o_proj_bias:
            r.append("o_proj ha bias: la somma per testa lo esclude, riportato a parte")
        if self.parallel:
            r.append("blocchi paralleli: attenzione e FFN leggono lo STESSO "
                     "stato e scrivono insieme. Non esiste uno stato pre-FFN, e "
                     "l'FFN di un blocco non vede l'attenzione del suo blocco")
        if self.moe:
            r.append("MoE con %s esperti, %s attivi per token: il blocco resta "
                     "additivo, l'FFN no"
                     % (self.n_experts or "?", self.top_k or "?"))
        if self.wiring:
            r.append("cablaggio esplicito%s"
                     % ((": " + self.wiring.notes) if self.wiring.notes else ""))
        for n in (self.notes or []):
            r.append(n)
        return r


def _layers(model):
    inner = getattr(model, "model", model)
    for attr in ("layers", "h", "blocks", "decoder"):
        obj = getattr(inner, attr, None)
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise RuntimeError("non trovo i blocchi del decoder in questo modello")


def _attn(layer):
    for attr in ("self_attn", "attn", "attention"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError("non trovo il modulo di attenzione nel blocco")


def _out_proj(attn):
    for attr in ("o_proj", "out_proj", "dense", "c_proj", "wo"):
        m = getattr(attn, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m
    raise RuntimeError("non trovo la proiezione di uscita dell'attenzione")


def _mlp(layer):
    for attr in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError("non trovo l'MLP nel blocco")


def describe(model, wiring=None):
    """Rileva la variante di architettura. Il rilevamento e' esplicito e
    conservativo: quello che non riconosce lo dichiara invece di indovinare."""
    layers = _layers(model)
    layer = layers[0]
    cfg = model.config
    attn = _attn(layer)
    op = _out_proj(attn)
    mlp = _mlp(layer)
    notes = []

    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    in_f = getattr(op, "in_features", None) or op.weight.shape[1]
    head_dim = in_f // n_heads
    if head_dim * n_heads != d_model:
        notes.append("head_dim per n_heads (%d) non fa d_model (%d): normale su "
                     "Gemma, la dimensione viene da o_proj" % (head_dim * n_heads, d_model))

    # QUALE modulo scrive nel residuo. La domanda non e' se il blocco abbia
    # norme prima, ma se ne abbia dopo: dove esiste una post-norma dell'FFN, il
    # vettore che entra nel residuo e' l'uscita della norma, non quella
    # dell'MLP, e l'aggancio va spostato di un modulo. Gemma-2 ha entrambe le
    # norme, OLMo 2 solo quelle in uscita: il punto di aggancio e' lo stesso.
    pre_ff = hasattr(layer, "pre_feedforward_layernorm")
    post_ff = hasattr(layer, "post_feedforward_layernorm")
    sandwich = post_ff
    if post_ff and pre_ff:
        norm_style = "sandwich"
    elif post_ff:
        norm_style = "post"
        notes.append("norme solo in uscita (stile OLMo 2): le due post-norme "
                     "scrivono nel residuo e l'FFN legge il residuo grezzo. "
                     "L'aggancio e' quello sandwich, il pre frame no: qui non "
                     "esiste uno stato normalizzato in ingresso all'FFN")
    else:
        norm_style = "pre"

    # CON QUALE convenzione la norma moltiplica. Gemma tiene il parametro come
    # scarto e usa (1 + w); Llama, Qwen e OLMo usano w. Sbagliarla non fa
    # crashare niente: sposta la scomposizione per testa di un fattore che
    # sembra un risultato. Il cancello per testa lo intercetta comunque.
    # IL MOLTIPLICATORE RESIDUO. Granite scrive nel residuo
    #     residual + uscita_del_modulo * residual_multiplier
    # con un fattore intorno a 0.2. Agganciando l'uscita del modulo si cattura
    # il valore PRIMA della moltiplicazione, quindi a + f vale circa cinque
    # volte il delta e il cancello di identita' fallisce con un errore
    # dell'ordine dell'unita'. Non e' un difetto degli hook: e' un fattore che
    # esiste solo in alcune famiglie e va letto dal config.
    rmul = float(getattr(cfg, "residual_multiplier", 1.0) or 1.0)
    if rmul != 1.0:
        notes.append("moltiplicatore residuo %.4f: i contributi vengono scalati "
                     "prima di entrare nel residuo, e la libreria lo applica "
                     "ai vettori restituiti" % rmul)

    nrm = getattr(layer, "post_attention_layernorm", None)
    unit_offset = nrm is not None and "gemma" in type(nrm).__name__.lower()
    if sandwich and not unit_offset:
        notes.append("la post-norma moltiplica per w, non per (1+w)")

    moe = any(k in type(mlp).__name__.lower() for k in ("moe", "sparse", "expert"))
    packed = getattr(mlp, "experts", None)
    if packed is not None and hasattr(packed, "gate_up_proj"):
        moe = True
        notes.append("esperti impacchettati come tensori 3D, non come moduli: "
                     "la scomposizione per esperto e' analitica, come per le teste")
    n_exp = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    top_k = (getattr(cfg, "num_experts_per_tok", None)
             or getattr(cfg, "moe_top_k", None)
             or getattr(cfg, "num_selected_experts", None))
    if moe and not (packed is not None and hasattr(packed, "gate_up_proj")):
        notes.append("esperti come moduli: la scomposizione per esperto si fa "
                     "con hook, uno per esperto")

    # Alcune famiglie il parallelismo lo hanno nel codice del blocco ma NON
    # nel config: Phi lo scrive a mano dentro PhiDecoderLayer.forward come
    # attn_out + ffn_out + residual, senza nessun flag da leggere. Senza questo
    # elenco il rilevamento tace e chi legge il log crede che il blocco sia
    # sequenziale. La scomposizione additiva resta esatta in entrambi i casi:
    # cambia l'interpretazione, non il cancello.
    MODELLI_PARALLELI = {"phi", "phi3", "gptj", "gpt_neox", "falcon", "rw"}
    parallel = bool(getattr(cfg, "parallel_attn", False)
                    or getattr(cfg, "use_parallel_residual", False)
                    or getattr(cfg, "model_type", "") in MODELLI_PARALLELI)
    if parallel:
        notes.append("con blocchi paralleli il pre frame non e' lo stato pre-FFN")

    fam = getattr(cfg, "model_type", type(model).__name__)
    if wiring is not None:
        notes.append("il cablaggio esplicito ha la precedenza sul rilevamento")
    return Architecture(family=fam, n_blocks=len(layers), d_model=d_model,
                        n_heads=n_heads, head_dim=head_dim, sandwich=sandwich,
                        o_proj_bias=(getattr(op, "bias", None) is not None),
                        parallel=parallel, moe=moe, n_experts=n_exp, top_k=top_k,
                        norm_style=norm_style, unit_offset=unit_offset,
                        residual_multiplier=rmul,
                        notes=notes,
                        wiring=wiring)


# =====================================================================
#  cattura dei contributi additivi
# =====================================================================
def offloaded_tensor(model, module, attr="weight"):
    """Recover a parameter that accelerate left on the meta device.

    With device_map='auto' the weights that do not fit in VRAM stay as
    placeholders: shape and dtype, no data. accelerate keeps the real values in
    a map attached to the module or to one of its ancestors, and materialises
    them only while that module runs, putting them back afterwards.

    PyTorch's own hooks run OUTSIDE that window. accelerate does not register a
    pre-hook: it REPLACES `module.forward`, and `Module.__call__` runs every
    forward-pre-hook before calling it and every forward-hook after it returns.
    So a pre-hook sees the weight before it is materialised and a post-hook sees
    it after it has been put back. "Read it inside the forward pass" is not
    reachable from a hook: the map has to be read directly.

    Returns None when there is no map, so the caller can fail with its own
    message instead of receiving zeros."""
    if model is None or module is None:
        return None
    qual = None
    for n, m in model.named_modules():
        if m is module:
            qual = n
            break
    if qual is None:
        return None
    for owner_name, owner in model.named_modules():
        wmap = getattr(getattr(owner, "_hf_hook", None), "weights_map", None)
        if wmap is None:
            continue
        if owner_name == qual:
            key = attr
        elif owner_name and qual.startswith(owner_name + "."):
            key = qual[len(owner_name) + 1:] + "." + attr
        else:
            continue
        try:
            t = wmap[key]
        except Exception:
            continue
        if t is not None:
            return t
    return None


def _weight(t, name="weight"):
    """Read a parameter that may live on the meta device.

    With device_map='auto' the weights not currently needed are placeholders:
    they carry shape and dtype but no data, and reading them raises. accelerate
    keeps the real values in a hook attached to the module, so they must be
    materialised rather than dereferenced. Failing loudly here is better than
    returning a tensor of zeros that would look like a legitimate measurement."""
    if t is None:
        return None
    if getattr(t, "is_meta", False) or str(t.device) == "meta":
        raise RuntimeError(
            "the %s parameter is on the meta device: with device_map='auto' the "
            "weights are materialised only while the module runs. Either load "
            "without device_map, or read this inside the forward pass." % name)
    return t.detach().float().cpu()


def _grab(buf, key):
    def hook(_m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        buf[key] = o
    return hook


class BlockCapture:
    """Cattura, per un blocco, i vettori che attenzione e FFN aggiungono al
    residuo, e la scomposizione per testa del contributo dell'attenzione.

    Usare come contesto:

        with BlockCapture(model, arch, block) as cap:
            out = model(**enc, output_hidden_states=True)
            a, f = cap.attn(), cap.ffn()
            per_head = cap.heads()          # [B, nH, d]

    Tutte le catture avvengono nella stessa passata in avanti: la
    scomposizione per testa non costa una passata in piu', perche' il modello
    calcola comunque le uscite delle teste e qui si evita solo di buttarle.
    """

    def __init__(self, model, arch, block, position=-1):
        self.model, self.arch, self.block, self.pos = model, arch, block, position
        layers = _layers(model)
        if not 0 <= block < len(layers):
            raise ValueError("blocco %d fuori intervallo (0..%d)" % (block, len(layers) - 1))
        self.layer = layers[block]
        w = getattr(arch, "wiring", None)
        self.wiring = w
        L = self.layer
        # il cablaggio esplicito ha la precedenza; dove manca si ricade sul
        # rilevamento, cosi' si puo' sovrascrivere solo cio' che serve
        self.attn_mod = (w.resolve(L, "attn_write") if w else None) or _attn(L)
        self.op = (w.resolve(L, "out_proj") if w else None) or _out_proj(_attn(L))
        self.mlp_mod = (w.resolve(L, "ffn_write") if w else None) or _mlp(L)
        self.post = (w.resolve(L, "post_norm") if w else None)
        if self.post is None and arch.sandwich:
            self.post = getattr(L, "post_attention_layernorm", None)
        self.router = (w.resolve(L, "router") if w else None)
        self.experts = (w.resolve(L, "experts") if w else None)
        # percorsi interni dell'FFN. In una SwiGLU il blocco calcola
        #     f = W_down ( silu(W_gate x) * (W_up x) )
        # e i due fattori hanno ruoli diversi: il GATE decide quanto passa, il
        # VALORE decide che cosa passa. Sono due domande separate, e la
        # distinzione e' misurabile solo se si catturano separatamente.
        m = self.mlp_mod
        # il GATE si cattura all'uscita dell'ATTIVAZIONE, non alla proiezione.
        # Applicare silu a mano assumerebbe quale non linearita' il modello usa,
        # e su un kernel fuso o una variante quell'assunzione e' invisibile e
        # sbagliata. act_fn e' il modulo che il modello esegue davvero.
        self.act_fn = (w.resolve(L, "act_fn") if w else None) or \
            getattr(m, "act_fn", None)
        self.gate_proj = (w.resolve(L, "gate_proj") if w else None) or \
            getattr(m, "gate_proj", None) or getattr(m, "w1", None)
        self.up_proj = (w.resolve(L, "up_proj") if w else None) or \
            getattr(m, "up_proj", None) or getattr(m, "w3", None)
        self.down_proj = (w.resolve(L, "down_proj") if w else None) or \
            getattr(m, "down_proj", None) or getattr(m, "w2", None)
        self.buf, self.handles = {}, []

    def __enter__(self):
        L, b, w = self.layer, self.buf, getattr(self.arch, "wiring", None)
        # con cablaggio esplicito i moduli sono gia' quelli che scrivono nel
        # residuo; senza, su sandwich norm si aggancia la post-norma
        if w is not None and w.attn_write is not None:
            tgt_a, tgt_f = self.attn_mod, self.mlp_mod
        elif self.arch.sandwich:
            tgt_a = L.post_attention_layernorm
            tgt_f = L.post_feedforward_layernorm
        else:
            tgt_a, tgt_f = self.attn_mod, self.mlp_mod
        self.handles = [
            tgt_a.register_forward_hook(_grab(b, "a")),
            tgt_f.register_forward_hook(_grab(b, "f")),
            self.op.register_forward_pre_hook(
                lambda m, i: b.__setitem__("z", i[0])),
        ]
        if self.post is not None:
            self.handles.append(self.post.register_forward_pre_hook(
                lambda m, i: b.__setitem__("pre_a", i[0])))
        for name, mod in (("gate", self.act_fn or self.gate_proj),
                          ("up", self.up_proj),
                          ("down_in", self.down_proj)):
            if mod is None:
                continue
            if name == "down_in":
                self.handles.append(mod.register_forward_pre_hook(
                    lambda m, i, k=name: b.__setitem__(k, i[0])))
            else:
                self.handles.append(mod.register_forward_hook(_grab(b, name)))
        if self.router is not None:
            # NON _grab: quello tiene solo il primo elemento di una tupla, e un
            # router ne restituisce fino a tre. Punteggi e indici calcolati dal
            # modello sono piu' fedeli di quelli ricalcolati qui, perche' una
            # implementazione puo' non rinormalizzare sui primi k: rifare il
            # softmax darebbe pesi che sommano a uno dove gli originali non lo
            # fanno. Si conserva tutto e si decide dopo.
            self.handles.append(self.router.register_forward_hook(
                lambda m, i, o: b.__setitem__("route", o)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def _w(self, module, name, attr="weight"):
        """Un peso del blocco, su CPU e in float32, anche se e' offloadato.

        Unico punto in cui la libreria legge un parametro. Se il tensore e' un
        segnaposto si tenta la mappa di accelerate; se non c'e' nemmeno quella
        si alza l'errore invece di restituire zeri, che passerebbero per una
        misura legittima."""
        t = getattr(module, attr, None)
        if t is None:
            return None
        if getattr(t, "is_meta", False) or str(getattr(t, "device", "")) == "meta":
            rec = offloaded_tensor(self.model, module, attr)
            if rec is None:
                raise RuntimeError(
                    "the %s parameter is on the meta device and accelerate "
                    "exposes no weights map for it: load without device_map, or "
                    "pin the block being read to a real device." % name)
            return rec.detach().float().cpu()
        return t.detach().float().cpu()

    def _gain(self, norm, name):
        """Il guadagno di una RMSNorm, con la convenzione giusta.

        Gemma memorizza il parametro come scarto e moltiplica per (1 + w).
        Llama, Qwen, OLMo moltiplicano per w. Applicare la convenzione sbagliata
        sposta la scomposizione per testa di una quantita' che sembra un
        risultato. La distinzione e' rilevata in describe() e verificata dal
        cancello per testa."""
        g = self._w(norm, name)
        if g is None:
            return None
        return (1.0 + g) if getattr(self.arch, "unit_offset", False) else g

    def _at(self, t):
        return t[:, self.pos, :].detach().float().cpu()

    def _res(self, t):
        """Cio' che ENTRA nel residuo, non cio' che il modulo restituisce.

        Dove esiste un moltiplicatore residuo i due differiscono per un
        fattore. Si applica qui e non dentro _at, perche' _at serve anche a
        leggere lo stato in INGRESSO a una norma, che il moltiplicatore non
        tocca. Scalando quello si romperebbe il calcolo dell'rms per testa.
        """
        m = getattr(self.arch, "residual_multiplier", 1.0)
        return t if m == 1.0 else t * m

    def attn(self):
        return self._res(self._at(self.buf["a"]))

    def ffn(self):
        return self._res(self._at(self.buf["f"]))

    def heads(self, which=None):
        """Contributo di ciascuna testa al residuo: [B, nH, d].

        o_proj e' lineare, quindi il contributo della testa h e' la sua fetta
        di z proiettata con la sua fetta di W_o. Su architetture sandwich la
        post-norma e' lineare A FRASE FISSATA, perche' rms(x) e' uno scalare:
        si applica lo stesso fattore a ogni fetta e la somma torna esatta.
        Il bias di o_proj, se c'e', NON e' attribuibile a nessuna testa e viene
        restituito separatamente da bias_term().

        DISPOSITIVI. Le attivazioni vengono portate su CPU da _at, mentre i PESI
        stanno dove sta il modello, di norma su GPU. Ogni peso letto qui va
        quindi spostato esplicitamente, altrimenti l'operazione trova operandi
        su dispositivi diversi. Il calcolo si fa su CPU perche' e' minuscolo e
        perche' il risultato serve su CPU comunque."""
        nH, hd = self.arch.n_heads, self.arch.head_dim
        W = self._w(self.op, "o_proj")                         # [d, nH*hd]
        Wh = W.view(W.shape[0], nH, hd)
        z = self._at(self.buf["z"]).view(-1, nH, hd)           # [B, nH, hd]
        if which is not None:
            # tenere tutte le teste costa n_heads volte il residuo per frase, e
            # su un modello grande e' quello che riempie la memoria prima dei
            # pesi. Chi vuole solo verificare la scomposizione ne chiede due.
            sel = torch.tensor(list(which))
            Wh, z = Wh[:, sel, :], z[:, sel, :]
        per = torch.einsum("dnh,bnh->bnd", Wh, z)              # pre-norma
        if self.post is not None and "pre_a" in self.buf:
            pre = self._at(self.buf["pre_a"])                  # x prima della post-norma
            eps = getattr(self.post, "eps",
                          getattr(self.post, "variance_epsilon", 1e-6))
            rms = pre.pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()
            gain = self._gain(self.post, "post-norm")
            per = (per / rms.unsqueeze(1)) * gain.view(1, 1, -1)
        return self._res(per)

    def swiglu(self, axis=None, pidx=None):
        """The two internal streams of a gated feed-forward, at the read token.

        Returns gate g (the OUTPUT of the activation), value u, the hidden state
        h fed to the down projection, and, when an axis is given, the EXACT
        gate/value split of each pair's gap along it.

        The axis is pulled back into the expanded basis, w = W_down^T v1, so the
        split is computed where the two streams live. On sandwich-norm models
        the pull-back carries the norm gain, w = W_down^T((1+gamma) v1), and the
        gap acquires a third term from the two sentences being normalised
        differently. That term belongs to neither stream and is reported apart.

        Returns None when the module is not gated: a plain two-matrix MLP has no
        gate to separate, and guessing one would invent a decomposition."""
        if not all(k in self.buf for k in ("gate", "up", "down_in")):
            return None
        g = self._at(self.buf["gate"])
        u = self._at(self.buf["up"])
        h = self._at(self.buf["down_in"])
        Wd = self._w(self.down_proj, "down_proj")                  # [d, d_i]
        out = dict(gate=g, up=u, h=h, recon=h @ Wd.T, W_down=Wd)

        if axis is None or pidx is None:
            return out

        from .stats import gate_value_split, gate_value_split_sandwich
        v = torch.as_tensor(axis).float().cpu()
        sandwich = self.post is not None and hasattr(self.mlp_mod, "gate_proj")
        post_ff = getattr(self.layer, "post_feedforward_layernorm", None)
        if post_ff is not None:
            gamma = self._gain(post_ff, "post-norm FFN")
            eps = getattr(post_ff, "eps",
                          getattr(post_ff, "variance_epsilon", 1e-6))
            w = Wd.T @ (v * gamma)
            X = (g * u) @ Wd.T
            s_all = 1.0 / torch.sqrt(X.pow(2).mean(-1) + eps)
        else:
            w = Wd.T @ v
            s_all = None

        gates, values, norms, totals = [], [], [], []
        for it, iff in pidx:
            if s_all is None:
                a_, b_, t_ = gate_value_split(g[it], u[it], g[iff], u[iff], w)
                n_ = torch.tensor(0.0)
            else:
                a_, b_, n_, t_ = gate_value_split_sandwich(
                    g[it], u[it], g[iff], u[iff], w,
                    float(s_all[it]), float(s_all[iff]))
            gates.append(float(a_)); values.append(float(b_))
            norms.append(float(n_)); totals.append(float(t_))
        gm = sum(gates) / len(gates)
        vm = sum(values) / len(values)
        nm = sum(norms) / len(norms)
        tm = sum(totals) / len(totals)
        # the share divides by the total: when the two streams nearly cancel the
        # total is a small difference of large terms and the ratio is noise
        cancel = abs(tm) < 0.25 * (abs(gm) + abs(vm))
        out.update(gate_term=gm, value_term=vm, norm_term=nm, total=tm,
                   gate_share=(None if cancel or abs(tm) < 1e-8 else gm / tm),
                   cancelling=cancel, has_norm_term=s_all is not None,
                   identity_error=abs(gm + vm + nm - tm) / max(abs(tm), 1e-8))
        return out

    def swiglu_gate(self, tol=1e-4):
        """Does g * u reproduce the hidden state actually fed to down_proj?

        A gate, not a convenience. g is captured at the activation output, so
        this checks the FUSION, not the non-linearity: if it fails the module is
        not the gated feed-forward the split assumes, or the wiring points at
        the wrong sub-module, and every share computed from it would be
        meaningless."""
        r = self.swiglu()
        if r is None:
            return float("nan"), False
        err = (r["gate"] * r["up"] - r["h"]).norm(dim=-1) \
            / r["h"].norm(dim=-1).clamp_min(1e-8)
        med = float(err.median())
        return med, med < tol

    def bias_term(self):
        """Il bias di o_proj, gia' passato per la post-norma se presente.
        Non appartiene a nessuna testa. Nelle differenze intra-coppia si
        cancella; nel cancello di identita' va contato."""
        bias = getattr(self.op, "bias", None)
        if bias is None:
            return None
        b = self._w(self.op, "o_proj bias", attr="bias")
        if self.post is not None and "pre_a" in self.buf:
            pre = self._at(self.buf["pre_a"])
            eps = getattr(self.post, "eps",
                          getattr(self.post, "variance_epsilon", 1e-6))
            rms = pre.pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()
            gain = self._gain(self.post, "post-norm")
            return self._res((b.unsqueeze(0) / rms) * gain.unsqueeze(0))
        return b.unsqueeze(0).expand(self._at(self.buf["a"]).shape[0], -1)


# =====================================================================
#  cancelli di correttezza
# =====================================================================
def experts_packed(self):
    """MoE whose experts are TENSORS, not modules.

    Some implementations keep every expert's weights in one packed parameter
    (`gate_up_proj` of shape [E, 2*d_i, d] and `down_proj` of shape [E, d, d_i])
    and dispatch by index. There is then no per-expert module to hook, and the
    decomposition has to be analytic, exactly as for attention heads.

    This is a packaging choice, not a mathematical one: `gate_up_proj[e]` is the
    same matrix an independent module would hold. It also gives MORE than hooks
    would: the weights of every expert are present, including those the router
    never selected, so one can ask what an unused expert WOULD have written.

    Returns per-expert weight views and the router matrix. The gate/up split of
    the packed tensor is a CONVENTION (first half gate, second half up) and the
    caller must verify it by reconstruction rather than trust it: swapping the
    halves would silently exchange the two terms of the split.

    Returns None when the wiring does not describe a packed layout.
    """
    w = self.wiring
    pack = (w.resolve(self.layer, "expert_pack") if w else None)
    if pack is None:
        m = self.mlp_mod
        pack = getattr(m, "experts", None)
    if pack is None or not hasattr(pack, "gate_up_proj"):
        return None

    gu = self._w(pack, "gate_up_proj", attr="gate_up_proj")   # [E, 2*d_i, d]
    dn = self._w(pack, "down_proj", attr="down_proj")         # [E, d, d_i]
    E = gu.shape[0]
    d_i = gu.shape[1] // 2
    rw = None
    if w and w.router_weight is not None:
        rw = _weight(w.resolve(self.layer, "router_weight"), "router weight")
    elif self.router is not None and hasattr(self.router, "weight"):
        rw = self._w(self.router, "router weight")       # [E, d]

    return dict(n_experts=E, d_inter=d_i,
                gate=gu[:, :d_i, :], up=gu[:, d_i:, :], down=dn,
                router_weight=rw, act_fn=getattr(pack, "act_fn", None))


def expert_contribution(self, packed, e, x):
    """What expert e writes for the states x, computed from its weights.

    x is [n, d], the input to the feed-forward at the read position. Returns
    [n, d], the vector that expert would add BEFORE the router weight is
    applied. Multiply by that weight to obtain the actual contribution.

    Computing rather than hooking is what makes the counterfactual possible:
    this works for experts the router did not select on these inputs, which is
    the only way to ask whether an unused expert would have written truth."""
    act = packed["act_fn"] or torch.nn.functional.silu
    g = act(x @ packed["gate"][e].T)
    u = x @ packed["up"][e].T
    return (g * u) @ packed["down"][e].T


def router_directions(self, packed=None):
    """The router matrix as K directions in the residual stream.

    Each row of the router weight is a direction: the token's state is scored
    against all of them and the top ones win. So the rows can be compared with
    the category axes by the same cosine machinery used everywhere else, with
    no forward pass at all, and the question 'does the router partition the
    space the way the dictionary does' becomes one Mantel test between two
    matrices already in hand."""
    packed = packed or self.experts_packed()
    if packed is None or packed["router_weight"] is None:
        return None
    R = packed["router_weight"]
    R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return dict(directions=R, cosines=R @ R.T, n_experts=R.shape[0])


BlockCapture.experts_packed = experts_packed
BlockCapture.expert_contribution = expert_contribution
BlockCapture.router_directions = router_directions


def routing(self, top_k=None):
    """MoE: quali esperti sono stati scelti al token di lettura e con che peso.

    Richiede un cablaggio con router. L'uscita dei router varia fra
    implementazioni: qui si accettano i logit grezzi [B, S, E] oppure una
    tupla di cui si prende il primo elemento. I pesi sono softmax sui top_k,
    che e' la forma piu' comune; se il tuo modello normalizza diversamente,
    leggi 'logits' e ricalcola come serve.

    ATTENZIONE, e' il punto che conta per il disegno sperimentale: il router
    instrada PER TOKEN, non per frase. Su una coppia minimale le due frasi
    differiscono in un token, quindi la domanda vera e' se quel token cambia
    gli esperti attivati al punto di lettura.
    """
    if "route" not in self.buf:
        raise RuntimeError("nessun router agganciato: serve un Wiring con router")
    r = self.buf["route"]
    tup = r if isinstance(r, tuple) else (r,)
    logits = tup[0]

    # DUE LAYOUT. Alcune implementazioni appiattiscono gli stati PRIMA del
    # router, quindi i logit arrivano [B*S, E] invece di [B, S, E]. La forma va
    # dedotta, non assunta: leggere la posizione sbagliata darebbe il routing di
    # un altro token senza che nulla lo segnali.
    if logits.dim() == 3:
        lg = logits[:, self.pos, :]
    else:
        B, S = self.buf["a"].shape[:2] if "a" in self.buf else (1, logits.shape[0])
        lg = logits.view(B, S, -1)[:, self.pos, :]
    lg = lg.detach().float().cpu()

    # SE IL MODELLO HA GIA' DECISO, si prende la sua decisione. Ricalcolare
    # top-k e softmax qui darebbe pesi diversi da quelli davvero usati ogni
    # volta che l'implementazione non rinormalizza sui primi k: il softmax su un
    # sottoinsieme somma a uno, le probabilita' originali no.
    if len(tup) >= 3:
        sc, ix = tup[1], tup[2]
        if sc.dim() == 3:
            sc, ix = sc[:, self.pos, :], ix[:, self.pos, :]
        else:
            B, S = self.buf["a"].shape[:2] if "a" in self.buf else (1, sc.shape[0])
            sc = sc.view(B, S, -1)[:, self.pos, :]
            ix = ix.view(B, S, -1)[:, self.pos, :]
        return dict(logits=lg, experts=ix.detach().cpu(),
                    weights=sc.detach().float().cpu(), source="model")

    k = top_k or getattr(self.arch, "top_k", None) or 2
    k = min(k, lg.shape[-1])
    val, idx = lg.topk(k, dim=-1)
    return dict(logits=lg, experts=idx, weights=torch.softmax(val, dim=-1),
                source="recomputed")


BlockCapture.routing = routing


def identity_gate(h_before, h_after, a, f, tol=1e-4, dtype_used="float32"):
    """a + f deve ricostruire h[b+1] - h[b].

    Restituisce (mediana dell'errore relativo, esito). Se l'esito e' falso, i
    numeri a valle NON vanno letti: la scomposizione non descrive il modello.
    Cause tipiche di un fallimento, in ordine di frequenza: hook sul modulo
    invece che sulla post-norma su architetture sandwich, uso di bfloat16,
    architettura post-norm vera in cui la scomposizione non esiste.
    """
    # detach: i cancelli sono misure, non parte di nessun calcolo differenziabile.
    # Senza questo, se il chiamante dimentica torch.no_grad() il grafo resta vivo
    # e la memoria cresce a ogni passata fino a esaurire il dispositivo.
    delta = (h_after - h_before).detach().float()
    err = (a.detach().float() + f.detach().float() - delta).norm(dim=-1)
    rel = err / delta.norm(dim=-1).clamp_min(1e-8)
    med = float(rel.median())
    ok = med < tol
    msg = ""
    if not ok:
        msg = "identita' additiva fallita (mediana %.2e)." % med
        if dtype_used != "float32":
            msg += " Il dtype e' %s: il cancello richiede float32." % dtype_used
        else:
            msg += (" Controllare il punto di aggancio: su architetture sandwich "
                    "va agganciata la post-norma, non il modulo.")
    return med, ok, msg


def head_gate(per_head, a_captured, bias=None, tol=1e-4):
    """La somma dei contributi per testa deve ricostruire il contributo di
    attenzione catturato indipendentemente, piu' il bias se presente."""
    s = per_head.detach().sum(1)
    if bias is not None:
        s = s + bias.detach()
    ac = a_captured.detach()
    err = (s - ac).norm(dim=-1) / ac.norm(dim=-1).clamp_min(1e-8)
    med = float(err.median())
    return med, med < tol
