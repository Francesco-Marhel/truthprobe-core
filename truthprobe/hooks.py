# -*- coding: utf-8 -*-
"""
truthprobe.hooks

Additive block decomposition and the gating mechanism that identifies invalid states.

THE PRINCIPLE
All mechanistic analysis relies on a single fundamental identity:

    h[b+1] = h[b] + a[b] + f[b]

where a[b] and f[b] are the vectors that attention and FFN ADD to the residual
stream. If this identity does not hold, every downstream numerical value is 
meaningless. For this reason, the gate is not a mere warning: it is an execution 
condition, and ignoring it leads to silently producing incorrect numbers.

The gate uses the MEDIAN relative error, not the maximum: when a block adds 
almost nothing, the denominator becomes tiny and the ratio explodes. Therefore, 
the maximum is fragile, whereas the median accurately reports the typical case.

It strictly requires float32 precision. In bfloat16, the identity fails due to 
catastrophic cancellation: large states are subtracted to obtain small deltas, 
and the reduced precision eats away the actual result.


ARCHITECTURAL VARIANTS: WHAT THEY BREAK AND HOW

  pre-norm            Llama, Qwen, Mistral. The module writes directly into the
                      residual stream. The hook is attached to the module output.

  sandwich norm       Gemma-2 and Gemma-3. The vector entering the residual stream 
                      is the module output passed through a post-normalization layer. 
                      The hook must capture the POST-NORM. Detection relies on the 
                      joint presence of both feedforward norms, never on names like 
                      'post_attention_layernorm'. On Llama and Qwen, that name exists 
                      but represents the PRE-norm of the MLP: a naming trap that 
                      causes the gate to fail at 0.94.

  bias on o_proj      GPT-2 and others. The sum of per-head contributions is NOT 
                      the output of o_proj: the bias is missing, which is a single 
                      term that cannot be attributed to any individual head. It is 
                      tracked separately, much like the norm term in the sandwich split. 
                      It cancels out in intra-pair differences, but the gate must 
                      be aware of it.

  parallel blocks     GPT-J, Falcon, PaLM. Attention and FFN BOTH read the exact 
                      same normalized state concurrently instead of sequentially. 
                      The additive identity still holds, meaning the decomposition 
                      remains valid, but a pre-FFN state DOES NOT exist: provenance 
                      checks with the pre-frame there measure a different property 
                      and must be re-interpreted.

  true post-norm      Normalization is applied to the SUM: h = norm(h + attn(h)). 
                      The residual stream is non-additive, and decomposition 
                      does not exist. This must be rejected, not patched. The gate 
                      will intercept it regardless.

  MoE                 The block remains additive, so block-level decomposition 
                      holds without modifications. The variance occurs INSIDE the 
                      FFN, which becomes a weighted sum of experts selected by a 
                      router: per-expert decomposition is a novel object entirely, 
                      not a variant of this module.

  deviant head_dim    On Gemma, head_dim multiplied by n_heads does not equal 
                      hidden_size. The head dimension must always be inferred 
                      from o_proj.in_features, never by dividing hidden_size.
"""


from dataclasses import dataclass, asdict
from typing import Optional, List

import torch


# =====================================================================
#  descrizione dell'architettura
# =====================================================================
@dataclass
class Wiring:
        """
    WHERE to look inside a block. Completely decoupled from WHAT to do with what 
    is captured, which is handled by the rest of the library and never changes.

    This is necessary because models call the same components by different names, 
    and no automatic detection can cover an architecture that did not exist when 
    the library was written. Instead of updating the library for every new case, 
    the wiring is passed from the outside:

        w = Wiring(
            attn_write = lambda L: L.post_attention_layernorm,  # what enters the residual stream
            ffn_write  = lambda L: L.post_feedforward_layernorm,
            out_proj   = lambda L: L.self_attn.o_proj,
            pre_norm_in= lambda L: L.post_attention_layernorm,  # sandwich norm only
        )
        arch = describe(model, wiring=w)

    Each field is a function that receives the block and returns the specific module 
    to be hooked. Anything not required remains None.

    The identity gate verifies the output regardless: if the wiring is incorrect, 
    the relative error explodes and execution halts instead of silently producing 
    meaningless numbers. Because of this, manual wiring is entirely safe: it is 
    treated as a proposal that gets verified.
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
        d["wiring"] = ("explicit: " + (self.wiring.notes or "provided by the user")
                       if self.wiring else "automatically detected")
        return d

    def summary(self):
        r = ["%s: %d blocchi, d=%d, %d teste da %d"
             % (self.family, self.n_blocks, self.d_model, self.n_heads, self.head_dim)]
        if self.residual_multiplier != 1.0:
            r.append("residual multiplier %.4f applied to contributions"
                     % self.residual_multiplier)
        if self.sandwich:
            r.append("output norms (%s): need to hook post-norma, "
                     "gain %s" % (self.norm_style,
                                      "(1+w)" if self.unit_offset else "w"))
        if self.o_proj_bias:
            r.append("o_proj ha bias: la somma per testa lo esclude, riportato a parte")
        if self.parallel:
            r.append("parallel blocks: attention and FFN read the same "
                     "state and write Together. There is no pre-FFN state, and "
                     "FFN does not see the attention of its own block.")
        if self.moe:
            r.append("MoE cwith %s experts, %s active per token: the block remain"
                     "additive, FFN no"
                     % (self.n_experts or "?", self.top_k or "?"))
        if self.wiring:
            r.append("explicit wiring%s"
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
    raise RuntimeError("not find the decoder blocks on this model.")


def _attn(layer):
    for attr in ("self_attn", "attn", "attention"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError("not find the attention-module in the block.")


def _out_proj(attn):
    for attr in ("o_proj", "out_proj", "dense", "c_proj", "wo"):
        m = getattr(attn, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m
    raise RuntimeError("not find the attention output projection.")


def _mlp(layer):
    for attr in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError("not found MLP into the block")


def describe(model, wiring=None):
"""Detects the architectural variant. Detection is explicit and conservative: 
    what it does not recognize, it declares instead of guessing."""
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
        notes.append("head_dim per n_heads (%d) does not equal d_model (%d): normal on "
                     "Gemma, the dimension comes from o_proj" % (head_dim * n_heads, d_model))

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
        notes.append("output-only norms (OLMo-2 style): the two post-norm "
                     "write into the residual stream and the FFN reads the raw residual. "
                     "The hook point is the sandwich one, but the pre-frame is not: here "
                     "there is NO normalized state entering into the FFN")
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
        notes.append("residual multiplier %.4f: contribution are scaled "
                     "before entering the residual stream, and the library applies "
                     "this scaling to the returned vectors" % rmul)

    nrm = getattr(layer, "post_attention_layernorm", None)
    unit_offset = nrm is not None and "gemma" in type(nrm).__name__.lower()
    if sandwich and not unit_offset:
        notes.append("la post-norma multiply by W, not by (1+w)")

    moe = any(k in type(mlp).__name__.lower() for k in ("moe", "sparse", "expert"))
    packed = getattr(mlp, "experts", None)
    if packed is not None and hasattr(packed, "gate_up_proj"):
        moe = True
        notes.append("experts packaged as 3D tensors, not as modules: "
                     "The per-expert decomposition is analytical, just like the per-head decomposition.")
    n_exp = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    top_k = (getattr(cfg, "num_experts_per_tok", None)
             or getattr(cfg, "moe_top_k", None)
             or getattr(cfg, "num_selected_experts", None))
    if moe and not (packed is not None and hasattr(packed, "gate_up_proj")):
        notes.append("expert like modules: the decomposition per expert it done "
                     "with hook, 1 per expert")

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
        notes.append("with parallel blocks the pre frame its not pre-FFN state")

    fam = getattr(cfg, "model_type", type(model).__name__)
    if wiring is not None:
        notes.append("Explicit wiring takes priority over detection.")
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
    """
    Captures, for a given block, the vectors added by attention and FFN to the 
    residual stream, along with the per-head decomposition of the attention contribution.

    Use as a context manager:

        with BlockCapture(model, arch, block) as cap:
            out = model(**enc, output_hidden_states=True)
            a, f = cap.attn(), cap.ffn()
            per_head = cap.heads()         # [B, nH, d]

    All extractions occur during the same forward pass: the per-head decomposition 
    does not require an additional pass, since the model computes the head outputs 
    anyway, and this context manager simply prevents them from being discarded.
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
    """A block weight, forced onto the CPU and cast to float32, even if it is offloaded.

    This is the only point where the library directly reads a model parameter. If the 
    tensor is a placeholder, it attempts to use the weight map from `accelerate`; if 
    even that map is missing, it explicitly raises an error instead of returning zeros, 
    which would otherwise masquerade as a legitimate measurement.
    """

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
    """The gain of an RMSNorm layer, calculated with the correct convention.

    Gemma stores the parameter as a deviation and multiplies by (1 + w). 
    Llama, Qwen, and OLMo multiply directly by w. Applying the incorrect convention 
    shifts the per-head decomposition by an artifact amount that looks like a valid result. 
    The distinction is detected in `describe()` and verified by the per-head gate.
    """
        g = self._w(norm, name)
        if g is None:
            return None
        return (1.0 + g) if getattr(self.arch, "unit_offset", False) else g

    def _at(self, t):
        return t[:, self.pos, :].detach().float().cpu()

    def _res(self, t):
    """WHAT ENTERS the residual stream, not what the module outputs.

    Where a residual multiplier exists, the two differ by a factor. It is applied 
    here and not inside `_at`, because `_at` is also used to read the INPUT state 
    to a norm, which is unaffected by the multiplier. Scaling that state would break 
    the per-head RMS calculation.
    """

        m = getattr(self.arch, "residual_multiplier", 1.0)
        return t if m == 1.0 else t * m

    def attn(self):
        return self._res(self._at(self.buf["a"]))

    def ffn(self):
        return self._res(self._at(self.buf["f"]))

    def heads(self, which=None):
    """Contribution of each individual head to the residual stream: [B, nH, d].

    `o_proj` is linear, so the contribution of head `h` is its slice of `z` projected 
    with its corresponding slice of `W_o`. In sandwich architectures, the post-norm 
    is linear AT A FIXED SENTENCE level because `rms(x)` is a scalar: the same factor 
    applies to each slice and the sum reconstructs exactly. 
    The bias of `o_proj`, if present, is NOT attributable to any head and is returned 
    separately via `bias_term()`.

    DEVICES. Activations are moved to the CPU by `_at`, whereas WEIGHTS remain 
    where the model is allocated, typically on the GPU. Every weight tensor read 
    here must therefore be explicitly moved, otherwise the operation encounters 
    operands across different devices. The computation is performed on the CPU 
    because it is tiny and because the final result is required by the CPU anyway.
    """

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
    """The bias of o_proj, already passed through the post-norm layer if present.
    It does not belong to any individual head. It cancels out in intra-pair 
    differences; however, it must be accounted for in the identity gate.
    """
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
    """MoE: which experts were chosen at the token readout, and with what weight.

    Requires a wiring setup that includes a router. Router outputs vary across
    implementations: this function accepts raw logits [B, S, E] or a tuple
    where the first element is extracted. Weights are computed via softmax over top_k,
    which is the most common form; if your model uses a different normalization,
    read 'logits' and recompute as needed.

    ATTENTION, this is the critical point for the experimental design: the router
    routes PER TOKEN, not per sentence. In a minimal pair, the two sentences
    differ by a single token, so the real question is whether that specific token
    changes the activated experts at the readout point.
    """

    if "route" not in self.buf:
        raise RuntimeError("no router hooked: need a Wiring with router")
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
    """Verifies that a + f reconstructs h[b+1] - h[b].

    Returns (median of the relative error, outcome). If the outcome is False, 
    downstream numerical values MUST NOT be read: the decomposition does not describe the model.
    Typical causes of failure, in order of frequency: hook attached to the module 
    instead of the post-norm in sandwich architectures, use of bfloat16, 
    or true post-norm architecture where decomposition does not exist.
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
        msg = "additive identity failed (median %.2e)." % med
        if dtype_used != "float32":
            msg += " The dtype used is %s: the gate strictly requires float32." % dtype_used
        else:
            msg += (" Verify the hook point: on sandwich architectures "
                    "the post-norm must be hooked, not the module.")
    return med, ok, msg


def head_gate(per_head, a_captured, bias=None, tol=1e-4):
    """The sum of per-head contributions must reconstruct the independently 
    captured attention contribution, plus the bias if present."""
    s = per_head.detach().sum(1)
    if bias is not None:
        s = s + bias.detach()
    ac = a_captured.detach()
    err = (s - ac).norm(dim=-1) / ac.norm(dim=-1).clamp_min(1e-8)
    med = float(err.median())
    return med, med < tol
