"""
fada_tf.py (2026-11-07, runs/screen/FADA_AUDIT_1107.md, user: "option a full fidelity"): FADA's planner and inverse dynamics model as the
paper specifies (Xie et al., arXiv 2606.28476v2, Sec 4, Eq. 4.1-4.6, Appendix B.1-B.3), in JAX (training) and numpy (step-by-step
rollouts, identical numerics), shared by the car and the quad.

Specified by the paper (followed):
  P_phi(O_t^H, c_t) -> Y_hat_t^K = o_t + DeltaY (Eq. B.1: residual future chunk); I_psi(O_t^H, A_t^H, Y^K) -> U^K (K-step action chunk);
  executed a_t = Pi_1(U^K), receding horizon; H = 30, K = 6; planner and IDM encoders: 3-layer, 4-head transformers, hidden 128; IDM
  decoder: 2 transformer decoder layers (same size), future tokens with FULL (non-causal) self-attention over the future chunk and
  cross-attention to all encoded history tokens; history tokens = obs embedding + executed-action embedding + positional encoding; IDM loss
  = first-action MSE (Eq. 4.2 / 4.5); planner loss through the stop-gradient IDM to the oracle's first action (Eq. 4.3); LoRA r 8,
  alpha 16, dropout 0.05 on the IDM only, planner and base IDM frozen (B.3).
Not specified by the paper (declared choices):
  pre-LayerNorm blocks, GELU feed-forward of width 4 x 128, learned positional embeddings, a final LayerNorm per stack; the planner's
  command c_t as one extra token (embedded) prepended to its history tokens and its head on the latest history token (o_t's); the
  planner's history tokens are obs only (the paper: P(O_t^H, c_t), no actions); history shorter than H at an episode's start is padded and
  masked out of attention (keys); LoRA on every Linear inside the IDM (embeddings, attention q/k/v/o, feed-forward, head) -- "inserted into
  the IDM" (layers unspecified).
"""
import numpy as np

D_MODEL, N_HEADS, D_FF = 128, 4, 512
H_HIST, K_FUT = 30, 6
N_ENC, N_DEC = 3, 2
LORA_R, LORA_ALPHA, LORA_DROP = 8, 16, 0.05


# ------------------------------------------------------------------ parameter construction (numpy; JAX trains the same pytree)
def _lin(rng, i, o, scale=None):
    s = (1.0 / i) ** 0.5 if scale is None else scale
    return dict(W=(rng.standard_normal((i, o)) * s).astype(np.float32), b=np.zeros(o, np.float32))


def _ln(d):
    return dict(g=np.ones(d, np.float32), b=np.zeros(d, np.float32))


def _attn(rng, d):
    return dict(q=_lin(rng, d, d), k=_lin(rng, d, d), v=_lin(rng, d, d), o=_lin(rng, d, d))


def _enc_block(rng, d):
    return dict(ln1=_ln(d), att=_attn(rng, d), ln2=_ln(d), ff1=_lin(rng, d, D_FF), ff2=_lin(rng, D_FF, d))


def _dec_block(rng, d):
    return dict(ln1=_ln(d), self_att=_attn(rng, d), ln2=_ln(d), cross_att=_attn(rng, d), ln3=_ln(d), ff1=_lin(rng, d, D_FF),
                ff2=_lin(rng, D_FF, d))


def init_planner(seed, d_o, d_c, H=H_HIST, K=K_FUT):
    rng = np.random.default_rng(seed)
    return dict(emb_o=_lin(rng, d_o, D_MODEL), emb_c=_lin(rng, d_c, D_MODEL),
                pos=(rng.standard_normal((H + 1, D_MODEL)) * 0.02).astype(np.float32),
                enc=[_enc_block(rng, D_MODEL) for _ in range(N_ENC)], lnf=_ln(D_MODEL),
                head=_lin(rng, D_MODEL, K * d_o, scale=0.01 / D_MODEL ** 0.5))


def init_idm(seed, d_o, d_a, H=H_HIST, K=K_FUT):
    rng = np.random.default_rng(seed)
    return dict(emb_o=_lin(rng, d_o, D_MODEL), emb_a=_lin(rng, d_a, D_MODEL), emb_y=_lin(rng, d_o, D_MODEL),
                pos_h=(rng.standard_normal((H, D_MODEL)) * 0.02).astype(np.float32),
                pos_f=(rng.standard_normal((K, D_MODEL)) * 0.02).astype(np.float32),
                enc=[_enc_block(rng, D_MODEL) for _ in range(N_ENC)], lnf_e=_ln(D_MODEL),
                dec=[_dec_block(rng, D_MODEL) for _ in range(N_DEC)], lnf_d=_ln(D_MODEL),
                head=_lin(rng, D_MODEL, d_a, scale=0.01 / D_MODEL ** 0.5))


def init_lora(seed, idm, r=LORA_R):
    """LoRA factors for every Linear (dict with W, b) inside the IDM: A ~ N(0, 1/in) (in x r), B = 0 (r x out) -> the merged IDM == the
    base at initialization."""
    rng = np.random.default_rng(seed)

    def rec(p):
        if isinstance(p, dict) and 'W' in p and 'b' in p:
            i, o = p['W'].shape
            return dict(A=(rng.standard_normal((i, r)) / np.sqrt(i)).astype(np.float32), B=np.zeros((r, o), np.float32))
        if isinstance(p, dict):
            return {k: rec(v) for k, v in p.items() if not isinstance(v, np.ndarray) or k in ()}
        if isinstance(p, list):
            return [rec(v) for v in p]
        return None
    return rec(idm)


def merge_lora(idm, lora, alpha=LORA_ALPHA, r=LORA_R, xp=np):
    """W + (alpha / r) A B for every Linear (the deployed, merged IDM)."""
    def rec(p, q):
        if isinstance(p, dict) and 'W' in p and 'b' in p:
            return dict(W=p['W'] + (alpha / r) * (q['A'] @ q['B']), b=p['b'])
        if isinstance(p, dict):
            return {k: (rec(v, q[k]) if k in q else v) for k, v in p.items()}
        if isinstance(p, list):
            return [rec(v, w) for v, w in zip(p, q)]
        return p
    return rec(idm, lora)


# ------------------------------------------------------------------ the forward pass, written once for numpy or jax.numpy (xp)
def _linear(p, x, xp, lora=None, drop_key=None, jax=None, alpha=LORA_ALPHA, r=LORA_R):
    y = x @ p['W'] + p['b']
    if lora is not None:
        xin = x
        if drop_key is not None:                       # LoRA dropout on the adapter's input (training only)
            keep = jax.random.bernoulli(drop_key, 1.0 - LORA_DROP, x.shape)
            xin = xp.where(keep, x / (1.0 - LORA_DROP), 0.0)
        y = y + (alpha / r) * ((xin @ lora['A']) @ lora['B'])
    return y


def _layernorm(p, x, xp):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / xp.sqrt(v + 1e-5) * p['g'] + p['b']


def _gelu(x, xp):
    return 0.5 * x * (1.0 + xp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class _Ctx:
    """Threads the optional LoRA tree and dropout keys through one forward pass."""
    def __init__(self, xp, lora=None, key=None, jax=None):
        self.xp, self.jax, self.key = xp, jax, key

    def k(self):
        if self.key is None:
            return None
        self.key, sub = self.jax.random.split(self.key)
        return sub


def _mha(p, xq, xkv, kmask, ctx, lp=None):
    """Multi-head attention; xq (..., Lq, d), xkv (..., Lk, d); kmask (..., Lk) bool (True = attend) or None. Full (non-causal)."""
    xp = ctx.xp
    L = lambda name, x: _linear(p[name], x, xp, None if lp is None else lp[name], ctx.k() if lp is not None else None, ctx.jax)
    q, k, v = L('q', xq), L('k', xkv), L('v', xkv)
    sh = lambda t: t.reshape(t.shape[:-1] + (N_HEADS, D_MODEL // N_HEADS))
    q, k, v = sh(q), sh(k), sh(v)                       # (..., L, h, dh)
    s = xp.einsum('...qhd,...khd->...hqk', q, k) / np.sqrt(D_MODEL // N_HEADS)
    if kmask is not None:
        s = xp.where(kmask[..., None, None, :], s, -1e9)
    s = s - s.max(-1, keepdims=True)
    w = xp.exp(s)
    w = w / w.sum(-1, keepdims=True)
    o = xp.einsum('...hqk,...khd->...qhd', w, v).reshape(xq.shape[:-1] + (D_MODEL,))
    return L('o', o)


def _enc(blocks, x, kmask, ctx, lblocks=None):
    xp = ctx.xp
    for i, b in enumerate(blocks):
        lb = None if lblocks is None else lblocks[i]
        h = _layernorm(b['ln1'], x, xp)
        x = x + _mha(b['att'], h, h, kmask, ctx, None if lb is None else lb['att'])
        h = _layernorm(b['ln2'], x, xp)
        f = _gelu(_linear(b['ff1'], h, xp, None if lb is None else lb['ff1'], ctx.k() if lb is not None else None, ctx.jax), xp)
        x = x + _linear(b['ff2'], f, xp, None if lb is None else lb['ff2'], ctx.k() if lb is not None else None, ctx.jax)
    return x


# (2026-11-07) the MLP ablation (the paper's Table 8: "replacing the attention-based modules with MLP counterparts"): the same inputs
# (the masked H-step history flattened, the command, the future chunk), the same outputs (o_t + DeltaY; the K-action chunk), 3 hidden
# layers of 512 (GELU) -- ~0.8 M parameters each, as the transformers' ~0.6 / 1.1 M. Selected by the parameter tree ('mlp' key).
MLP_W = 512


def _mlp_init(rng, i, o, head_scale):
    sizes = [i, MLP_W, MLP_W, MLP_W]
    L = [_lin(rng, a_, b_) for a_, b_ in zip(sizes[:-1], sizes[1:])]
    return L + [_lin(rng, MLP_W, o, scale=head_scale)]


def init_planner_mlp(seed, d_o, d_c, H=H_HIST, K=K_FUT):
    rng = np.random.default_rng(seed)
    return dict(mlp=_mlp_init(rng, H * d_o + d_c, K * d_o, 0.01 / MLP_W ** 0.5))


def init_idm_mlp(seed, d_o, d_a, H=H_HIST, K=K_FUT):
    rng = np.random.default_rng(seed)
    return dict(mlp=_mlp_init(rng, H * d_o + H * d_a + K * d_o, K * d_a, 0.01 / MLP_W ** 0.5))


def _mlp_fwd(L, x, xp):
    for l in L[:-1]:
        x = _gelu(_linear(l, x, xp), xp)
    return _linear(L[-1], x, xp)


def planner_fwd(P, O, C, hmask, xp=np):
    """O (..., H, d_o) oldest -> newest (o_t last), C (..., d_c) the command, hmask (..., H) bool valid history.
    Returns Y_hat (..., K, d_o) = o_t + DeltaY."""
    if 'mlp' in P:
        H, d_o = O.shape[-2], O.shape[-1]
        x = xp.concatenate([(O * hmask[..., None]).reshape(O.shape[:-2] + (H * d_o,)), C], -1)
        dY = _mlp_fwd(P['mlp'], x, xp)
        return O[..., -1:, :] + dY.reshape(dY.shape[:-1] + (dY.shape[-1] // d_o, d_o))
    ctx = _Ctx(xp)
    H, d_o = O.shape[-2], O.shape[-1]
    tok = _linear(P['emb_o'], O, xp) + P['pos'][1:H + 1]
    ctok = (_linear(P['emb_c'], C, xp) + P['pos'][0])[..., None, :]
    x = xp.concatenate([ctok, tok], -2)
    m = xp.concatenate([xp.ones(hmask.shape[:-1] + (1,), bool), hmask], -1)
    x = _layernorm(P['lnf'], _enc(P['enc'], x, m, ctx), xp)
    dY = _linear(P['head'], x[..., -1, :], xp)          # the latest history token (o_t's)
    K = dY.shape[-1] // d_o
    return O[..., -1:, :] + dY.reshape(dY.shape[:-1] + (K, d_o))


def idm_fwd(I, O, A, Y, hmask, xp=np, lora=None, key=None, jax=None):
    """O (..., H, d_o), A (..., H, d_a) the executed actions BEFORE each history obs (a_{t-H} .. a_{t-1}), Y (..., K, d_o) the future chunk,
    hmask (..., H). Returns U (..., K, d_a); the executed action is U[..., 0, :]."""
    if 'mlp' in I:
        H, K, d_a = O.shape[-2], Y.shape[-2], A.shape[-1]
        m = hmask[..., None]
        x = xp.concatenate([(O * m).reshape(O.shape[:-2] + (-1,)), (A * m).reshape(A.shape[:-2] + (-1,)), Y.reshape(Y.shape[:-2] + (-1,))], -1)
        U = _mlp_fwd(I['mlp'], x, xp)
        return U.reshape(U.shape[:-1] + (U.shape[-1] // d_a, d_a))
    ctx = _Ctx(xp, lora, key, jax)
    l = (lambda name: None) if lora is None else (lambda name: lora[name])
    kk = (lambda: ctx.k()) if lora is not None else (lambda: None)
    H, K = O.shape[-2], Y.shape[-2]
    h = _linear(I['emb_o'], O, xp, l('emb_o'), kk(), jax) + _linear(I['emb_a'], A, xp, l('emb_a'), kk(), jax) + I['pos_h'][:H]
    mem = _layernorm(I['lnf_e'], _enc(I['enc'], h, hmask, ctx, None if lora is None else lora['enc']), xp)
    y = _linear(I['emb_y'], Y, xp, l('emb_y'), kk(), jax) + I['pos_f'][:K]
    for i, b in enumerate(I['dec']):
        lb = None if lora is None else lora['dec'][i]
        z = _layernorm(b['ln1'], y, xp)
        y = y + _mha(b['self_att'], z, z, None, ctx, None if lb is None else lb['self_att'])        # full self-attention (no causal mask)
        z = _layernorm(b['ln2'], y, xp)
        y = y + _mha(b['cross_att'], z, mem, hmask, ctx, None if lb is None else lb['cross_att'])
        z = _layernorm(b['ln3'], y, xp)
        f = _gelu(_linear(b['ff1'], z, xp, None if lb is None else lb['ff1'], kk() if lb is not None else None, jax), xp)
        y = y + _linear(b['ff2'], f, xp, None if lb is None else lb['ff2'], kk() if lb is not None else None, jax)
    y = _layernorm(I['lnf_d'], y, xp)
    return _linear(I['head'], y, xp, l('head'), kk(), jax)


def d_obs(I, d_a):
    """(2026-11-07 LEAK FIX) the IDM's observation width d_o, from its weights. Models trained before the fix have o_t = [state,
    a_{t-1}] (car 15, quad 11): Y = o_{t+1..t+K} then CONTAINS a_t (o_{t+1}'s previous-action slot), the IDM's own target -- it copies the
    action out of Y (target-data MSE 3e-5 vs 8e-3 for copying a_{t-1}) and learns no dynamics. Fixed models: o_t = the state only (car 13,
    quad 7); the executed actions enter only as the IDM's action tokens A^H (the paper's I(O^H, A^H, Y))."""
    if 'mlp' in I:
        n_in = I['mlp'][0]['W'].shape[0]
        return (n_in - H_HIST * d_a) // (H_HIST + K_FUT)
    return I['emb_y']['W'].shape[0]


def h_hist(I):
    """(2026-11-07) the history length H the IDM was built with (transformer: its history positions; MLP models are H_HIST)."""
    return I['pos_h'].shape[0] if 'pos_h' in I else H_HIST


# ------------------------------------------------------------------ the deployed policy, one control step at a time (numpy)
class FadaPolicy:
    """Stateful receding-horizon FADA: keeps o and executed-action histories; act(o_t, c_t) -> a_t. reset() at each episode start.
    The executed action fed back (observe_action) is the one actually applied (after any external noise / clipping by the caller)."""
    def __init__(self, P, I, d_a, H=None):
        self.P, self.I, self.H, self.d_a = P, I, (h_hist(I) if H is None else H), d_a
        self.d_o = d_obs(I, d_a)
        self.reset()

    def reset(self):
        self.O, self.A, self.prev_a = [], [], None

    def act(self, o, c):
        o = np.asarray(o, np.float32)
        self.A.append(np.zeros(self.d_a, np.float32) if self.prev_a is None else self.prev_a)
        self.O.append(o)
        self.O, self.A = self.O[-self.H:], self.A[-self.H:]
        n = len(self.O)
        O = np.zeros((self.H, len(o)), np.float32); A = np.zeros((self.H, self.d_a), np.float32); m = np.zeros(self.H, bool)
        O[self.H - n:], A[self.H - n:], m[self.H - n:] = np.stack(self.O), np.stack(self.A), True
        if _use_jit():                                     # (2026-11-07) the same forward, XLA-compiled (FADA_JIT=1; test 9 of fada_tf_tests.py)
            return np.asarray(_jit_step()(self._jP(), self._jI(), O, A, np.asarray(c, np.float32), m))
        Y = planner_fwd(self.P, O, np.asarray(c, np.float32), m)
        U = idm_fwd(self.I, O, A, Y, m)
        return U[0]

    def _jP(self):
        if getattr(self, '_jp', None) is None:
            import jax.numpy as jnp
            self._jp = _tree(self.P, jnp.asarray)
        return self._jp

    def _jI(self):
        if getattr(self, '_ji', None) is None:
            import jax.numpy as jnp
            self._ji = _tree(self.I, jnp.asarray)
        return self._ji

    def observe_action(self, a):
        self.prev_a = np.asarray(a, np.float32)


_JIT = None


def _use_jit():
    import os
    return os.environ.get('FADA_JIT', '0') == '1'


def _tree(t, f):
    if isinstance(t, dict):
        return {k: _tree(v, f) for k, v in t.items()}
    if isinstance(t, list):
        return [_tree(v, f) for v in t]
    return f(t)


def _jit_step():
    """One receding-horizon step (planner then IDM, first action) compiled once per process."""
    global _JIT
    if _JIT is None:
        import jax
        import jax.numpy as jnp

        def f(P, I, O, A, c, m):
            Y = planner_fwd(P, O, c, m, xp=jnp)
            return idm_fwd(I, O, A, Y, m, xp=jnp)[0]
        _JIT = jax.jit(f)
    return _JIT


def to_np(tree):
    if isinstance(tree, dict):
        return {k: to_np(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [to_np(v) for v in tree]
    return np.asarray(tree, np.float32)


def save(path, P, I, meta):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(dict(P=to_np(P), I=to_np(I), meta=meta), f)


def load(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


# (2026-11-07) a version-independent flat form (plain named arrays, no pickle: the quad robots' container runs numpy 1.21)
def flatten(tree, prefix):
    out = {}

    def rec(t, path):
        if isinstance(t, dict):
            for k, v in t.items():
                rec(v, path + [str(k)])
        elif isinstance(t, (list, tuple)):
            for i, v in enumerate(t):
                rec(v, path + [f"#{i}"])
        else:
            out[prefix + "__" + "__".join(path)] = np.asarray(t, np.float32)
    rec(tree, [])
    return out


def unflatten(d, prefix):
    root = {}
    for key, v in d.items():
        if not key.startswith(prefix + "__"):
            continue
        path = key[len(prefix) + 2:].split("__")
        node = root
        for p, nxt in zip(path[:-1], path[1:]):
            node = node.setdefault(p, {})
        node[path[-1]] = np.asarray(v, np.float32)

    def fix(t):                                  # dicts whose keys are all "#i" back to lists
        if isinstance(t, dict):
            t = {k: fix(v) for k, v in t.items()}
            if t and all(k.startswith("#") for k in t):
                return [t[f"#{i}"] for i in range(len(t))]
        return t
    return fix(root)
