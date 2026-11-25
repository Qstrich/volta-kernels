"""Reference, stock and deliberately broken implementations of one interface.

attn(q, k, v, P, lens=None, sq=None, sk=None) -> out
  q, k, v: [B, L, H, 64] FP16. Token i is in frame i // P. Query i sees key j iff j//P <= i//P and j < lens[b].
  q and k are RMS-normalized over the last dim in FP32 (eps 2^-23), times optional FP32 per-dim scales sq, sk [64].
  logits = q.k / 8, softmax in FP32. Rows i >= lens[b] are exactly 0.
"""
import importlib, math, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend as S
EPS = 2.0 ** -23

def rms(x, s):
    x = x / (x.square().mean(-1, keepdim=True) + EPS).sqrt()
    return x if s is None else x * s.to(x.dtype)

def vis_mask(L, P, lens, dev):
    f = torch.arange(L, device=dev) // P
    m = (f[:, None] >= f[None, :])[None, None]
    if lens is not None:
        ok = torch.arange(L, device=dev)[None] < lens[:, None]
        m = (m & ok[:, None, None, :]) | ~ok[:, None, :, None]     # padded rows see everything, zeroed below
    return m

def zero_pad(o, lens):
    if lens is None: return o
    ok = torch.arange(o.shape[1], device=o.device)[None] < lens[:, None]
    return o * ok[:, :, None, None].to(o.dtype)

def reference(q, k, v, P, lens=None, sq=None, sk=None, dtype=torch.float32):
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    q, k = rms(q, sq).transpose(1, 2), rms(k, sk).transpose(1, 2)
    a = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    a = a.masked_fill(~vis_mask(q.shape[2], P, lens, q.device), float("-inf")).softmax(-1)
    return zero_pad((a @ v.transpose(1, 2)).transpose(1, 2), lens)

def _stock(q, k, v, P, lens, sq, sk, mask_fn=vis_mask, norm=rms):
    qn, kn = (norm(x.float(), s).to(q.dtype).transpose(1, 2) for x, s in ((q, sq), (k, sk)))
    m = mask_fn(q.shape[1], P, lens, q.device)
    with sdpa_kernel([S.EFFICIENT_ATTENTION]):
        return zero_pad(F.scaled_dot_product_attention(qn, kn, v.transpose(1, 2), attn_mask=m).transpose(1, 2), lens)

def stock(q, k, v, P, lens=None, sq=None, sk=None): return _stock(q, k, v, P, lens, sq, sk)

# --- deliberately broken: the suite must fail every one of these ---
def bad_token_causal(q, k, v, P, lens=None, sq=None, sk=None): return _stock(q, k, v, 1, lens, sq, sk)
def bad_no_mask(q, k, v, P, lens=None, sq=None, sk=None): return _stock(q, k, v, 10**9, lens, sq, sk)
def bad_strict(q, k, v, P, lens=None, sq=None, sk=None):
    def m(L, P, lens, dev):
        f = torch.arange(L, device=dev) // P; t = vis_mask(L, P, lens, dev)
        return t & ((f[:, None] > f[None, :]) | (f[:, None] == 0) | (f[:, None] == f[None, :]) & (torch.arange(L, device=dev)[None] % P != 0))
    return _stock(q, k, v, P, lens, sq, sk, mask_fn=m)
def bad_edge(q, k, v, P, lens=None, sq=None, sk=None):          # one extra key visible past each frame boundary
    def m(L, P, lens, dev):
        f = torch.arange(L, device=dev) // P; j = torch.arange(L, device=dev)
        return vis_mask(L, P, lens, dev) | ((j[None] == (f[:, None] + 1) * P) & (j[None] < L))[None, None]
    return _stock(q, k, v, P, lens, sq, sk, mask_fn=m)
def bad_ignores_lens(q, k, v, P, lens=None, sq=None, sk=None):
    return zero_pad(_stock(q, k, v, P, None, sq, sk), lens)
def bad_no_padzero(q, k, v, P, lens=None, sq=None, sk=None):
    qn, kn = (rms(x.float(), s).to(q.dtype).transpose(1, 2) for x, s in ((q, sq), (k, sk)))
    with sdpa_kernel([S.EFFICIENT_ATTENTION]):
        return F.scaled_dot_product_attention(qn, kn, v.transpose(1, 2), attn_mask=vis_mask(q.shape[1], P, lens, q.device)).transpose(1, 2)
def bad_no_qknorm(q, k, v, P, lens=None, sq=None, sk=None):
    return _stock(q, k, v, P, lens, sq, sk, norm=lambda x, s: x if s is None else x * s)
def bad_fp16_norm(q, k, v, P, lens=None, sq=None, sk=None):      # squares overflow / underflow in FP16
    def n(x, s):
        x = x.half(); x = x / (x * x).mean(-1, keepdim=True).add(EPS).sqrt()
        return x.float() if s is None else x.float() * s
    return _stock(q, k, v, P, lens, sq, sk, norm=n)
def bad_ignores_scale(q, k, v, P, lens=None, sq=None, sk=None):
    return _stock(q, k, v, P, lens, sq, sk, norm=lambda x, s: rms(x, None if s is None else s * 0 + 1))
def bad_eps(q, k, v, P, lens=None, sq=None, sk=None):
    return _stock(q, k, v, P, lens, sq, sk, norm=lambda x, s: (x / (x.square().mean(-1, keepdim=True) + 1e-3).sqrt()) * (1 if s is None else s))
def bad_nan_to_num(q, k, v, P, lens=None, sq=None, sk=None): return _stock(q, k, v, P, lens, sq, sk).nan_to_num()
def bad_scale(q, k, v, P, lens=None, sq=None, sk=None):          # wrong logit scale by 1%
    return _stock(q, k, v, P, lens, sq, sk, norm=lambda x, s: rms(x, s) * 1.005)

class _Fudge(torch.autograd.Function):                           # identity forward, gradient x 1.01
    @staticmethod
    def forward(c, x): return x.clone()
    @staticmethod
    def backward(c, g): return g * 1.01
def bad_grad_detached_norm(q, k, v, P, lens=None, sq=None, sk=None):
    return _stock(q, k, v, P, lens, sq, sk, norm=lambda x, s: (x / (x.square().mean(-1, keepdim=True) + EPS).sqrt().detach()) * (1 if s is None else s))
def bad_grad_v(q, k, v, P, lens=None, sq=None, sk=None): return _stock(q, k, _Fudge.apply(v), P, lens, sq, sk)
def bad_grad_sk(q, k, v, P, lens=None, sq=None, sk=None):
    return _stock(q, k, v, P, lens, sq, None if sk is None else _Fudge.apply(sk))

def get(name):
    if name == "kernel": return importlib.import_module("volta_attn").attn
    return globals()[name]
MUTANTS = [n for n in list(globals()) if n.startswith("bad_")]
