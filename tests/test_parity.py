"""Parity suite for kernel 1. Thresholds and cases are read-only to the optimisation loop.
VOLTA_IMPL=stock|kernel|bad_*   (marker 'grad': needs backward)"""
import os, pytest, torch, impls
IMPL = os.environ.get("VOLTA_IMPL", "kernel"); attn = impls.get(IMPL)
TOL, GTOL, D = 1e-3, 2e-3, 64            # TOL: max abs err, or max abs err / max|V| when |V| > 1

def data(B, L, H, seed=0, qs=1.0, vmax=1.0):
    g = torch.Generator("cuda").manual_seed(seed)
    r = lambda: torch.randn(B, L, H, D, device="cuda", generator=g)
    v = (torch.rand(B, L, H, D, device="cuda", generator=g) * 2 - 1) * vmax
    return (r() * qs).half(), (r() * qs).half(), v.half()
def scales(seed=1):
    g = torch.Generator("cuda").manual_seed(seed)
    return tuple(torch.rand(D, device="cuda", generator=g) + 0.5 for _ in range(2))
def check(q, k, v, P, lens=None, sq=None, sk=None, tol=TOL):
    with torch.no_grad():
        o = attn(q, k, v, P, lens, sq, sk); r = impls.reference(q, k, v, P, lens, sq, sk)
    assert o.dtype == torch.float16 and o.shape == v.shape
    assert torch.isfinite(o).all()
    err = (o.float() - r).abs().max().item() / max(1.0, v.abs().max().item())
    assert err < tol, f"err {err:.2e}"
    if lens is not None:
        for b, n in enumerate(lens.tolist()): assert (o[b, n:] == 0).all()
    return o

@pytest.mark.parametrize("T", [1, 2, 4, 8])
@pytest.mark.parametrize("B,H", [(1, 6), (4, 6), (2, 16)])
@pytest.mark.parametrize("scaled", [False, True])
def test_project_shapes(T, B, H, scaled):
    check(*data(B, 196 * T, H), 196, None, *(scales() if scaled else (None, None)))

@pytest.mark.parametrize("L,P", [(1, 1), (2, 1), (63, 7), (64, 64), (65, 64), (127, 13), (128, 1), (129, 129), (197, 197),
                                 (200, 196), (391, 196), (393, 196), (777, 111), (1000, 333), (50, 1000), (1569, 196)])
@pytest.mark.parametrize("B,H", [(1, 1), (3, 5)])
def test_odd_sizes(L, P, B, H): check(*data(B, L, H, seed=L), P)

@pytest.mark.parametrize("L,P,lens", [(784, 196, [784, 1, 196, 300]), (392, 196, [392, 391, 0, 197]), (333, 37, [64, 65, 333, 1])])
@pytest.mark.parametrize("scaled", [False, True])
def test_lens(L, P, lens, scaled):
    q, k, v = data(len(lens), L, 6, seed=3); n = torch.tensor(lens, device="cuda")
    o = check(q, k, v, P, n, *(scales() if scaled else (None, None)))
    q2, k2, v2 = (x.clone() for x in (q, k, v))                     # finite garbage in padding must not matter
    for b, m in enumerate(lens): q2[b, m:] = 5.; k2[b, m:] = -3.; v2[b, m:] = 1e4
    s = scales() if scaled else (None, None)
    with torch.no_grad(): o2 = attn(q2, k2, v2, P, n, *s)
    assert torch.equal(o, o2)

def test_causal():
    P, T = 196, 4; q, k, v = data(2, P * T, 6, seed=4)
    with torch.no_grad(): o = attn(q, k, v, P)
    for t in range(1, T):
        q2, k2, v2 = (x.clone() for x in (q, k, v))
        for x in (q2, k2, v2): x[:, t * P:] = torch.randn_like(x[:, t * P:]) * 50
        with torch.no_grad(): o2 = attn(q2, k2, v2, P)
        assert torch.equal(o[:, :t * P], o2[:, :t * P])

@pytest.mark.parametrize("qs", [1e-3, 1e-2, 30.0, 6000.0])
def test_qk_magnitude(qs): check(*data(2, 400, 4, seed=5, qs=qs), 100)
def test_zero_qk():
    q, k, v = data(2, 300, 4, seed=6); q[:, ::3] = 0; k[:, ::5] = 0; check(q, k, v, 100)
def test_all_equal_keys():
    q, k, v = data(2, 300, 4, seed=7); k[:] = k[:, :1]; check(q, k, v, 100)
@pytest.mark.parametrize("vmax", [4.0, 100.0, 30000.0])
def test_large_v(vmax): check(*data(2, 400, 4, seed=8, vmax=vmax), 100)
def test_constant_v():                                             # rows of P sum to 1
    q, k, v = data(2, 784, 6, seed=9); v[:] = 1.; o = check(q, k, v, 196); assert (o - 1).abs().max() < TOL
@pytest.mark.parametrize("gain", [4.0, 16.0, 60.0])
def test_extreme_logits(gain):
    """Saturated softmax: q_i matches exactly one key in frame 0; logits up to ~8 gain^2 (~29000). No near-ties."""
    L, P, H = 320, 64, 4; g = torch.Generator("cuda").manual_seed(10)
    Q = torch.linalg.qr(torch.randn(D, D, device="cuda", generator=g))[0] * 8
    q = Q[torch.arange(L, device="cuda") % D][None, :, None].expand(2, L, H, D).half().contiguous()
    k = q.clone(); v = data(2, L, H, seed=11)[2]
    s = torch.full((D,), gain, device="cuda")
    k[:, P:] = -k[:, P:]                                            # later frames never match
    check(q, k, v, P, None, s, s, tol=4e-3 if gain > 8 else TOL)

@pytest.mark.parametrize("which", ["q", "k", "v"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite(which, bad):
    q, k, v = data(3, 400, 4, seed=12); x = {"q": q, "k": k, "v": v}[which]; x[1, 120, 2, 5] = bad
    with torch.no_grad(): o = attn(q, k, v, 100)
    if which == "q": assert not torch.isfinite(o[1, 120, 2]).any()
    elif which == "k": assert not torch.isfinite(o[1, 100:, 2]).any()
    else: assert not torch.isfinite(o[1, 100:, 2, 5]).any()
    ok = torch.ones(3, 4, dtype=torch.bool); ok[1, 2] = False        # other batches and heads untouched
    assert torch.isfinite(o.transpose(1, 2)[ok.cuda()]).all()

def test_deterministic():
    x = data(2, 500, 6, seed=13)
    with torch.no_grad(): assert torch.equal(attn(*x, 196), attn(*x, 196))

def test_reference_fp64():
    x = data(2, 400, 4, seed=14); s = scales()
    r, d = impls.reference(*x, 100, None, *s), impls.reference(*x, 100, None, *s, dtype=torch.float64)
    assert (r - d).abs().max() < 2e-6
def test_fp64_spot():
    x = data(2, 784, 6, seed=15); s = scales()
    with torch.no_grad(): o = attn(*x, 196, None, *s)
    assert (o.double() - impls.reference(*x, 196, None, *s, dtype=torch.float64)).abs().max() < TOL

def grads(fn, q, k, v, P, lens, sq, sk, w):
    xs = [t.detach().clone().requires_grad_() for t in (q, k, v, sq, sk)]
    o = fn(xs[0], xs[1], xs[2], P, lens, xs[3], xs[4])
    return torch.autograd.grad((o.float() * w).sum(), xs)
@pytest.mark.grad
@pytest.mark.parametrize("B,L,H,P,lens", [(2, 392, 6, 196, None), (1, 1568, 16, 196, None), (2, 784, 6, 196, [784, 300]),
                                          (1, 1, 1, 1, None), (2, 333, 3, 37, None), (3, 129, 5, 64, [129, 0, 70])])
@pytest.mark.parametrize("qs", [1.0, 30.0, 0.01])
def test_grad(B, L, H, P, lens, qs):
    q, k, v = data(B, L, H, seed=16, qs=qs); n = None if lens is None else torch.tensor(lens, device="cuda")
    w = torch.randn(v.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(17))
    s = scales()
    g = grads(attn, q, k, v, P, n, *s, w); r = grads(impls.reference, q, k, v, P, n, *s, w)
    for name, a, b in zip(("q", "k", "v", "sq", "sk"), g, r):
        assert torch.isfinite(a).all(), name
        if b.float().norm() < 1e-6:        # true zero: rounding noise in normalized space, times 1/|q| for q and k
            assert a.float().norm() < 1e-5 * (max(1, 1 / qs) if name in "qk" else 1), name; continue
        e = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert e < GTOL, f"grad {name}: {e:.2e}"
