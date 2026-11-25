import sys, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend as S
from common import *
hold_ballast()
BK = {"mem_efficient": S.EFFICIENT_ATTENTION, "math": S.MATH, "flash": S.FLASH_ATTENTION, "cudnn": S.CUDNN_ATTENTION}
P, D = 196, 64
which = sys.argv[1] if len(sys.argv) > 1 else "full"   # full | one
backends = (sys.argv[2] if len(sys.argv) > 2 else "mem_efficient,math").split(",")
def mk(B, L, h, rg=False):
    return [torch.randn(B, L, h, D, device="cuda", dtype=torch.float16).requires_grad_(rg) for _ in range(3)]
def rms32(x): return F.rms_norm(x.float(), (D,)).to(x.dtype)
def rms16(x): return F.rms_norm(x, (D,))
NORM = {"none": lambda x: x, "fp32": rms32, "fp16": rms16}
def run(tag, f, **meta):
    try:
        r = timeit(f); log(suite=tag, status="OK", **meta, **r)
    except torch.OutOfMemoryError:
        log(suite=tag, status="OOM", **meta); torch.cuda.empty_cache()
    except Exception as e:
        log(suite=tag, status="ERR", error=str(e).splitlines()[0][:200], **meta)
def case(be, T, h, B, mask, norm, mode):
    L = T * P; m = block_mask(T) if mask == "block" else None
    x = mk(B, L, h, mode == "fwdbwd"); gout = torch.ones(B, h, L, D, device="cuda", dtype=torch.float16)
    nf = NORM[norm]
    def f():
        q, k, v = x
        o = F.scaled_dot_product_attention(nf(q).transpose(1, 2), nf(k).transpose(1, 2), v.transpose(1, 2), attn_mask=m)
        if mode == "fwdbwd": torch.autograd.grad(o, x, gout)
    flop = 4 * B * h * L * L * D * (3 if mode == "fwdbwd" else 1)   # QK^T + PV, derived; backward counted as 2x forward
    with sdpa_kernel([BK[be]]), torch.set_grad_enabled(mode == "fwdbwd"):
        run("sdpa", f, backend=be, T=T, L=L, heads=h, B=B, mask=mask, qknorm=norm, mode=mode, dense_flop=flop,
            needed_frac=(T + 1) / (2 * T) if mask == "block" else 1.0)
def perframe(T, h, B, norm, mode):
    L = T * P; x = mk(B, L, h, mode == "fwdbwd"); gout = torch.ones(B, h, L, D, device="cuda", dtype=torch.float16); nf = NORM[norm]
    def f():
        q, k, v = nf(x[0]).transpose(1, 2), nf(x[1]).transpose(1, 2), x[2].transpose(1, 2)
        o = torch.cat([F.scaled_dot_product_attention(q[:, :, t*P:(t+1)*P], k[:, :, :(t+1)*P], v[:, :, :(t+1)*P]) for t in range(T)], 2)
        if mode == "fwdbwd": torch.autograd.grad(o, x, gout)
    with sdpa_kernel([S.EFFICIENT_ATTENTION]), torch.set_grad_enabled(mode == "fwdbwd"):
        run("perframe", f, T=T, heads=h, B=B, qknorm=norm, mode=mode)
if which == "perframe":
    for T in (2, 4, 8):
        for h in (6, 16):
            for B in (4, 16):
                for mode in ("fwd", "fwdbwd"):
                    for norm in ("none", "fp32"): perframe(T, h, B, norm, mode)
    sys.exit()
if which == "one":
    case("mem_efficient", 4, 6, 4, "block", "none", "fwd"); sys.exit()
for be in backends:
    for T in (2, 4, 8):
        for h in (6, 16):
            for B in (1, 4, 16):
                for mode in ("fwd", "fwdbwd"):
                    case(be, T, h, B, "block", "none", mode)
                    case(be, T, h, B, "none", "none", mode)
                    if be == "mem_efficient":
                        case(be, T, h, B, "block", "fp32", mode); case(be, T, h, B, "block", "fp16", mode)
                torch.cuda.empty_cache()
# kernel 2 baseline: prefix KV (T frames) expanded to N, concatenated with per-candidate suffix KV
for T in (2, 4, 8):
    for h in (6, 16):
        for N in (1, 8, 32):
            Lp = T * P
            kp, vp = (torch.randn(1, h, Lp, D, device="cuda", dtype=torch.float16) for _ in range(2))
            q, ks, vs = (torch.randn(N, h, P, D, device="cuda", dtype=torch.float16) for _ in range(3))
            def f():
                k = torch.cat([kp.expand(N, -1, -1, -1), ks], 2); v = torch.cat([vp.expand(N, -1, -1, -1), vs], 2)
                F.scaled_dot_product_attention(q, k, v)
            with sdpa_kernel([S.EFFICIENT_ATTENTION]), torch.no_grad():
                run("kv", f, T=T, Lprefix=Lp, heads=h, N=N, prefix_mib_once=2 * h * Lp * D * 2 / 2**20)
