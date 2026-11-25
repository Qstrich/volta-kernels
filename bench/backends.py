import warnings, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend as S
from common import *
hold_ballast()
BK = {"math": S.MATH, "mem_efficient": S.EFFICIENT_ATTENTION, "flash": S.FLASH_ATTENTION, "cudnn": S.CUDNN_ATTENTION}
T = 2; P = 49; L = T * P; m = block_mask(T, P)
def inputs(dt):
    g = torch.Generator("cuda").manual_seed(0)
    return [torch.randn(2, L, 4, 64, device="cuda", generator=g).mul_(0.5).to(dt).transpose(1, 2).requires_grad_() for _ in range(3)]
for dt in (torch.float16, torch.float32):
    for mask in ("none", "is_causal", "bool", "float"):
        am = {"bool": m, "float": torch.zeros(L, L, device="cuda", dtype=dt).masked_fill(~m, float("-inf"))}.get(mask)
        kw = dict(attn_mask=am, is_causal=mask == "is_causal")
        # FP32 math reference of the same inputs
        q = [x.detach().float().requires_grad_() for x in inputs(dt)]
        with sdpa_kernel([S.MATH]):
            kr = dict(attn_mask=am.float() if am is not None and am.dtype != torch.bool else am, is_causal=kw["is_causal"])
            ref = F.scaled_dot_product_attention(*q, **kr)
            gref = torch.autograd.grad(ref, q, torch.ones_like(ref))
        for name, be in BK.items():
            rec = dict(suite="backends", backend=name, dtype=str(dt), mask=mask)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    x = inputs(dt)
                    with sdpa_kernel([be]):
                        o = F.scaled_dot_product_attention(*x, **kw)
                        rec["fwd"] = "OK"; rec["fwd_maxerr"] = (o.float() - ref).abs().max().item()
                        rec["node"] = type(o.grad_fn).__name__
                        gr = torch.autograd.grad(o, x, torch.ones_like(o))
                        torch.cuda.synchronize()
                        rec["bwd"] = "OK"
                        rec["bwd_relerr"] = max(((a.float() - b).norm() / b.norm()).item() for a, b in zip(gr, gref))
                except Exception as e:
                    rec["bwd" if "fwd" in rec else "fwd"] = "ERR"; rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            rec["warnings"] = [str(x.message).splitlines()[0][:250] for x in w][:12]
            log(**rec)
