"""Kernel 1 forward vs both stock baselines (masked SDPA; one unmasked SDPA per frame), FP32 QK-norm, same process."""
import sys, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend as S
from common import *
import volta_attn
hold_ballast(); torch.set_grad_enabled(False)
P, D = 196, 64; tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
def norm(x, s):
    x = F.rms_norm(x.float(), (D,)); return (x if s is None else x * s).half()
for T in (2, 4, 8):
    for h in (6, 16):
        for B in (1, 4, 16):
            L = T * P; m = block_mask(T)
            q, k, v = (torch.randn(B, L, h, D, device="cuda", dtype=torch.float16) for _ in range(3))
            for scaled in (False, True):
                sq, sk = (torch.rand(D, device="cuda") + 0.5 for _ in range(2)) if scaled else (None, None)
                def masked(): return F.scaled_dot_product_attention(norm(q, sq).transpose(1, 2), norm(k, sk).transpose(1, 2), v.transpose(1, 2), attn_mask=m)
                def perframe():
                    a, b, c = norm(q, sq).transpose(1, 2), norm(k, sk).transpose(1, 2), v.transpose(1, 2)
                    return torch.cat([F.scaled_dot_product_attention(a[:, :, t*P:(t+1)*P], b[:, :, :(t+1)*P], c[:, :, :(t+1)*P]) for t in range(T)], 2)
                r = {}
                with sdpa_kernel([S.EFFICIENT_ATTENTION]):
                    for n, f in (("masked", masked), ("perframe", perframe), ("kernel", lambda: volta_attn.attn(q, k, v, P, None, sq, sk))): r[n] = timeit(f)
                best = min(r["masked"]["ms_med"], r["perframe"]["ms_med"])
                log(suite="k1_fwd", tag=tag, T=T, heads=h, B=B, scaled=scaled, **{n: x for n, x in r.items()}, speedup=best / r["kernel"]["ms_med"],
                    eff_tflops=4 * B * h * L * L * D * (T + 1) / (2 * T) / r["kernel"]["ms_med"] / 1e9)
