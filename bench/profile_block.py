"""Attention share of one pre-norm ViT block, by A/B removal plus the torch profiler."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend as S
from torch.profiler import profile, ProfilerActivity
from common import *
hold_ballast()
P = 196
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h, s.d = h, d
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.mode = "full"
    def forward(s, x, m):
        B, L, _ = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, L, 3, s.h, 64).unbind(2)
        if s.mode != "no_qknorm" and s.mode != "no_attn_no_qknorm":
            q, k = F.rms_norm(q.float(), (64,)).half(), F.rms_norm(k.float(), (64,)).half()
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = v + 0 * (q + k) if s.mode.startswith("no_attn") else F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + s.proj(o.transpose(1, 2).reshape(B, L, -1))
        return x + s.mlp(s.n2(x))
for d, h, B in ((384, 6, 16), (1024, 16, 4)):
    for T in (2, 4, 8):
        m = block_mask(T); blk = Block(d, h).cuda().half()
        x = torch.randn(B, T * P, d, device="cuda", dtype=torch.float16, requires_grad=True)
        for mode in ("fwd", "fwdbwd"):
            def step():
                o = blk(x, m)
                if mode == "fwdbwd": torch.autograd.grad(o.float().sum(), [x] + list(blk.parameters()))
            with sdpa_kernel([S.EFFICIENT_ATTENTION]), torch.set_grad_enabled(mode == "fwdbwd"):
                res = {}
                for s in ("full", "no_qknorm", "no_attn", "no_attn_no_qknorm"):
                    blk.mode = s; res[s] = timeit(step)["ms_med"]
                busy, fmha, top = {}, 0, None
                for md in ("full", "no_qknorm", "no_attn", "no_attn_no_qknorm"):
                    blk.mode = md
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for _ in range(5): step()
                        torch.cuda.synchronize()
                    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(("aten::", "ProfilerStep"))]
                    agg = {}
                    for e in ev: agg[e.name] = agg.get(e.name, 0) + e.device_time_total
                    busy[md] = sum(agg.values()) / 5 / 1e3
                    if md == "full":
                        fmha = sum(v for k, v in agg.items() if "fmha" in k) / 5 / 1e3
                        top = [(k[:50], round(v / 5 / 1e3, 3)) for k, v in sorted(agg.items(), key=lambda x: -x[1])[:6]]
            log(suite="block2", d=d, heads=h, B=B, T=T, mode=mode, wall_ms=res, gpu_busy_ms=busy, fmha_ms=fmha,
                fmha_share=fmha / busy["full"], attn_share=1 - busy["no_attn"] / busy["full"],
                attn_qk_share=1 - busy["no_attn_no_qknorm"] / busy["full"], top=top)
