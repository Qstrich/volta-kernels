import torch, volta_ext
from common import *
hold_ballast()
torch.set_grad_enabled(False)
def rec(name, f, flop, **m):
    r = timeit(f, target=0.4); log(suite="ceil", name=name, tflops=flop / (r["ms_min"] / 1e3) / 1e12, tflops_med=flop / (r["ms_med"] / 1e3) / 1e12, **m, **r)
import sys
for dt in () if "triad" in sys.argv else (torch.float16, torch.float32):
    for n in (4096, 8192):
        a, b = torch.randn(n, n, device="cuda", dtype=dt), torch.randn(n, n, device="cuda", dtype=dt)
        rec("matmul", lambda: torch.matmul(a, b), 2 * n**3, n=n, dtype=str(dt))
    for L in (392, 784, 1568):
        for B in (24, 96):
            q, k = torch.randn(B, L, 64, device="cuda", dtype=dt), torch.randn(B, 64, L, device="cuda", dtype=dt)
            rec("bmm_qk", lambda: torch.bmm(q, k), 2 * B * L * L * 64, L=L, B=B, dtype=str(dt))
            p, v = torch.rand(B, L, L, device="cuda", dtype=dt), torch.randn(B, L, 64, device="cuda", dtype=dt)
            rec("bmm_pv", lambda: torch.bmm(p, v), 2 * B * L * L * 64, L=L, B=B, dtype=str(dt))
            del p, v
for gib in (1, 2, 4):
    n = gib * 2**30 // 4
    a, b, c = (torch.ones(n, device="cuda") for _ in range(3))
    r = timeit(lambda: volta_ext.triad(a, b, c, 1.5), target=0.4)
    log(suite="ceil", name="triad_v2", gib=gib, gbps=3 * n * 4 / (r["ms_min"] / 1e3) / 1e9, gbps_med=3 * n * 4 / (r["ms_med"] / 1e3) / 1e9, **r)
    del a, b, c; torch.cuda.empty_cache()
