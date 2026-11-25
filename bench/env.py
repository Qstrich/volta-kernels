import subprocess, torch
from common import *
hold_ballast()
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
log(suite="env", torch=torch.__version__, cuda=torch.version.cuda, arch=torch.cuda.get_arch_list(),
    dev=torch.cuda.get_device_name(0), driver=sh("nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader"),
    nvcc=sh("nvcc --version | tail -2 | head -1"), gcc=sh("gcc --version | head -1"), cudnn=torch.backends.cudnn.version(),
    clocks=smi("clocks.sm,clocks.max.sm,clocks.applications.gr,clocks.mem"))
assert "sm_70" in torch.cuda.get_arch_list(), "torch build lacks sm_70"
import volta_ext
n = 1 << 20
b, c, a = torch.randn(n, device="cuda"), torch.randn(n, device="cuda"), torch.empty(n, device="cuda")
volta_ext.triad(a, b, c, 1.5); torch.cuda.synchronize()
err = (a - (b + 1.5 * c)).abs().max().item()
log(suite="ext", triad_max_err=err); assert err < 1e-5
