import json, os, statistics, subprocess, threading, time
import torch

OUT = os.environ.get("OUT", ".")
BALLAST = None

def hold_ballast(gib=1.5):
    """Hold >1 GB so a peer scheduler's free-memory check skips this GPU."""
    global BALLAST
    BALLAST = torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

def log(**rec):
    rec["host_gpu"] = os.environ.get("VOLTA_HOST_GPU")
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

def smi(fields="clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"):
    out = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    return dict(zip(fields.split(","), out))

class ClockSampler:
    def __enter__(self):
        self.s, self.stop = [], threading.Event()
        def go():
            while not self.stop.is_set():
                try: self.s.append(smi())
                except Exception: pass
                self.stop.wait(0.15)
        self.t = threading.Thread(target=go, daemon=True); self.t.start()
        return self
    def __exit__(self, *a):
        self.stop.set(); self.t.join()
    def summary(self):
        if not self.s: return {}
        c = [int(x["clocks.sm"]) for x in self.s if x.get("clocks.sm", "").isdigit()]
        return {"sm_mhz_med": statistics.median(c) if c else None, "sm_mhz_min": min(c) if c else None,
                "throttle": sorted({x.get("clocks_throttle_reasons.active") for x in self.s})}

def timeit(fn, repeats=7, target=0.15, warm=3):
    """Median/min/max seconds per call, from CUDA events. Peak memory excludes what was live before the call."""
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); one = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() - live
    n = max(1, min(2000, int(target / max(one, 1e-6))))
    out = []
    with ClockSampler() as cs:
        for _ in range(repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n): fn()
            b.record(); b.synchronize()
            out.append(a.elapsed_time(b) / 1e3 / n)
    return {"ms_med": statistics.median(out) * 1e3, "ms_min": min(out) * 1e3, "ms_max": max(out) * 1e3,
            "inner": n, "peak_mib": peak / 2**20, **cs.summary()}

def block_mask(T, P=196, dev="cuda"):
    f = torch.arange(T, device=dev).repeat_interleave(P)
    return f[:, None] >= f[None, :]
