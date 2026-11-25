import json, os
D = os.environ.get("OUT", "/data1/j50057870/volta-kernels/step1")
R = [json.loads(l) for l in open(f"{D}/results.jsonl")]
g = lambda **k: [r for r in R if all(r.get(a) == b for a, b in k.items())]
out = []; P = out.append
def table(h, rows): P("| " + " | ".join(h) + " |"); P("|" + "---|" * len(h)); [P("| " + " | ".join(map(str, r)) + " |") for r in rows]; P("")
sd = {}
for r in g(suite="sdpa"):
    sd.setdefault((r["backend"], r["T"], r["heads"], r["B"], r["mask"], r["qknorm"], r["mode"]), r)
pf = {(r["T"], r["heads"], r["B"], r["qknorm"], r["mode"]): r["ms_med"] for r in g(suite="perframe")}
P(open(f"{D}/findings.md").read())
P("## SDPA backend support (forced one at a time)\n")
table(["backend", "dtype", "mask", "fwd", "bwd", "fwd max err vs FP32 math", "bwd rel err"],
      [(r["backend"], r["dtype"][6:], r["mask"], r.get("fwd"), r.get("bwd", "-"), f"{r['fwd_maxerr']:.1e}" if "fwd_maxerr" in r else "-", f"{r['bwd_relerr']:.1e}" if "bwd_relerr" in r else "-") for r in g(suite="backends")])
for mode in ("fwd", "fwdbwd"):
    P(f"## Baseline, mem-efficient SDPA, {mode}: median ms (FP16, d=64/head)\n"); rows = []
    for T in (2, 4, 8):
        for h in (6, 16):
            for B in (1, 4, 16):
                f = lambda m, n: sd[("mem_efficient", T, h, B, m, n, mode)]
                a, n, q32, q16 = f("block", "none"), f("none", "none"), f("block", "fp32"), f("block", "fp16")
                fl = mode == "fwdbwd" and a["ms_med"] < 1.3
                best = min(q32["ms_med"], pf.get((T, h, B, "fp32", mode), 1e9))
                rows.append((T, h, B, f"{n['ms_med']:.3f}", f"{a['ms_med']:.3f}", f"{(a['ms_max']-a['ms_min'])/a['ms_med']*100:.0f}%", f"{q32['ms_med']:.3f}", f"{q16['ms_med']:.3f}",
                             f"{pf[(T,h,B,'fp32',mode)]:.3f}" if (T, h, B, "fp32", mode) in pf else "-", f"{best:.3f}", f"{best/1.3:.3f}", f"{a['dense_flop']/a['ms_med']/1e9:.1f}", "CPU-bound" if fl else ""))
    table(["T", "heads", "B", "no mask", "block mask", "spread", "+QK-norm FP32", "+QK-norm FP16", "per-frame +QKn FP32", "best (QKn FP32)", "1.3x target", "derived TFLOPS (dense)", "note"], rows)
P("## Kernel 2 baseline (prefix KV expanded, no_grad)\n")
table(["T prefix", "heads", "N", "ms", "peak MiB", "prefix stored once MiB"], [(r["T"], r["heads"], r["N"], f"{r['ms_med']:.3f}", f"{r['peak_mib']:.0f}", f"{r['prefix_mib_once']:.2f}") for r in g(suite="kv")])
P("## Ceilings\n")
table(["name", "n/L/GiB", "B", "dtype", "median", "unit", "SM MHz med", "throttle reasons"],
      [(r["name"], r.get("n") or r.get("L") or r.get("gib"), r.get("B", ""), r.get("dtype", "")[6:], f"{r.get('tflops_med', r.get('gbps_med')):.1f}", "TFLOPS" if "tflops_med" in r else "GB/s", r["sm_mhz_med"], ",".join(r["throttle"])) for r in g(suite="ceil") if r["name"] != "triad"])
P("## Attention share of one ViT block (GPU busy time, masked SDPA, FP32 QK-norm)\n")
table(["d", "B", "T", "mode", "wall ms", "GPU busy ms", "SDPA kernels ms", "SDPA kernel share", "attn share (A/B)", "attn+QK-norm share", "block speedup if attn+QKn 1.3x"],
      [(r["d"], r["B"], r["T"], r["mode"], f"{r['wall_ms']['full']:.1f}", f"{r['gpu_busy_ms']['full']:.2f}", f"{r['fmha_ms']:.2f}", f"{r['fmha_share']:.0%}", f"{r['attn_share']:.0%}", f"{r['attn_qk_share']:.0%}", f"{1/(1-r['attn_qk_share']+r['attn_qk_share']/1.3):.2f}x") for r in g(suite="block2")])
open(f"{D}/REPORT.md", "w").write("\n".join(out))
