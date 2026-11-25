import json, os, sys
tag = sys.argv[1]
R = [r for r in map(json.loads, open(os.environ.get("OUT", "/data1/j50057870/volta-kernels/step2") + "/results.jsonl")) if r.get("suite") == "k1_fwd" and r["tag"] == tag]
print("| T | heads | B | scaled | masked ms | per-frame ms | kernel ms | kernel spread | speedup vs best | peak MiB masked/kernel | derived TFLOPS (needed only) |\n|" + "---|" * 11)
for r in R:
    k = r["kernel"]; print(f"| {r['T']} | {r['heads']} | {r['B']} | {r['scaled']} | {r['masked']['ms_med']:.3f} | {r['perframe']['ms_med']:.3f} | {k['ms_med']:.3f} | {(k['ms_max']-k['ms_min'])/k['ms_med']:.0%} | {r['speedup']:.2f}x | {r['masked']['peak_mib']:.0f}/{k['peak_mib']:.0f} | {r['eff_tflops']:.1f} |")
