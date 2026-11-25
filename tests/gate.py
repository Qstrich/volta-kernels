"""Stock must pass, every mutant must fail. Prints one line per implementation."""
import os, subprocess, sys, impls
bad = 0
for n in ["stock"] + impls.MUTANTS:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_parity.py", "-c", "tests/pytest.ini", "--tb=line"] + (["-x"] if n != "stock" else []),
                       env={**os.environ, "VOLTA_IMPL": n}, capture_output=True, text=True)
    ok = (r.returncode == 0) if n == "stock" else (r.returncode == 1)
    bad += not ok; print(n, "OK" if ok else "UNEXPECTED", r.returncode, r.stdout.strip().splitlines()[-1], flush=True)
    if not ok or n == "stock": print("\n".join(l for l in r.stdout.splitlines() if "Error" in l or l.startswith(("FAILED", "E ")))[:5000])
sys.exit(bad)
