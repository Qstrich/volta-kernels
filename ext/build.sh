#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p build
rm -f build/volta_ext*.so                     # never leave a stale binary behind a failed build
python setup.py build_ext --build-lib build --build-temp build/tmp > build/log 2>&1 || { grep -nE 'error' build/log | head -20; echo BUILD FAILED; exit 1; }
so=$(ls build/volta_ext*.so); echo "== $so"; cuobjdump --list-elf "$so"; cuobjdump -res-usage "$so" 2>/dev/null | grep -A1 attn_k | grep -oE 'ILb[01]|REG:[0-9]+'
