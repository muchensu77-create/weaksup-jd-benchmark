#!/usr/bin/env python3
"""泄漏审计：确认每个训练集与其 gold 测试集【输入级零重叠】。
论文承诺随artifact发布。用法: python audit_overlap.py"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "code" / "data"
PAIRS = [
    ("icv_train_conservative_clean.csv", "icv_test_gold.csv"),
    ("icv_train_relaxed_clean.csv",      "icv_test_gold.csv"),
]
def norm(s): return " ".join(str(s).split()).strip()
ok = True
for tr, te in PAIRS:
    t = set(norm(x) for x in pd.read_csv(D / tr)["text"])
    g = set(norm(x) for x in pd.read_csv(D / te)["text"])
    ov = len(t & g)
    print(f"{tr:34s} vs {te:20s}: overlap={ov} / gold={len(g)}  {'OK' if ov == 0 else 'LEAK!'}")
    ok = ok and ov == 0
print("\nRESULT:", "ZERO LEAKAGE ✅" if ok else "LEAKAGE DETECTED ❌")
sys.exit(0 if ok else 1)
