#!/usr/bin/env python3
"""跨平台干净重训：在【已剔除gold、零泄漏】的训练集上 train+eval。
先自检 overlap=0，再跑 ICV(relaxed + conservative)，结果存 *_clean.json。"""
import subprocess, sys, os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("GZROOT") or Path(__file__).resolve().parents[2])
PY = sys.executable
RUN = str(ROOT / "pipeline" / "scripts" / "run_gold_eval.py")
D = ROOT / "code" / "data"
EXP = ROOT / "pipeline" / "experiments"; EXP.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, GZROOT=str(ROOT), TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="1")

# ---- 自检：训练集与 gold 零重叠 ----
print("== overlap 自检 ==")
ok = True
for tr, gd in [("icv_train_relaxed_clean.csv","icv_test_gold.csv"),
               ("icv_train_conservative_clean.csv","icv_test_gold.csv")]:
    t = set(pd.read_csv(D/tr)["text"].astype(str)); g = set(pd.read_csv(D/gd)["text"].astype(str))
    ov = len(t & g); print(f"  {tr}: {len(t)} rows, overlap={ov}")
    ok = ok and ov == 0
if not ok:
    sys.exit("！overlap 非零，终止")
print("零泄漏，开始重训\n")

JOBS = [
    ("icv", "icv_train_relaxed_clean.csv",      "icv_test_gold.csv", "results_gold_icv_clean.json"),
    ("icv", "icv_train_conservative_clean.csv", "icv_test_gold.csv", "results_gold_icv_cons_clean.json"),
]
for i, (dom, tr, te, out) in enumerate(JOBS, 1):
    print(f"\n==== [{i}/3] {dom} train={tr} ====", flush=True)
    r = subprocess.run([PY, RUN, "--domain", dom, "--train", str(D/tr),
                        "--test", str(D/te), "--out", str(EXP/out)], env=env)
    if r.returncode != 0:
        sys.exit(f"步骤 {i} 失败")
print("\n==== 全部完成 ====  结果: results_gold_icv_clean.json / results_gold_icv_cons_clean.json")
