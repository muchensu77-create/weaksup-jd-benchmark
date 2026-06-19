#!/usr/bin/env python3
"""
Mac(MPS) 评测启动器 —— 复用 code/ 的 train.py / evaluate.py，跑 4 方法 × 多 seed。

域适配关键：在 import train/evaluate 之前先改 config.PROTOTYPE_CSV / RULES_JSON，
所有消费方(dataset/rules/evaluate)用 `from config import X` 在导入时绑定，自然吃到 LAE 原型/规则。

ICV:  python run_gold_eval.py --train .../icv_train_labeled_v2.csv --test .../icv_test_gold.csv --out .../results_gold_icv.json
LAE:  python run_gold_eval.py --domain lae --train .../lae_train_labeled.csv --test .../lae_test_gold.csv --out .../results_gold_lae.json
冒烟: --methods hybrid_or --seeds 42 --epochs 1
"""
import sys, os, json, argparse, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

# 可移植 ROOT：优先环境变量 GZROOT，否则由脚本位置推导(pipeline/scripts/ -> 项目根)
ROOT = os.environ.get("GZROOT") or str(Path(__file__).resolve().parents[2])
METRIC_KEYS = ["accuracy", "macro_f1", "positive_macro_f1", "negative_f1", "top5_accuracy"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["icv", "lae"], default="icv")
    ap.add_argument("--methods", nargs="+",
                    default=["rf_probability", "hybrid_or", "neurosymbolic", "bert_finetune"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lambda_val", type=float, default=0.1)
    ap.add_argument("--prototype_csv", default=None)
    ap.add_argument("--rules_json", default=None)
    ap.add_argument("--out", default="")
    return ap.parse_args()

def main():
    a = parse_args()
    # 入参转绝对路径(相对 ROOT)，因为下面会 chdir 到 code/
    def absify(p): return p if (p is None or os.path.isabs(p)) else os.path.join(ROOT, p)
    a.train, a.test = absify(a.train), absify(a.test)
    a.out = absify(a.out) if a.out else a.out
    a.prototype_csv, a.rules_json = absify(a.prototype_csv), absify(a.rules_json)
    sys.path.insert(0, ROOT + "/code")
    sys.path.insert(1, ROOT + "/code/experiments_v1")
    os.chdir(ROOT + "/code")

    # ---- 关键：导入 train/evaluate 之前先打补丁 config ----
    import config as cfg
    if not os.path.exists(cfg.PROTOTYPE_CSV):
        shutil.copy(ROOT + "/code/data/jd.csv", cfg.PROTOTYPE_CSV)
    proto = a.prototype_csv or (ROOT + "/code/data/lae_prototypes.csv" if a.domain == "lae" else cfg.PROTOTYPE_CSV)
    rules = a.rules_json or (ROOT + "/code/data/lae_rules.json" if a.domain == "lae" else cfg.RULES_JSON)
    cfg.PROTOTYPE_CSV = proto
    cfg.RULES_JSON = rules
    print(f"[domain={a.domain}] PROTOTYPE_CSV={proto}\n            RULES_JSON={rules}")

    from train import train          # 此时 from config import 取到已打补丁的值
    from evaluate import evaluate

    results, agg = {}, {}
    for m in a.methods:
        per = defaultdict(list); results[m] = {}
        for s in a.seeds:
            print(f"\n>>> domain={a.domain} method={m} seed={s} epochs={a.epochs}")
            ck = train(method=m, seed=s, train_data=a.train,
                       epochs=a.epochs, lambda_val=a.lambda_val, tag=f"gold_{a.domain}_{m}_s{s}")
            met = evaluate(checkpoint_path=ck, test_data=a.test, method=m)
            results[m][s] = met
            print(f"    -> {({k: round(met[k],4) for k in METRIC_KEYS if k in met})}")
            for k in METRIC_KEYS:
                if k in met: per[k].append(met[k])
        agg[m] = {k: {"mean": round(float(np.mean(v)), 4),
                      "std": round(float(np.std(v)), 4), "values": v}
                  for k, v in per.items() if v}

    output = {"domain": a.domain, "methods": a.methods, "seeds": a.seeds,
              "train": a.train, "test": a.test, "epochs": a.epochs,
              "prototype_csv": proto, "rules_json": rules,
              "per_run": results, "aggregated": agg}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(output, open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n[写出] {a.out}")
    print("\n===== 汇总 =====")
    for m, ag in agg.items():
        print(m, {k: f"{v['mean']}±{v['std']}" for k, v in ag.items()})

if __name__ == "__main__":
    main()
