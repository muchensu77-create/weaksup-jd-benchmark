#!/usr/bin/env python3
"""Codex 复查要求的 trivial 基线（推理即可，无需训练）：
majority-negative / similarity-only(text2vec最近原型) / rules-only。
在 ICV gold(247) 上算 Acc / active-Macro-F1 / Top5，与主表口径一致。"""
import sys, json, os
import numpy as np, pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score

ROOT = "/Users/su/Desktop/岗位分类论文_IEEE_Access投稿"
sys.path.insert(0, ROOT + "/code")
from rules import load_rules, rule_apply  # ICV 规则/原型默认路径

ENC = "shibing624/text2vec-base-chinese"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

gold = pd.read_csv(f"{ROOT}/code/data/icv_test_gold.csv")
y = gold["label"].astype(int).values
proto = pd.read_csv(f"{ROOT}/code/data/jd.csv")              # 215 原型, 行i -> 类 i+1
proto_full = pd.read_csv(f"{ROOT}/code/experiments_v1/data/prototypes.csv")
rules = load_rules(f"{ROOT}/code/experiments_v1/rules.json")

def metrics(y_true, y_pred, top5=None):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pos = [(t, p) for t, p in zip(y_true, y_pred) if t != 0]
    pmf1 = f1_score([t for t, _ in pos], [p for _, p in pos], average="macro", zero_division=0) if pos else 0.0
    out = {"acc": round(acc, 4), "macro_f1": round(mf1, 4), "pos_macro_f1": round(pmf1, 4)}
    if top5 is not None:
        out["top5"] = round(float(np.mean([yt in t5 for yt, t5 in zip(y_true, top5)])), 4)
    return out

# 1) majority-negative：全预测 0
print("majority-negative :", metrics(y, np.zeros_like(y)))

# 2) similarity-only：text2vec 最近原型(类=idx+1)，不预测负类
model = SentenceTransformer(ENC, device=DEV)
def emb(texts): return model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
pe = emb(proto["all"].astype(str).tolist())
ge = emb(gold["text"].astype(str).tolist())
sim = ge @ pe.T
pred_sim = sim.argmax(1) + 1
top5 = [set((np.argsort(-row)[:5] + 1).tolist()) for row in sim]
print("similarity-only   :", metrics(y, pred_sim, top5))

# 3) rules-only：规则触发则用规则标签，否则预测负类0
pred_rule = []
for t in gold["text"].astype(str):
    lab, _ = rule_apply(t, proto_full, rules)
    pred_rule.append(int(lab) if (lab is not None) else 0)
print("rules-only        :", metrics(y, np.array(pred_rule)))

# 存档
res = {
    "majority_negative": metrics(y, np.zeros_like(y)),
    "similarity_only": metrics(y, pred_sim, top5),
    "rules_only": metrics(y, np.array(pred_rule)),
    "note": "ICV gold 247 rows, 122 active classes incl negative; metrics match paper protocol",
}
os.makedirs(f"{ROOT}/pipeline/experiments", exist_ok=True)
json.dump(res, open(f"{ROOT}/pipeline/experiments/baselines_icv.json", "w"), ensure_ascii=False, indent=2)
print("\n写出 pipeline/experiments/baselines_icv.json")
