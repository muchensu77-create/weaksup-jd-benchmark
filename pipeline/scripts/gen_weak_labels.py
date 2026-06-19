"""Phase A3/A4: drive weak-label generation for LAE and extended ICV data.

Reuses experiments/weak_label_gen.generate_weak_labels with custom prototypes +
custom rules path. Unlike the upstream CLI, this driver does not assert 215
prototypes so it works for LAE (181 classes) and skips the ICV-tailored rules.

Outputs (under pipeline/data/):
  LAE: lae_train_labeled.csv, lae_test.csv
  ICV(v2): icv_train_labeled_v2.csv, icv_test_v2.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os as _os  # 可移植 ROOT
ROOT = Path(_os.environ.get("GZROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "code"))                  # config.py, rules.py
sys.path.insert(1, str(ROOT / "code" / "experiments_v1"))  # weak_label_gen.py

# monkeypatch RULES_JSON before importing
import config as cfg  # noqa: E402

# we will override cfg.RULES_JSON per-run
import rules as rules_mod  # noqa: E402
from weak_label_gen import encode_texts, build_position_name_clusters  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gen_weak_labels")

OUT_DIR = ROOT / "code" / "data"   # 训练集直接写到 runner 读取处
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "pipeline" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_rules_from(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exclude_gold(unlabeled: pd.DataFrame, gold_csv: str) -> pd.DataFrame:
    """从未标注池剔除 gold 测试文本(精确匹配)，杜绝 train/test 输入级泄漏。"""
    import os as _o
    if not gold_csv or not _o.path.exists(gold_csv):
        return unlabeled
    gold = set(pd.read_csv(gold_csv)["text"].astype(str))
    before = len(unlabeled)
    out = unlabeled[~unlabeled["text"].astype(str).isin(gold)].reset_index(drop=True)
    log.info(f"exclude_gold: 剔除 {before - len(out)} 条与 gold 重叠的未标注样本 ({before}->{len(out)})")
    return out


def weak_label_core(
    prototypes_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    encoder_name: str,
    tau_high: float,
    tau_low: float,
    delta_inter: float,
    delta_intra: float,
    rules: dict,
    num_classes: int,
) -> tuple[pd.DataFrame, dict]:
    """Copy of experiments.weak_label_gen.generate_weak_labels with rules passed in."""
    log.info(f"Encoder = {encoder_name}; {len(prototypes_df)} prototypes, {len(unlabeled_df)} unlabeled")
    model = SentenceTransformer(encoder_name)

    proto_emb = encode_texts(model, prototypes_df["all"].tolist(), batch_size=64)
    un_emb = encode_texts(model, unlabeled_df["text"].tolist(), batch_size=64)
    sim = un_emb @ proto_emb.T

    clusters = build_position_name_clusters(prototypes_df)

    results = []
    stats = {"positive": 0, "negative": 0, "abstain": 0, "rule_pos": 0, "rule_reject": 0}

    for i in range(len(unlabeled_df)):
        text = unlabeled_df["text"].iloc[i]
        sims = sim[i]
        sidx = np.argsort(sims)[::-1]
        s_max = sims[sidx[0]]
        s_second = sims[sidx[1]]
        k_star = int(sidx[0])
        k_second = int(sidx[1])
        delta = delta_intra if k_second in clusters.get(k_star, set()) else delta_inter

        rule_label, rule_name = rules_mod.rule_apply(text, prototypes_df, rules)

        if rule_label == 0:
            label = 0
            triggered = rule_name
            conf = float(s_max)
            stats["negative"] += 1
            stats["rule_reject"] += 1
        elif s_max < tau_low:
            label = 0
            triggered = "low_similarity"
            conf = float(s_max)
            stats["negative"] += 1
        elif s_max > tau_high and (s_max - s_second) > delta and rule_label != 0:
            if rule_label is not None and rule_label > 0:
                label = rule_label
                triggered = rule_name
                stats["rule_pos"] += 1
            else:
                label = k_star + 1
                triggered = "similarity"
            conf = float(s_max)
            stats["positive"] += 1
        else:
            stats["abstain"] += 1
            continue

        results.append({
            "text": text,
            "weak_label": int(label),
            "confidence": round(float(conf), 4),
            "rule_triggered": triggered,
        })

    return pd.DataFrame(results), stats


def stratified_test_split(labeled_df: pd.DataFrame, n_test: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by weak_label — cap test at min(n_test, 30% of total)."""
    target_test = min(n_test, int(len(labeled_df) * 0.3))
    grouped = labeled_df.groupby("weak_label")
    fraction = target_test / max(1, len(labeled_df))
    test_parts, train_parts = [], []
    for _, sub in grouped:
        sub_shuffled = sub.sample(frac=1.0, random_state=seed)
        n_take = int(round(len(sub_shuffled) * fraction)) if len(sub_shuffled) >= 3 else 0
        test_parts.append(sub_shuffled.iloc[:n_take])
        train_parts.append(sub_shuffled.iloc[n_take:])
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_lae(args):
    proto = pd.read_csv(OUT_DIR / "lae_prototypes.csv")
    raw = pd.read_csv(OUT_DIR / "lae_raw.csv")
    # Build "text" column for prototypes already present; ensure unlabeled has text
    unlabeled = raw[["text", "positionName"]].copy()
    unlabeled = exclude_gold(unlabeled, args.exclude_gold or str(ROOT / "code" / "data" / "lae_test_gold.csv"))
    # Apply length filter to keep pool manageable
    if args.max_unlabeled and len(unlabeled) > args.max_unlabeled:
        unlabeled = unlabeled.sample(n=args.max_unlabeled, random_state=42).reset_index(drop=True)

    rules = load_rules_from(str(OUT_DIR / "lae_rules.json"))

    labeled_df, stats = weak_label_core(
        prototypes_df=proto,
        unlabeled_df=unlabeled,
        encoder_name=cfg.ENCODER_NAME,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        delta_inter=args.delta_inter,
        delta_intra=args.delta_intra,
        rules=rules,
        num_classes=len(proto),
    )
    log.info(f"LAE weak-labeling stats: {stats}")
    log.info(f"Labeled rows: {len(labeled_df)}")
    if len(labeled_df) == 0:
        raise SystemExit("No labeled rows produced; aborting.")

    train_df, test_df = stratified_test_split(labeled_df, n_test=args.n_test, seed=42)
    # need a 'label' column for test (equiv to weak_label here since we use weak labels as silver ground truth)
    test_df = test_df.copy()
    test_df["label"] = test_df["weak_label"]
    train_df.to_csv(OUT_DIR / "lae_train_labeled.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "lae_test.csv", index=False, encoding="utf-8-sig")
    log.info(f"Wrote LAE train/test: {len(train_df)}/{len(test_df)}")
    (LOG_DIR / "lae_weak_label_stats.json").write_text(
        json.dumps({"stats": stats, "n_train": len(train_df), "n_test": len(test_df),
                    "n_classes": int(proto['prototype_id'].max()) + 1,
                    "tau_high": args.tau_high, "tau_low": args.tau_low}, indent=2),
        encoding="utf-8")


def run_icv_v2(args):
    # prototypes = experiments/data/prototypes.csv (215 rows + implicit class 0)
    proto = pd.read_csv(cfg.PROTOTYPE_CSV)  # jd.csv (215)
    # Re-load experiments/data/prototypes.csv for the "all" column + prototype_id
    proto_full = pd.read_csv(ROOT / "code" / "experiments_v1" / "data" / "prototypes.csv")
    # unlabeled = mongodb_data.csv (40K) with filtering
    mongo = pd.read_csv(ROOT / "code" / "data" / "mongodb_data.csv", low_memory=False)
    mongo = mongo.dropna(subset=["positionDetail"]).copy()
    mongo["positionDetail"] = mongo["positionDetail"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    mongo = mongo[mongo["positionDetail"].str.len() >= 60].copy()
    if args.max_unlabeled and len(mongo) > args.max_unlabeled:
        mongo = mongo.sample(n=args.max_unlabeled, random_state=42).reset_index(drop=True)
    mongo["text"] = mongo["positionName"].fillna("") + "。" + mongo["positionDetail"]
    unlabeled = mongo[["text", "positionName"]].copy()
    unlabeled = exclude_gold(unlabeled, args.exclude_gold or str(ROOT / "code" / "data" / "icv_test_gold.csv"))

    rules = load_rules_from(str(ROOT / "code" / "experiments_v1" / "rules.json"))
    labeled_df, stats = weak_label_core(
        prototypes_df=proto_full,
        unlabeled_df=unlabeled,
        encoder_name=cfg.ENCODER_NAME,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        delta_inter=args.delta_inter,
        delta_intra=args.delta_intra,
        rules=rules,
        num_classes=len(proto_full) + 1,
    )
    log.info(f"ICV-v2 weak-labeling stats: {stats}")

    train_df, test_df = stratified_test_split(labeled_df, n_test=args.n_test, seed=42)
    test_df = test_df.copy()
    test_df["label"] = test_df["weak_label"]
    train_df.to_csv(OUT_DIR / "icv_train_labeled_v2.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "icv_test_v2.csv", index=False, encoding="utf-8-sig")
    log.info(f"Wrote ICV-v2 train/test: {len(train_df)}/{len(test_df)}")
    (LOG_DIR / "icv_v2_weak_label_stats.json").write_text(
        json.dumps({"stats": stats, "n_train": len(train_df), "n_test": len(test_df),
                    "n_classes": 216, "tau_high": args.tau_high, "tau_low": args.tau_low}, indent=2),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["lae", "icv_v2"], required=True)
    ap.add_argument("--tau_high", type=float, default=0.70)
    ap.add_argument("--tau_low", type=float, default=0.33)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--max_unlabeled", type=int, default=0, help="0 = use all; else subsample")
    ap.add_argument("--delta_inter", type=float, default=cfg.DELTA_INTER, help="跨名margin(放小→更多标签更多噪声)")
    ap.add_argument("--delta_intra", type=float, default=cfg.DELTA_INTRA)
    ap.add_argument("--exclude_gold", default=None, help="剔除该 gold csv 的文本以防泄漏(默认按域自动选)")
    args = ap.parse_args()
    if args.domain == "lae":
        run_lae(args)
    else:
        run_icv_v2(args)


if __name__ == "__main__":
    main()
