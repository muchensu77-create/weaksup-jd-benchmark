"""
Weak label generation pipeline.

Encodes prototype and unlabeled JDs with text2vec-base-chinese, computes
cosine similarity, and applies 3-tier labeling (positive / negative / abstain).

Usage:
    python weak_label_gen.py --input data/unlabeled.csv --output data/train_labeled.csv

Python interpreter: E:/Anaconda/python.exe
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    DELTA_INTER,
    DELTA_INTRA,
    ENCODER_NAME,
    NUM_CLASSES,
    PROTOTYPE_CSV,
    TAU_HIGH_ICV,
    TAU_LOW_ICV,
)
from rules import load_rules, rule_apply

logger = logging.getLogger(__name__)


def load_prototypes(path: str = PROTOTYPE_CSV) -> pd.DataFrame:
    """Load the 215-row prototype CSV."""
    df = pd.read_csv(path)
    assert len(df) == 215, f"Expected 215 prototypes, got {len(df)}"
    assert "positionName" in df.columns, "Missing positionName column"
    assert "all" in df.columns, "Missing 'all' column"
    logger.info("Loaded %d prototypes from %s", len(df), path)
    return df


def load_unlabeled(path: str) -> pd.DataFrame:
    """Load unlabeled JDs from CSV or XLSX."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    # Expect a text column; try common names
    text_col = None
    for candidate in ["all", "text", "content", "positionDetail", "jd_text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"Cannot find text column. Available: {list(df.columns)}"
        )
    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    logger.info("Loaded %d unlabeled JDs from %s (text_col=%s)", len(df), path, text_col)
    return df


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts into normalized embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def build_position_name_clusters(prototypes_df: pd.DataFrame) -> dict:
    """Map each prototype index to the set of indices sharing the same positionName."""
    clusters = {}
    name_groups = prototypes_df.groupby("positionName").groups
    for name, indices in name_groups.items():
        idx_set = set(int(i) for i in indices)
        for i in idx_set:
            clusters[i] = idx_set
    return clusters


def generate_weak_labels(
    prototypes_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    encoder_name: str = ENCODER_NAME,
    tau_high: float = TAU_HIGH_ICV,
    tau_low: float = TAU_LOW_ICV,
    delta_inter: float = DELTA_INTER,
    delta_intra: float = DELTA_INTRA,
) -> pd.DataFrame:
    """Generate weak labels for unlabeled JDs.

    Labeling logic (3-tier):
      - Positive: s_max > tau_high AND gap > delta AND rule != reject -> label = k*
      - Negative: s_max < tau_low OR rule == reject -> label = 0
      - Abstain:  otherwise -> skip (not included in output)

    Returns a DataFrame with columns: text, weak_label, confidence, rule_triggered
    """
    logger.info("Loading sentence encoder: %s", encoder_name)
    model = SentenceTransformer(encoder_name)

    # Encode prototypes (use the 'all' column which concatenates name + detail)
    proto_texts = prototypes_df["all"].tolist()
    proto_embeddings = encode_texts(model, proto_texts)  # (215, dim)

    # Encode unlabeled JDs
    unlabeled_texts = unlabeled_df["text"].tolist()
    unlabeled_embeddings = encode_texts(model, unlabeled_texts)  # (N, dim)

    # Cosine similarity matrix: (N, 215)
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    sim_matrix = unlabeled_embeddings @ proto_embeddings.T

    # Build positionName clusters for delta selection
    clusters = build_position_name_clusters(prototypes_df)

    # Load rules
    rules = load_rules()

    results = []
    stats = {"positive": 0, "negative": 0, "abstain": 0}

    for i in range(len(unlabeled_texts)):
        text = unlabeled_texts[i]
        sims = sim_matrix[i]  # (215,)

        # Sort similarities descending
        sorted_idx = np.argsort(sims)[::-1]
        s_max = sims[sorted_idx[0]]
        k_star = sorted_idx[0]  # 0-based prototype index
        s_second = sims[sorted_idx[1]]
        k_second = sorted_idx[1]

        # Determine delta: intra-cluster vs inter-cluster
        if k_second in clusters.get(k_star, set()):
            delta = delta_intra
        else:
            delta = delta_inter

        # Apply rules
        rule_label, rule_name = rule_apply(text, prototypes_df, rules)

        # 3-tier labeling
        if rule_label == 0:
            # Rule says reject -> negative class
            label = 0
            conf = float(s_max)
            triggered = rule_name
            stats["negative"] += 1
        elif s_max < tau_low:
            # Low similarity -> negative
            label = 0
            conf = float(s_max)
            triggered = "low_similarity"
            stats["negative"] += 1
        elif s_max > tau_high and (s_max - s_second) > delta and rule_label != 0:
            # High similarity with sufficient gap -> positive
            # Use rule label if available and consistent, else use similarity
            if rule_label is not None and rule_label > 0:
                label = rule_label
                triggered = rule_name
            else:
                label = int(k_star) + 1  # Convert to 1-based class index
                triggered = "similarity"
            conf = float(s_max)
            stats["positive"] += 1
        else:
            # Ambiguous -> abstain
            stats["abstain"] += 1
            continue

        results.append(
            {
                "text": text,
                "weak_label": label,
                "confidence": round(conf, 4),
                "rule_triggered": triggered,
            }
        )

    result_df = pd.DataFrame(results)

    # Print statistics
    total = stats["positive"] + stats["negative"] + stats["abstain"]
    logger.info("=" * 60)
    logger.info("Weak Label Generation Statistics:")
    logger.info("  Total JDs:   %d", total)
    logger.info("  Positive:    %d (%.1f%%)", stats["positive"], 100 * stats["positive"] / max(total, 1))
    logger.info("  Negative:    %d (%.1f%%)", stats["negative"], 100 * stats["negative"] / max(total, 1))
    logger.info("  Abstain:     %d (%.1f%%)", stats["abstain"], 100 * stats["abstain"] / max(total, 1))
    logger.info("  Labeled out: %d", len(result_df))
    logger.info("=" * 60)

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Generate weak labels for JD classification")
    parser.add_argument("--input", type=str, required=True, help="Path to unlabeled JD CSV/XLSX")
    parser.add_argument("--output", type=str, default=os.path.join(DATA_DIR, "train_labeled.csv"),
                        help="Output path for labeled data")
    parser.add_argument("--encoder", type=str, default=ENCODER_NAME, help="Sentence encoder model name")
    parser.add_argument("--tau_high", type=float, default=TAU_HIGH_ICV)
    parser.add_argument("--tau_low", type=float, default=TAU_LOW_ICV)
    parser.add_argument("--delta_inter", type=float, default=DELTA_INTER)
    parser.add_argument("--delta_intra", type=float, default=DELTA_INTRA)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    prototypes_df = load_prototypes()
    unlabeled_df = load_unlabeled(args.input)

    result_df = generate_weak_labels(
        prototypes_df=prototypes_df,
        unlabeled_df=unlabeled_df,
        encoder_name=args.encoder,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        delta_inter=args.delta_inter,
        delta_intra=args.delta_intra,
    )

    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info("Saved %d labeled samples to %s", len(result_df), args.output)


if __name__ == "__main__":
    main()
