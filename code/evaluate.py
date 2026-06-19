"""
Evaluation script for JD classification models.

Loads a trained checkpoint, runs inference on test data, computes metrics.
For hybrid_or: applies rule-priority fusion at inference time.

Usage:
    python evaluate.py --checkpoint output/model_neurosymbolic_seed42_lam0.1.pt \
                       --test_data data/test.csv --method neurosymbolic

Python interpreter: E:/Anaconda/python.exe
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from config import DATA_DIR, NUM_CLASSES, OUTPUT_DIR, PROTOTYPE_CSV
from dataset import JDTestDataset
from models import BERTClassifier
from rules import load_rules, rule_apply

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> BERTClassifier:
    """Load model from checkpoint."""
    model = BERTClassifier()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint: %s (method=%s, seed=%s)",
        checkpoint_path,
        ckpt.get("method", "?"),
        ckpt.get("seed", "?"),
    )
    return model


def predict(
    model: BERTClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference and return predicted logits, labels (if available).

    Returns
    -------
    all_logits : np.ndarray, shape (N, num_classes)
    all_labels : np.ndarray or None, shape (N,)
    all_texts  : list[str] (empty if not available)
    """
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
            if "label" in batch:
                all_labels.append(batch["label"].numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return all_logits, all_labels


def hybrid_or_fusion(
    logits: np.ndarray,
    texts: list[str],
    prototypes_df: pd.DataFrame,
    rules: dict,
) -> np.ndarray:
    """Apply rule-OR fusion: if a rule fires, override model prediction.

    For each sample:
      - If a positive rule fires -> use rule label
      - If a reject rule fires -> use class 0
      - Otherwise -> use argmax of model logits
    """
    preds = np.argmax(logits, axis=1)
    n_overridden = 0

    for i, text in enumerate(texts):
        rule_label, rule_name = rule_apply(text, prototypes_df, rules)
        if rule_label is not None:
            preds[i] = rule_label
            n_overridden += 1

    logger.info(
        "Hybrid-OR fusion: %d/%d predictions overridden by rules",
        n_overridden,
        len(preds),
    )
    return preds


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Compute evaluation metrics.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, positive_macro_f1, negative_f1, top5_accuracy
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-F1 (all classes)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Positive-Macro-F1: macro F1 over classes 1..215 only
    positive_mask = y_true > 0
    if positive_mask.sum() > 0:
        positive_macro_f1 = f1_score(
            y_true[positive_mask],
            y_pred[positive_mask],
            average="macro",
            zero_division=0,
        )
    else:
        positive_macro_f1 = 0.0

    # Negative-F1: F1 for class 0 (binary: is it class 0 or not?)
    y_true_neg = (y_true == 0).astype(int)
    y_pred_neg = (y_pred == 0).astype(int)
    negative_f1 = f1_score(y_true_neg, y_pred_neg, average="binary", zero_division=0)

    # Top-5 Accuracy
    top5_preds = np.argsort(logits, axis=1)[:, -5:]  # top-5 indices per sample
    top5_correct = np.array(
        [y_true[i] in top5_preds[i] for i in range(len(y_true))]
    )
    top5_accuracy = top5_correct.mean()

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "positive_macro_f1": round(float(positive_macro_f1), 4),
        "negative_f1": round(float(negative_f1), 4),
        "top5_accuracy": round(float(top5_accuracy), 4),
    }
    return metrics


def evaluate(
    checkpoint_path: str,
    test_data: str,
    method: str,
    output_path: str = None,
) -> dict:
    """Full evaluation pipeline.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint.
    test_data : str
        Path to test CSV.
    method : str
        Method name (for hybrid_or fusion).
    output_path : str, optional
        Path to save results JSON.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # Load model
    model = load_model(checkpoint_path, device)

    # Load test dataset
    test_dataset = JDTestDataset(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    if not test_dataset.has_labels:
        logger.warning("Test dataset has no labels -- cannot compute metrics")
        return {}

    # Predict
    logits, y_true = predict(model, test_loader, device)

    # Get predictions
    if method == "hybrid_or":
        # Load prototypes and rules for fusion
        prototypes_df = pd.read_csv(PROTOTYPE_CSV)
        rules = load_rules()
        texts = test_dataset.df["text"].tolist()
        y_pred = hybrid_or_fusion(logits, texts, prototypes_df, rules)
    else:
        y_pred = np.argmax(logits, axis=1)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, logits)
    metrics["method"] = method
    metrics["checkpoint"] = os.path.basename(checkpoint_path)
    metrics["n_samples"] = len(y_true)

    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info("  %-20s: %s", k, v)
    logger.info("=" * 60)

    # Save results
    if output_path is None:
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"eval_{ckpt_stem}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", output_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate JD classification model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument(
        "--test_data",
        type=str,
        default=os.path.join(DATA_DIR, "test.csv"),
        help="Test CSV path",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["rf_probability", "hybrid_or", "neurosymbolic", "bert_finetune"],
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    evaluate(
        checkpoint_path=args.checkpoint,
        test_data=args.test_data,
        method=args.method,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
