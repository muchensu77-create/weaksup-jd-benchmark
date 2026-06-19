"""
Training script for weak-supervision JD classification.

Supports four training strategies:
  - rf_probability:   standard CE on weak labels
  - hybrid_or:        same training as rf_probability; rule-OR fusion at inference
  - neurosymbolic:    joint loss = CE + lambda * KL(rule_prior || model) + gamma * L2
  - bert_finetune:    standard CE (identical to rf_probability, named for clarity)

Usage:
    python train.py --method neurosymbolic --seed 42 --dataset icv
    python train.py --method hybrid_or --seed 42 --train_data data/train_labeled.csv

Python interpreter: E:/Anaconda/python.exe
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    GAMMA,
    LABEL_SMOOTH_EPS,
    LAMBDA_ICV,
    LEARNING_RATE,
    NUM_CLASSES,
    OUTPUT_DIR,
    WARMUP_RATIO,
)
from dataset import JDWeakLabelDataset
from models import BERTClassifier

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = LABEL_SMOOTH_EPS,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Cross-entropy loss with label smoothing."""
    log_probs = F.log_softmax(logits, dim=-1)
    # One-hot with smoothing
    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, eps / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - eps)
    loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
    return loss


def kl_rule_loss(
    logits: torch.Tensor,
    rule_prior: torch.Tensor,
    rule_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between rule prior and model output, masked.

    KL(rule_prior || model_softmax), only for samples where a rule fired.
    """
    if rule_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    model_log_probs = F.log_softmax(logits, dim=-1)
    # rule_prior is already a probability distribution
    rule_prior = rule_prior.clamp(min=1e-8)

    # KL(P || Q) = sum P * log(P / Q) = sum P * (log P - log Q)
    kl = (rule_prior * (rule_prior.log() - model_log_probs)).sum(dim=-1)
    # Apply mask: only count samples where rules fired
    masked_kl = (kl * rule_mask).sum() / rule_mask.sum().clamp(min=1)
    return masked_kl


def l2_regularization(model: nn.Module) -> torch.Tensor:
    """L2 regularization on classifier head weights."""
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "classifier" in name and "weight" in name:
            l2 = l2 + param.pow(2).sum()
    return l2


def compute_loss(
    method: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    rule_prior: torch.Tensor,
    rule_mask: torch.Tensor,
    model: nn.Module,
    lambda_val: float = LAMBDA_ICV,
    gamma: float = GAMMA,
) -> torch.Tensor:
    """Compute training loss based on the selected method.

    Parameters
    ----------
    method : str
        One of: rf_probability, hybrid_or, neurosymbolic, bert_finetune
    logits : (batch, num_classes)
    labels : (batch,)
    rule_prior : (batch, num_classes)
    rule_mask : (batch,)
    model : nn.Module
    lambda_val : float
        Weight for KL rule constraint (neurosymbolic only).
    gamma : float
        Weight for L2 regularization.
    """
    # Base CE loss with label smoothing (used by all methods)
    ce_loss = label_smoothed_ce(logits, labels)

    if method in ("rf_probability", "hybrid_or", "bert_finetune"):
        # Standard CE training
        return ce_loss

    elif method == "neurosymbolic":
        # Joint loss: CE + lambda * KL(rule || model) + gamma * L2
        kl_loss = kl_rule_loss(logits, rule_prior, rule_mask)
        l2_loss = l2_regularization(model)
        total = ce_loss + lambda_val * kl_loss + gamma * l2_loss
        return total

    else:
        raise ValueError(f"Unknown method: {method}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    method: str,
    lambda_val: float,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        rule_prior = batch["rule_prior"].to(device)
        rule_mask = batch["rule_mask"].to(device)

        logits = model(input_ids, attention_mask)
        loss = compute_loss(
            method, logits, labels, rule_prior, rule_mask, model, lambda_val
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                "  Epoch %d | Batch %d/%d | Loss: %.4f | LR: %.2e",
                epoch + 1,
                batch_idx + 1,
                len(dataloader),
                loss.item(),
                scheduler.get_last_lr()[0],
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def train(
    method: str,
    seed: int,
    train_data: str,
    lambda_val: float = LAMBDA_ICV,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    output_dir: str = OUTPUT_DIR,
    tag: str = "",
) -> str:
    """Full training pipeline. Returns path to saved checkpoint.

    Parameters
    ----------
    method : str
        Training method.
    seed : int
        Random seed.
    train_data : str
        Path to training CSV.
    lambda_val : float
        Rule constraint weight (only for neurosymbolic).
    tag : str
        Optional tag appended to checkpoint filename.

    Returns
    -------
    str
        Path to the saved model checkpoint.
    """
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info("Device: %s", device)
    logger.info("Method: %s | Seed: %d | Lambda: %.4f", method, seed, lambda_val)

    # Dataset and dataloader
    compute_prior = method == "neurosymbolic"
    dataset = JDWeakLabelDataset(train_data, compute_rule_prior=compute_prior)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Model
    model = BERTClassifier().to(device)
    logger.info("Model params: %s", f"{model.get_num_params():,}")

    # Optimizer: AdamW with weight decay on non-bias, non-LayerNorm params
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # Linear warmup scheduler
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training: %d epochs, %d steps/epoch", epochs, len(dataloader))
    t0 = time.time()
    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler, device, method, lambda_val, epoch
        )
        logger.info("Epoch %d/%d | Avg Loss: %.4f", epoch + 1, epochs, avg_loss)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds", elapsed)

    # Save checkpoint
    tag_str = f"_{tag}" if tag else ""
    ckpt_name = f"model_{method}_seed{seed}_lam{lambda_val}{tag_str}.pt"
    ckpt_path = os.path.join(output_dir, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": method,
            "seed": seed,
            "lambda_val": lambda_val,
            "epochs": epochs,
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train JD classifier with weak supervision")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["rf_probability", "hybrid_or", "neurosymbolic", "bert_finetune"],
        help="Training strategy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train_data",
        type=str,
        default=os.path.join(DATA_DIR, "train_labeled.csv"),
        help="Path to training CSV",
    )
    parser.add_argument("--lambda_val", type=float, default=LAMBDA_ICV, help="Rule constraint weight")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--dataset", type=str, default="icv", choices=["icv", "lae"], help="Dataset name")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for checkpoint filename")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train(
        method=args.method,
        seed=args.seed,
        train_data=args.train_data,
        lambda_val=args.lambda_val,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
