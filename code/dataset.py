"""
PyTorch Dataset for weak-labeled JD classification.

Loads weak-labeled data, tokenizes with BERT tokenizer, and returns tensors
needed for training (including confidence weights and rule priors).

Python interpreter: E:/Anaconda/python.exe
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, NUM_CLASSES, PROTOTYPE_CSV
from rules import build_rule_prior, load_rules

logger = logging.getLogger(__name__)


class JDWeakLabelDataset(Dataset):
    """Dataset for weakly-labeled job descriptions.

    Each sample returns:
        - input_ids:         (max_length,) token ids
        - attention_mask:    (max_length,) attention mask
        - label:             scalar weak label (0..215)
        - confidence_weight: scalar confidence score from weak labeling
        - rule_mask:         scalar 1 if a rule fired, 0 otherwise
        - rule_prior:        (NUM_CLASSES,) soft rule prior distribution
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = MODEL_NAME,
        max_length: int = 512,
        num_classes: int = NUM_CLASSES,
        prototype_csv: str = PROTOTYPE_CSV,
        compute_rule_prior: bool = True,
    ):
        """
        Parameters
        ----------
        data_path : str
            Path to the weak-labeled CSV (columns: text, weak_label, confidence, rule_triggered).
        tokenizer_name : str
            HuggingFace tokenizer name / path.
        max_length : int
            Maximum token sequence length.
        num_classes : int
            Total number of classes (216).
        prototype_csv : str
            Path to prototype JD CSV (for building rule priors).
        compute_rule_prior : bool
            Whether to compute rule priors (needed for neurosymbolic training).
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_length = max_length

        # Load data
        self.df = pd.read_csv(data_path)
        assert "text" in self.df.columns, "Missing 'text' column"
        assert "weak_label" in self.df.columns, "Missing 'weak_label' column"
        logger.info("Loaded %d samples from %s", len(self.df), data_path)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Rule priors
        self.rule_priors = []
        self.rule_masks = []
        if compute_rule_prior and os.path.exists(prototype_csv):
            proto_df = pd.read_csv(prototype_csv)
            rules = load_rules()
            for _, row in self.df.iterrows():
                prior = build_rule_prior(
                    row["text"], num_classes, proto_df, rules
                )
                if prior is not None:
                    self.rule_priors.append(prior)
                    self.rule_masks.append(1.0)
                else:
                    # Uniform prior when no rule fires
                    self.rule_priors.append(
                        [1.0 / num_classes] * num_classes
                    )
                    self.rule_masks.append(0.0)
        else:
            # No rule priors: uniform + no mask
            for _ in range(len(self.df)):
                self.rule_priors.append([1.0 / num_classes] * num_classes)
                self.rule_masks.append(0.0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["text"])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Label and confidence
        label = int(row["weak_label"])
        confidence = float(row.get("confidence", 1.0))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "label": torch.tensor(label, dtype=torch.long),
            "confidence_weight": torch.tensor(confidence, dtype=torch.float),
            "rule_mask": torch.tensor(self.rule_masks[idx], dtype=torch.float),
            "rule_prior": torch.tensor(self.rule_priors[idx], dtype=torch.float),
        }


class JDTestDataset(Dataset):
    """Simple dataset for test/inference -- no weak labels needed.

    Expects CSV with columns: text, label (ground truth, optional).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = MODEL_NAME,
        max_length: int = 512,
    ):
        self.df = pd.read_csv(data_path)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Identify text column
        text_col = None
        for c in ["text", "all", "content", "positionDetail", "jd_text"]:
            if c in self.df.columns:
                text_col = c
                break
        if text_col is None:
            raise ValueError(f"No text column found in {data_path}")
        self.df = self.df.rename(columns={text_col: "text"})
        self.has_labels = "label" in self.df.columns
        logger.info(
            "Test dataset: %d samples, has_labels=%s", len(self.df), self.has_labels
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["text"])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.has_labels:
            item["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Smoke test: create a tiny dummy dataset
    import tempfile

    dummy_data = pd.DataFrame(
        {
            "text": ["测试岗位描述一", "测试岗位描述二"],
            "weak_label": [1, 0],
            "confidence": [0.85, 0.30],
            "rule_triggered": ["similarity", "low_similarity"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8-sig") as f:
        dummy_data.to_csv(f, index=False)
        tmp_path = f.name

    ds = JDWeakLabelDataset(tmp_path, compute_rule_prior=False)
    sample = ds[0]
    print("Sample keys:", list(sample.keys()))
    print("input_ids shape:", sample["input_ids"].shape)
    print("label:", sample["label"].item())
    os.unlink(tmp_path)
