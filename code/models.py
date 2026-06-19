"""
Model definitions for JD classification.

BERTClassifier: chinese-bert-wwm-ext backbone + linear classification head.

Python interpreter: E:/Anaconda/python.exe
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from config import MODEL_NAME, NUM_CLASSES

logger = logging.getLogger(__name__)


class BERTClassifier(nn.Module):
    """BERT-based classifier with a single linear head.

    Architecture:
        chinese-bert-wwm-ext -> [CLS] pooling -> Dropout -> Linear(768, 216)

    Forward returns raw logits (216-dim).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        logger.info(
            "BERTClassifier: backbone=%s, hidden=%d, num_classes=%d",
            model_name,
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids : (batch, seq_len)
        attention_mask : (batch, seq_len)

        Returns
        -------
        logits : (batch, num_classes)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch, num_classes)
        return logits

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = BERTClassifier()
    print(f"Total trainable parameters: {model.get_num_params():,}")

    # Quick forward pass test with dummy input
    dummy_ids = torch.randint(0, 1000, (2, 32))
    dummy_mask = torch.ones(2, 32, dtype=torch.long)
    logits = model(dummy_ids, dummy_mask)
    print(f"Logits shape: {logits.shape}")  # (2, 216)
