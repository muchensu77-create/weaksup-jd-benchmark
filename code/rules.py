"""
Rule functions for ICV (Intelligent Connected Vehicle) domain.

Hard-constraint rules based on keyword matching.  Rules are loaded from
rules.json so they are easy to extend without touching code.

Python interpreter: E:/Anaconda/python.exe
"""

import json
import logging
import os
from typing import Optional, Tuple

import pandas as pd

from config import RULES_JSON

logger = logging.getLogger(__name__)


def load_rules(path: str = RULES_JSON) -> dict:
    """Load rule definitions from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    logger.info(
        "Loaded %d positive rules and %d reject rules",
        len(rules.get("positive_rules", [])),
        len(rules.get("reject_rules", [])),
    )
    return rules


def _match_keywords(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _find_prototype_index(
    target_names: list[str], prototypes_df: pd.DataFrame
) -> Optional[int]:
    """Return 1-based prototype index for the first matching positionName.

    Prototype classes are 1..215; class 0 is the negative class.
    Returns None if no match is found in the prototype table.
    """
    for name in target_names:
        mask = prototypes_df["positionName"].str.contains(name, na=False)
        matches = prototypes_df.index[mask]
        if len(matches) > 0:
            # Return 1-based index (row 0 in CSV -> class 1)
            return int(matches[0]) + 1
    return None


def rule_apply(
    text: str, prototypes_df: pd.DataFrame, rules: Optional[dict] = None
) -> Tuple[Optional[int], Optional[str]]:
    """Apply hard-constraint rules to a single JD text.

    Parameters
    ----------
    text : str
        The job description text.
    prototypes_df : pd.DataFrame
        The prototype DataFrame (215 rows with positionName column).
    rules : dict, optional
        Pre-loaded rules dict.  If None, loads from RULES_JSON.

    Returns
    -------
    (label_or_none, rule_name)
        - (int, str) if a positive rule matched -> (class_index, rule_name)
        - (0, str) if a reject rule matched -> (0, rule_name)  (negative)
        - (None, None) if no rule matched
    """
    if rules is None:
        rules = load_rules()

    # Check reject rules first (higher priority to filter out-of-domain)
    for rule in rules.get("reject_rules", []):
        if _match_keywords(text, rule["keywords"]):
            return 0, rule["name"]

    # Check positive rules
    for rule in rules.get("positive_rules", []):
        if _match_keywords(text, rule["keywords"]):
            idx = _find_prototype_index(
                rule.get("target_positionName", []), prototypes_df
            )
            if idx is not None:
                return idx, rule["name"]

    return None, None


def build_rule_prior(
    text: str,
    num_classes: int,
    prototypes_df: pd.DataFrame,
    rules: Optional[dict] = None,
    confidence: float = 0.9,
) -> Optional[list]:
    """Build a soft rule-prior distribution over classes for a single text.

    Returns a probability vector of length num_classes where:
      - If a positive rule fires: concentration on the matched class
      - If a reject rule fires: uniform over non-negative classes (discourage class 0)
        Actually: put mass on class 0
      - If no rule fires: return None (no prior)

    This is used in the neurosymbolic KL loss term.
    """
    label, rule_name = rule_apply(text, prototypes_df, rules)
    if label is None:
        return None

    prior = [(1.0 - confidence) / (num_classes - 1)] * num_classes
    prior[label] = confidence
    return prior


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke test
    rules = load_rules()
    print(f"Positive rules: {len(rules['positive_rules'])}")
    print(f"Reject rules:   {len(rules['reject_rules'])}")

    test_texts = [
        "负责ISO 26262功能安全开发流程",
        "CCAR-145部维修管理",
        "负责Python后端开发",
        "AUTOSAR Classic平台BSW集成",
    ]
    dummy_df = pd.DataFrame(
        {
            "positionName": [
                "功能安全工程师",
                "AUTOSAR软件工程师",
                "嵌入式软件工程师",
            ]
        }
    )
    for t in test_texts:
        label, name = rule_apply(t, dummy_df, rules)
        print(f"  Text: {t[:40]}...  -> label={label}, rule={name}")
