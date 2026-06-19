"""
Configuration constants for the weak supervision JD classification pipeline.

Python interpreter: E:/Anaconda/python.exe
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

PROTOTYPE_CSV = os.path.join(PROJECT_DIR, "jd.csv")  # 215 prototype JDs
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RULES_JSON = os.path.join(BASE_DIR, "rules.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
MODEL_NAME = "hfl/chinese-bert-wwm-ext"          # classifier backbone
ENCODER_NAME = "shibing624/text2vec-base-chinese"  # sentence encoder for weak labels

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
NUM_CLASSES = 216  # 215 prototype classes (1-indexed) + 1 negative class (index 0)

# ---------------------------------------------------------------------------
# Weak-labeling thresholds
# ---------------------------------------------------------------------------
TAU_HIGH_ICV = 0.72   # similarity above this -> positive candidate
TAU_LOW_ICV = 0.35    # similarity below this -> negative (class 0)
DELTA_INTER = 0.08    # min gap between top-1 and top-2 (different positionName)
DELTA_INTRA = 0.03    # min gap between top-1 and top-2 (same positionName cluster)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
LAMBDA_ICV = 0.1      # rule-constraint weight for neurosymbolic loss
GAMMA = 1e-4          # L2 regularization weight
LEARNING_RATE = 2e-5
EPOCHS = 5
BATCH_SIZE = 32
LABEL_SMOOTH_EPS = 0.1
WARMUP_RATIO = 0.10   # linear warmup fraction

# ---------------------------------------------------------------------------
# Multi-seed experiments
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
METHODS = ["rf_probability", "hybrid_or", "neurosymbolic", "bert_finetune"]

# ---------------------------------------------------------------------------
# Lambda ablation grid
# ---------------------------------------------------------------------------
LAMBDA_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# ---------------------------------------------------------------------------
# Noise injection rates
# ---------------------------------------------------------------------------
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
