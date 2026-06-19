# A Leakage-Controlled Reproducible Benchmark for Weakly Supervised Job-Description Classification

Artifacts for the paper *"When Do Domain Rules Help Weakly Supervised Job-Description
Classification?"* (Zheng, Su, Ruan, Li, Li). Every number in the paper is reproducible
from these files. All trained-model results are **input-disjoint** from the gold test sets
(gold texts are removed from the unlabeled pool before weak-labeling; verified by a
zero-overlap audit).

## Layout
- `code/` — pipeline (BERT-wwm classifier, weak-label rules, evaluation).
- `code/data/` — prototypes (`jd.csv`, `lae_prototypes.csv`), rules, **gold test sets**
  (`icv_test_gold.csv` 247, `lae_test_gold.csv` 120), **clean train sets**
  (`icv_train_{conservative,relaxed}_clean.csv`, `lae_train_clean.csv`), and the unlabeled
  pools (`mongodb_data.csv`, `lae_raw.csv`).
- `pipeline/scripts/` — weak-label generation (`gen_weak_labels.py`, with `exclude_gold`
  de-duplication), leakage audit (`audit_overlap.py`), training-free baselines
  (`compute_baselines.py`), train+eval driver (`run_gold_eval.py`, `RUN_CLEAN.py`).
- `results/` — per-seed result JSONs cited in the paper.
- `MANIFEST.md` — maps each result file to its exact train/test CSV, row counts, SHA-256.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build

# 1) verify zero train/test leakage
python pipeline/scripts/audit_overlap.py            # -> ZERO LEAKAGE

# 2) training-free baselines (incl. similarity-only)
python pipeline/scripts/compute_baselines.py        # -> results/baselines_icv.json

# 3) train + evaluate all methods on the clean (input-disjoint) sets
python pipeline/scripts/RUN_CLEAN.py                # -> results/results_gold_*_clean.json

# (optional) regenerate the clean weak-label train sets from the unlabeled pool:
python pipeline/scripts/gen_weak_labels.py --domain icv_v2 --max_unlabeled 0 \
    --tau_high 0.60 --delta_inter 0.02 --delta_intra 0.01   # gold excluded automatically
```

## Notes
- Models: `hfl/chinese-bert-wwm-ext` (classifier), `shibing624/text2vec-base-chinese`
  (similarity encoder); downloaded from HuggingFace on first run.
- Unlabeled JDs were crawled from public recruitment platforms; no proprietary data.
- Absolute performance is modest by design; this is an honest negative/benchmark study.
