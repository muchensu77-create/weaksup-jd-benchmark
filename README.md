# Weakly Supervised Job-Description Classification Benchmark

This repository contains the data, code, result files, and documentation for the PeerJ Computer Science manuscript:

**When do domain rules help weakly supervised job-description classification? A leakage-controlled, reproducible study with simple baselines in a large label space**

The project evaluates whether domain rules improve weakly supervised job-description (JD) classification in a large prototype-matching label space. The final submitted study focuses on Intelligent Connected Vehicles (ICV). It compares rule-free weak-label training, posterior rule fusion, rule-constrained joint training, a plain BERT fine-tuning reference, and training-free baselines.

## Dataset Overview

The benchmark uses Chinese-language job-description text because the source task is Chinese ICV recruitment analysis. English documentation is provided here and in `CODEBOOK.md` so editors, reviewers, and readers can understand the structure and meaning of each data file.

Core data files:

- `code/data/jd.csv`: 215 expert-curated ICV prototype job descriptions. Prototype class labels are 1-based: row 1 maps to class label `1`, row 2 maps to class label `2`, and so on through `215`.
- `code/data/icv_test_gold.csv`: 247 independently human-labeled gold-test job descriptions over 122 active classes. Label `0` means out-of-scope/negative; labels `1` to `215` refer to prototype rows in `jd.csv`.
- `code/data/icv_train_conservative_clean.csv`: 624 leakage-controlled weak-label training rows generated with conservative thresholds.
- `code/data/icv_train_relaxed_clean.csv`: 8,291 leakage-controlled weak-label training rows generated with relaxed thresholds.
- `code/data/mongodb_data.csv`: public recruitment-platform crawl used as the unlabeled source pool.
- `code/rules.json`: ICV domain rules used for weak labeling and posterior fusion.

Gold-test texts are removed from the unlabeled pool before weak-label generation. Exact-match audits confirm zero input overlap between the clean training files and the gold-test file.

## Code Overview

Main code modules:

- `code/config.py`: paths, model names, thresholds, hyperparameters, seed list, and method names.
- `code/dataset.py`: PyTorch datasets for weak-label training and gold-test evaluation.
- `code/models.py`: BERT-based classifier with a linear 216-class output head.
- `code/rules.py`: rule loading, rule matching, and rule-prior construction.
- `code/evaluate.py`: checkpoint evaluation and HybridOR posterior-fusion inference.
- `pipeline/scripts/gen_weak_labels.py`: weak-label generation with gold-text exclusion.
- `pipeline/scripts/audit_overlap.py`: train/test input-overlap audit.
- `pipeline/scripts/compute_baselines.py`: training-free majority-negative, rules-only, and similarity-only baselines.
- `pipeline/scripts/RUN_CLEAN.py`: clean train/evaluate driver for the reported experiments.
- `pipeline/scripts/run_gold_eval.py`: per-method/per-seed gold-test evaluation driver.

## Requirements

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model code uses PyTorch, Transformers, pandas, scikit-learn, NumPy, tqdm, and sentence-transformers/text2vec-compatible HuggingFace models. The classifier backbone is `hfl/chinese-bert-wwm-ext`; the similarity encoder is `shibing624/text2vec-base-chinese`. Models are downloaded from HuggingFace on first use.

For GPU runs, install the PyTorch build appropriate for your CUDA version. CPU execution is possible for lightweight inspection and baselines but is slow for full BERT training.

## Usage

Verify the leakage-control audit:

```bash
python pipeline/scripts/audit_overlap.py
```

Expected result: zero exact text overlap between each clean weak-label training file and `icv_test_gold.csv`.

Run training-free baselines:

```bash
python pipeline/scripts/compute_baselines.py
```

This writes `results/baselines_icv.json`.

Run the clean model experiments:

```bash
python pipeline/scripts/RUN_CLEAN.py
```

This trains/evaluates the reported methods over seeds 42, 123, and 456, and writes clean result JSON files under `results/`.

Regenerate relaxed weak labels from the unlabeled pool:

```bash
python pipeline/scripts/gen_weak_labels.py \
  --domain icv_v2 \
  --max_unlabeled 0 \
  --tau_high 0.60 \
  --delta_inter 0.02 \
  --delta_intra 0.01
```

The script excludes gold-test texts before weak-label generation.

## Method Summary

The benchmark formulates JD classification as prototype-instance matching. A job description is mapped either to one of 215 ICV prototype classes or to class `0`, the out-of-scope/negative class. Weak labels are generated from sentence-encoder prototype similarity plus domain rules. The reported experiments compare:

- `rf_probability`: rule-free classifier trained on weak labels.
- `hybrid_or`: posterior fusion where rules override model predictions at inference time.
- `neurosymbolic`: rule-constrained joint training with a KL rule-prior term.
- `bert_finetune`: plain BERT fine-tuning reference on weak labels.
- Training-free baselines: majority-negative, rules-only, and similarity-only.

## Results and Manifest

Reported result files are under `results/`. `MANIFEST.md` maps each reported result to the exact train/test CSV files, row counts, and SHA-256 hashes. The key result is that rule integration shows no clear advantage over the rule-free weak-label baseline, while a training-free similarity-only baseline is stronger than all trained models under this candidate-assisted gold-test protocol.

## Language and Codebook

The job-description text fields are in Chinese because they are the original public recruitment texts analyzed by the study. File purposes, column meanings, label conventions, and threshold settings are documented in English in `CODEBOOK.md`.

## Citation

If you use this repository, please cite the associated PeerJ Computer Science manuscript:

Zheng, H., Su, M., Ruan, Y., Li, X., & Li, M. *When do domain rules help weakly supervised job-description classification? A leakage-controlled, reproducible study with simple baselines in a large label space.*

## License and Contributions

This repository is provided as a reproducibility artifact for academic review and research use. Please open an issue for questions about reproducing the reported results.

