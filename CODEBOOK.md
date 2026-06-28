# English Codebook for the Job-Description Benchmark

This codebook explains the Chinese-language data files used in the ICV weakly supervised job-description classification benchmark. The text fields remain in Chinese because the study analyzes Chinese recruitment postings in their original language. This document defines file purposes, columns, label conventions, and the relationship between files.

## Label Convention

- Class `0`: out-of-scope / negative class. The job description should not be matched to any ICV prototype.
- Classes `1` to `215`: ICV prototype classes. These labels map to rows in `code/data/jd.csv` using 1-based indexing. For example, label `1` corresponds to the first row of `jd.csv`, label `2` to the second row, and label `215` to the 215th row.

## Core Files

### `code/data/jd.csv`

Expert-curated ICV prototype job descriptions. This file defines the positive label space.

Columns:

- `positionName`: Chinese job-title name of the prototype.
- `positionDetail`: Chinese prototype job-description details, including duties and requirements.
- `all`: concatenation of `positionName` and `positionDetail`; used as the full prototype text for similarity matching.

Rows: 215 prototype rows. Label mapping is row index + 1.

### `code/data/icv_test_gold.csv`

Independent human-labeled gold-test set used for final evaluation.

Columns:

- `text`: Chinese job-description text.
- `label`: human-assigned gold label. `0` means out-of-scope/negative; `1` to `215` refer to prototype rows in `jd.csv`.

Rows: 247 job descriptions over 122 active classes, including 61 negative examples.

Construction note: annotators selected or rejected candidates from sentence-encoder top-10 prototype suggestions. This is disclosed in the manuscript as a limitation because it can favor the similarity-only baseline.

### `code/data/icv_train_conservative_clean.csv`

Leakage-controlled weak-label training set generated under the conservative operating point.

Columns:

- `text`: Chinese job-description text used for weak-label training.
- `weak_label`: automatically generated weak label. `0` means negative; `1` to `215` refer to prototype rows in `jd.csv`.
- `confidence`: cosine-similarity confidence score associated with the weak-label decision.
- `rule_triggered`: source or rule channel that determined the weak label. Typical values include `similarity` and rule identifiers.

Rows: 624 training rows.

Generation parameters:

- `tau_high = 0.70`
- `tau_low = 0.35`
- `delta_inter = 0.08`
- `delta_intra = 0.03`

### `code/data/icv_train_relaxed_clean.csv`

Leakage-controlled weak-label training set generated under the relaxed operating point.

Columns:

- `text`: Chinese job-description text used for weak-label training.
- `weak_label`: automatically generated weak label. `0` means negative; `1` to `215` refer to prototype rows in `jd.csv`.
- `confidence`: cosine-similarity confidence score associated with the weak-label decision.
- `rule_triggered`: source or rule channel that determined the weak label. Typical values include `similarity` and rule identifiers.

Rows: 8,291 training rows.

Generation parameters:

- `tau_high = 0.60`
- `tau_low = 0.33`
- `delta_inter = 0.02`
- `delta_intra = 0.01`

### `code/data/mongodb_data.csv`

Unlabeled public recruitment-posting crawl used as the source pool before weak labeling.

Columns:

- `_id`: source database identifier.
- `keyword`: search keyword used during collection.
- `platform`: source recruitment platform.
- `classify_id`: source-side category identifier, if available.
- `keyword_id`: source-side keyword identifier, if available.
- `positionName`: Chinese job-title name.
- `positionHref`: source URL or link path for the job posting, when available.
- `positionDetail`: Chinese job-description details.
- `education`: education requirement text.
- `workYear`: work-experience requirement text.
- `createTime`: source posting or collection time field.
- `address`: job location text.
- `companyFullName`: company name.
- `companyHref`: source URL or link path for the company, when available.
- `industryLables`: source industry tags; spelling follows the original exported field name.
- `salary`: salary text from the source posting.
- `positionLables`: source position tags; spelling follows the original exported field name.
- `state`: source posting status field.
- `data_json_url`: source API or JSON endpoint URL, when available.

Rows in the local artifact: 103,275 source rows.

## Rule File

### `code/rules.json`

ICV domain rules used in weak-label generation, posterior fusion, and rule-prior construction.

Main fields:

- `positive_rules`: high-precision rules that assign a positive prototype class when keywords or structured patterns match.
- `reject_rules`: rules that assign the negative class `0` for out-of-scope postings.

Rules are intentionally simple and auditable. They use keyword matching and prototype-name lookup rather than hidden model predictions.

## Result Files

### `results/results_gold_icv_clean.json`

Main relaxed-budget trained-model results on `icv_test_gold.csv`, using `icv_train_relaxed_clean.csv`.

### `results/results_gold_icv_cons_clean.json`

Conservative-budget trained-model results on `icv_test_gold.csv`, using `icv_train_conservative_clean.csv`.

### `results/baselines_icv.json`

Training-free baseline results on `icv_test_gold.csv`: majority-negative, rules-only, and similarity-only.

## Leakage Control

The gold-test descriptions in `icv_test_gold.csv` were removed from the unlabeled source pool before weak-label generation. The clean training files therefore do not contain any exact text duplicates from the gold-test file. The audit script `pipeline/scripts/audit_overlap.py` verifies this property.

## Deprecated or Excluded Files

Older leaked or superseded results are not part of the final submitted study. Do not use files marked as deprecated or files from earlier LAE experiments when reproducing the PeerJ submission. The final PeerJ submission is ICV-only.

