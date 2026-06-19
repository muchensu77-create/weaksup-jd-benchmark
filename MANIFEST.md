# Reproducibility Manifest (leakage-controlled)

All trained-model results use input-disjoint train/test (gold texts removed from the unlabeled pool before weak-labeling; 0/247 ICV, 0/120 LAE overlap).

| Result | Role | Train | Train rows | Test | Test rows | Params |
|---|---|---|---|---|---|---|
| `results_gold_icv_clean.json` | ICV main (Table 1) | `icv_train_relaxed_clean.csv` | 8291 | `icv_test_gold.csv` | 247 | relaxed tau0.60/delta0.02 |
| `results_gold_icv_cons_clean.json` | ICV conservative (Table 3) | `icv_train_conservative_clean.csv` | 624 | `icv_test_gold.csv` | 247 | conservative tau0.70/delta0.08 |
| `results_gold_lae_clean.json` | LAE failure (Table 4) | `lae_train_clean.csv` | 1675 | `lae_test_gold.csv` | 120 | relaxed tau0.60/delta0.02 |
| `baselines_icv.json` | baselines (Table 2) | `(training-free)` | — | `icv_test_gold.csv` | 247 | n/a |

## SHA-256 (16-hex)

| File | rows | sha |
|---|---|---|
| `code/data/icv_train_conservative_clean.csv` | 624 | `45e15bd38469f1b4` |
| `code/data/icv_train_relaxed_clean.csv` | 8291 | `fa0c828a39a1a81d` |
| `code/data/lae_train_clean.csv` | 1675 | `0a51298c3de6185f` |
| `code/data/icv_test_gold.csv` | 247 | `12c984c8b33224e5` |
| `code/data/lae_test_gold.csv` | 120 | `314d5c06ca2f6351` |

## Excluded (deprecated/leaked)

- `改稿/experiments/_deprecated/results_gold_*_gpu.json`, `results_gold_icv.json` — superseded leaked-train results, DO NOT use
- `改稿/data/_deprecated/`, `code/output/*.pt`

## Repo URL
[TO BE INSERTED]