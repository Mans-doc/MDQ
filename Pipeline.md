# Pipeline Documentation

Full description of the ML pipeline in `solution.py`.

---

## Overview

```
Raw Parquet Files
      │
      ▼
 STEP 1: Data Loading          load_data()
      │
      ▼
 STEP 2: Preprocessing         preprocess()
      │
      ▼
 STEP 3: EDA                   run_eda()
      │
      ▼
 STEP 4: Feature Engineering   compute_biz_mcc_set() → build_features()
      │
      ▼
 STEP 5: Model Training        prepare_train_test() → train_all_models()
      │                        ├── train_logistic_regression()
      │                        ├── train_random_forest()
      │                        └── train_lgbm_optuna()
      ▼
 STEP 6: Evaluation            evaluate_models()
      │
      ▼
 STEP 7: SHAP Analysis         run_shap()
      │
      ▼
 STEP 8: Consumer Scoring      score_consumers()
      │
      ▼
 STEP 9: Segment Profiling     segment_profiles()
      │
      ▼
   reports/
```

---

## Step-by-step

### STEP 1 — Data Loading (`load_data`)

Loads three parquet files. Validates that all files exist before proceeding — exits with a clear message if a file is missing.

Assigns labels:
- `business_cards_MDQ.parquet` → `segment = 1`
- `consumer_cards_MDQ.parquet` → `segment = 0`

---

### STEP 2 — Preprocessing (`preprocess`)

- Concatenates business and consumer datasets
- Downcasts `float64 → float32`, `int64 → int32` to reduce memory usage
- Parses `transaction_date` and `transaction_timestamp`
- Extracts `hour`, `weekday`, `month`, `week` from timestamps
- Left-joins `merchants_reference` on `merchant_id` to add `merchant_country` and `recurring_capable`

---

### STEP 3 — EDA (`run_eda`)

Generates `reports/eda_overview.png` with 6 panels:

| Panel | What it shows |
|-------|--------------|
| A | Transactions per card distribution |
| B | log(average transaction amount) |
| C | Online vs POS channel split |
| D | Top-15 MCC codes for business cards |
| E | Transaction volume by hour of day |
| F | Transaction volume by day of week |

---

### STEP 4 — Feature Engineering

#### `compute_biz_mcc_set()`

Identifies MCC codes that appear **3× more frequently** in business card transactions than in consumer card transactions. These codes form `BIZ_MCC_SET` — a data-driven signal used in feature construction.

> **Note on data leakage:** `BIZ_MCC_SET` is computed at the population level (across all cards), not at the individual card level. This means no information from held-out test cards leaks into the feature itself. The set is used as a fixed reference dictionary, not derived from any card's individual label.

#### `build_features()`

Aggregates all transactions to card level. One row = one card.

| Feature | Description | Signal |
|---------|-------------|--------|
| `total_txn_count` | Total number of transactions | Activity volume |
| `total_amount` | Total spend | Revenue proxy |
| `avg_amount` | Mean transaction amount | Ticket size |
| `median_amount` | Median transaction amount | Robust ticket size |
| `std_amount` | Std of transaction amounts | Spend variability |
| `amount_cv` | Coefficient of variation (std/mean) | Order irregularity |
| `biz_hours_share` | Share of txns during 09–18 Mon–Fri | B2B activity timing |
| `weekend_share` | Share of txns on weekends | Consumer vs business timing |
| `night_share` | Share of txns between 22–06 | Consumer leisure signal |
| `online_share` | Share of online channel txns | Business prefers online |
| `foreign_merchant_share` | Share of txns at non-KZ merchants | Cross-border B2B |
| `recurring_share` | Share of recurring transactions | SaaS / subscription signal |
| `recurring_capable_share` | Share of merchants that support recurring | B2B merchant profile |
| `unique_mcc` | Number of distinct MCC codes | Category diversity |
| `biz_mcc_share` | Share of txns in business MCC codes | Direct B2B signal |
| `unique_merchants` | Number of distinct merchants | Supplier diversity |
| `active_months` | Months with at least one transaction | Activity span |
| `mcc_entropy` | Shannon entropy of MCC distribution | Low = focused niche |
| `hhi_merchants` | Herfindahl index of merchant distribution | High = concentrated |
| `txn_per_month` | Transactions normalised by active months | Comparable activity rate |
| `amount_per_month` | Amount normalised by active months | Comparable volume |
| `merch_per_month` | Merchants normalised by active months | Supplier expansion rate |

---

### STEP 5 — Model Training (`train_all_models`)

**Train/test split:** 80/20 stratified by `segment`.

Three models trained and compared:

#### Logistic Regression (baseline)
- `StandardScaler` + `LogisticRegression(C=0.1, class_weight='balanced')`
- Purpose: linear baseline, fast, interpretable

#### Random Forest (baseline)
- `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`
- Purpose: non-linear baseline without hyperparameter tuning

#### LightGBM + Optuna (final model)
- Optuna runs **30 trials** of Bayesian hyperparameter search
- Each trial evaluated with **3-fold stratified CV** on the training set
- Parameters tuned: `n_estimators`, `num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- Best parameters are used to train the final model on the full training set

---

### STEP 6 — Evaluation (`evaluate_models`)

Metrics computed on the held-out test set:

| Metric | Description |
|--------|-------------|
| ROC-AUC | Overall discrimination ability |
| PR-AUC (Avg Precision) | Performance under class imbalance |
| F1 score | Harmonic mean of precision and recall |
| Optimal threshold | Threshold that maximises F1 on test set |
| Confusion Matrix | TP, TN, FP, FN counts at optimal threshold |

Output: `reports/model_evaluation.png`

---

### STEP 7 — SHAP Analysis (`run_shap`)

Uses `shap.TreeExplainer` on the LightGBM model.

- Sample: up to 2,000 test cards
- Generates `shap_feature_importance.png` (bar chart — mean absolute SHAP)
- Generates `shap_summary.png` (beeswarm — direction and magnitude per feature)

SHAP explains **why** a specific card received a high score, making the model auditable and explainable to business stakeholders.

---

### STEP 8 — Consumer Scoring (`score_consumers`)

All 80,000 consumer cards are scored with `predict_proba`.

Output columns in `final_submission.csv`:

| Column | Description |
|--------|-------------|
| `card_number` | Card identifier |
| `p_business` | Probability of hidden business behaviour (0–1) |

Top 50 cards are exported with full feature profile to `top_50_candidates_detailed.csv`.

---

### STEP 9 — Segment Profiling (`segment_profiles`)

Cards are split into three groups using the optimal threshold:

| Group | Condition |
|-------|-----------|
| Real Business | `segment == 1` (known business cards) |
| Hidden Entrepreneur | consumer card with `p_business ≥ threshold` |
| Regular Consumer | consumer card with `p_business < threshold` |

Median feature values are compared across groups.
Output: `reports/segment_profiles.png`

---

## Reproducibility

- `RANDOM_STATE = 42` used in all stochastic components
- All paths are relative to `solution.py` location — runs on any OS
- All outputs are deterministic given the same input data and seed
