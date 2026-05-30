# MDQ — Mastercard Data Quest 2026

**Task:** Identify hidden entrepreneurs — consumer cardholders who exhibit business-like transaction behaviour — using ML on MasterCard transaction data.

---

## Problem Statement

A segment of small business owners and self-employed individuals uses personal consumer cards for commercial activity. These clients generate business-level transaction volumes but are undetected by the bank, resulting in missed revenue from business card products, POS acquiring, and SME lending.

Our model detects these "hidden entrepreneurs" from transaction patterns alone — no manual review required.

---

## Project Structure

```
MDQ/
├── data/
│   └── raw/
│       ├── business_cards_MDQ.parquet    # 25,000 cards · 3M transactions
│       ├── consumer_cards_MDQ.parquet    # 80,000 cards · 10M transactions
│       └── merchants_reference.parquet  # 2,165 merchants with MCC codes
├── reports/                             # All output files (auto-generated)
│   ├── eda_overview.png
│   ├── model_evaluation.png
│   ├── shap_feature_importance.png
│   ├── shap_summary.png
│   ├── segment_profiles.png
│   ├── final_submission.csv
│   └── top_50_candidates_detailed.csv
├── solution.py                          # Full ML pipeline
├── requirements.txt
├── Pipeline.md
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Mans-doc/MDQ.git
cd MDQ
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add datasets
Place the three `.parquet` files into `data/raw/`:
```
data/raw/business_cards_MDQ.parquet
data/raw/consumer_cards_MDQ.parquet
data/raw/merchants_reference.parquet
```

### 4. Run the pipeline
```bash
python solution.py
```

All outputs are saved to `reports/`.

---

## Approach

| Step | Description |
|------|-------------|
| EDA | Distribution analysis across business vs consumer segments |
| Feature Engineering | 20+ behavioral features per card: MCC entropy, HHI concentration, business-hours ratio, channel mix, recurring patterns |
| Modelling | Logistic Regression (baseline) → Random Forest (baseline) → **LightGBM + Optuna** (final) |
| Evaluation | ROC-AUC, PR-AUC, F1 at optimal threshold, Confusion Matrix |
| Explainability | SHAP feature importance and summary plots |
| Output | Scored and ranked list of all 80,000 consumer cards |

---

## Key Features

- **MCC Entropy** — diversity of spending categories per card
- **HHI (Herfindahl Index)** — concentration in top merchant categories
- **Business-hours ratio** — share of transactions during 09:00–18:00 on weekdays
- **Recurring + tokenized patterns** — SaaS and B2B subscription signals
- **Foreign merchant share** — cross-border B2B payment indicator
- **Amount CV** — coefficient of variation in transaction amounts (order irregularity signal)

---

## Output

`reports/final_submission.csv` — all 80,000 consumer cards ranked by `p_business` score (0–1).

`reports/top_50_candidates_detailed.csv` — top 50 hidden entrepreneur profiles with full feature breakdown.

---

## Team

FourSight - Mastercard Data Quest 2026 — AIESEC Kazakhstan
