# =============================================================================
# Mastercard Data Quest 2026
# Выявление «скрытых предпринимателей» в транзакционных данных физических лиц
# =============================================================================
# Автор: команда
# Дата:  2026-05-28
# Описание: Полный воспроизводимый ML-пайплайн:
#   1. Загрузка и объединение данных
#   2. EDA
#   3. Feature Engineering (агрегации на уровне карты)
#   4. Обучение моделей (Baseline: LogisticRegression + LightGBM с Optuna)
#   5. Оценка метрик (ROC-AUC, PR-AUC, F1, Confusion Matrix)
#   6. Интерпретация SHAP
#   7. Применение модели к consumer-сегменту и анализ кандидатов
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay,
    f1_score, precision_score, recall_score
)

# ──────────────────────────────────────────────────
# 0. НАСТРОЙКИ ПУТЕЙ
# ──────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\admin\Downloads\MDQ")
OUT_DIR  = Path(r"C:\Users\admin\MDQ\outputs")
OUT_DIR.mkdir(exist_ok=True)

BIZ_PATH   = DATA_DIR / "business_cards_MDQ.parquet"
CONS_PATH  = DATA_DIR / "consumer_cards_MDQ.parquet"
MERCH_PATH = DATA_DIR / "merchants_reference.parquet"

print("DATA LOADING...")

# ──────────────────────────────────────────────────
# 1. ЗАГРУЗКА
# ──────────────────────────────────────────────────
df_biz   = pd.read_parquet(BIZ_PATH)
df_cons  = pd.read_parquet(CONS_PATH)
df_merch = pd.read_parquet(MERCH_PATH)

df_biz["segment"]  = 1   # бизнес-поведение (таргет = 1)
df_cons["segment"] = 0   # потребительское поведение (таргет = 0)

print(f"  Business cards:  {len(df_biz):>10,} транзакций, {df_biz['card_number'].nunique():>8,} карт")
print(f"  Consumer cards:  {len(df_cons):>10,} транзакций, {df_cons['card_number'].nunique():>8,} карт")
print(f"  Merchants ref:   {len(df_merch):>10,} мерчантов")
print(f"  Колонки (biz):   {df_biz.columns.tolist()}")
print(f"  Колонки (cons):  {df_cons.columns.tolist()}")
print(f"  Колонки (merch): {df_merch.columns.tolist()}")

# ──────────────────────────────────────────────────
# 2. ОБЪЕДИНЕНИЕ И ДЖОИН С МЕРЧАНТАМИ
# ──────────────────────────────────────────────────
print("\nMERGING DATASETS...")

df = pd.concat([df_biz, df_cons], ignore_index=True)

# Оптимизация памяти
for col in df.select_dtypes("float64").columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes("int64").columns:
    df[col] = df[col].astype("int32")

# Парсим дату/время
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
if "transaction_timestamp" in df.columns:
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df["hour"]    = df["transaction_timestamp"].dt.hour
    df["weekday"] = df["transaction_timestamp"].dt.dayofweek  # 0=пн, 6=вс
elif "transaction_date" in df.columns:
    df["hour"]    = 12  # нет timestamp — ставим нейтральное значение
    df["weekday"] = df["transaction_date"].dt.dayofweek

df["month"] = df["transaction_date"].dt.to_period("M")

# Джоин с мерчантами
df = df.merge(
    df_merch[["merchant_id", "mcc", "merchant_country", "recurring_capable"]],
    on="merchant_id", how="left", suffixes=("", "_merch")
)
# Если mcc дублируется — оставляем из мерчантов (более надёжный источник)
if "mcc_merch" in df.columns:
    df["mcc"] = df["mcc_merch"].fillna(df["mcc"])
    df.drop(columns=["mcc_merch"], inplace=True)

print(f"  Объединённый датасет: {len(df):,} транзакций")

# ──────────────────────────────────────────────────
# 3. EDA — БЫСТРЫЙ ВИЗУАЛЬНЫЙ АНАЛИЗ
# ──────────────────────────────────────────────────
print("\nRUNNING EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EDA: Business vs Consumer — ключевые распределения", fontsize=14, fontweight="bold")

seg_labels = {1: "Business", 0: "Consumer"}
colors     = {1: "#2196F3", 0: "#FF9800"}

# (A) Транзакций на карту
txn_per_card = df.groupby(["card_number", "segment"]).size().reset_index(name="txn_count")
for seg, grp in txn_per_card.groupby("segment"):
    axes[0, 0].hist(grp["txn_count"].clip(upper=grp["txn_count"].quantile(0.99)),
                    bins=50, alpha=0.6, label=seg_labels[seg], color=colors[seg], density=True)
axes[0, 0].set_title("Транзакций на карту (до 99-перц.)")
axes[0, 0].set_xlabel("Кол-во транзакций")
axes[0, 0].legend()

# (B) Средний чек
amount_col = "transaction_amount_kzt" if "transaction_amount_kzt" in df.columns else df.select_dtypes("float32").columns[0]
avg_amt = df.groupby(["card_number", "segment"])[amount_col].mean().reset_index(name="avg_amt")
for seg, grp in avg_amt.groupby("segment"):
    axes[0, 1].hist(np.log1p(grp["avg_amt"].clip(lower=0)),
                    bins=50, alpha=0.6, label=seg_labels[seg], color=colors[seg], density=True)
axes[0, 1].set_title("log(Средний чек + 1)")
axes[0, 1].set_xlabel("log(avg_amount)")
axes[0, 1].legend()

# (C) Online vs Offline
if "channel" in df.columns:
    ch = df.groupby(["segment", "channel"]).size().unstack(fill_value=0)
    ch.index = [seg_labels[i] for i in ch.index]
    ch.plot(kind="bar", ax=axes[0, 2], color=["#4CAF50", "#F44336"])
    axes[0, 2].set_title("Online vs Offline транзакции")
    axes[0, 2].set_xlabel("")
    axes[0, 2].tick_params(axis="x", rotation=0)

# (D) Топ-15 MCC у Business
top_mcc_biz = df[df["segment"] == 1]["mcc"].value_counts().head(15)
axes[1, 0].barh(top_mcc_biz.index.astype(str), top_mcc_biz.values, color="#2196F3")
axes[1, 0].set_title("Топ-15 MCC (Business)")
axes[1, 0].set_xlabel("Кол-во транзакций")
axes[1, 0].invert_yaxis()

# (E) Час транзакции
hour_dist = df.groupby(["hour", "segment"]).size().reset_index(name="cnt")
for seg, grp in hour_dist.groupby("segment"):
    axes[1, 1].plot(grp["hour"], grp["cnt"], label=seg_labels[seg], color=colors[seg])
axes[1, 1].set_title("Транзакции по часам суток")
axes[1, 1].set_xlabel("Час")
axes[1, 1].set_ylabel("Кол-во транзакций")
axes[1, 1].legend()

# (F) День недели
weekday_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
wd_dist = df.groupby(["weekday", "segment"]).size().reset_index(name="cnt")
for seg, grp in wd_dist.groupby("segment"):
    axes[1, 2].plot(grp["weekday"], grp["cnt"], marker="o", label=seg_labels[seg], color=colors[seg])
axes[1, 2].set_title("Транзакции по дням недели")
axes[1, 2].set_xticks(range(7))
axes[1, 2].set_xticklabels(weekday_names)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  EDA saved to {OUT_DIR / 'eda_overview.png'}")

# ──────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────────
print("\nFEATURE ENGINEERING...")

amount_col = "transaction_amount_kzt" if "transaction_amount_kzt" in df.columns else \
             [c for c in df.columns if "amount" in c.lower()][0]

# Вспомогательные бинарные колонки
df["is_online"]    = (df["channel"] == "online").astype("int8") if "channel" in df.columns else 0
df["is_recurring"] = df["Is_recurring"].astype("int8") if "Is_recurring" in df.columns else 0
df["is_foreign"]   = (df["country"] != "KZ").astype("int8") if "country" in df.columns else 0  # примерная страна клиента

# Бизнес-часы: 9-18 в будни
df["is_biz_hours"] = ((df["hour"] >= 9) & (df["hour"] < 18) & (df["weekday"] < 5)).astype("int8")
df["is_weekend"]   = (df["weekday"] >= 5).astype("int8")
df["is_night"]     = ((df["hour"] < 6) | (df["hour"] >= 22)).astype("int8")

# Топ «бизнес-MCC» — берём MCC, которые в 3+ раза чаще встречаются у business, чем у consumer
mcc_biz  = df[df["segment"] == 1]["mcc"].value_counts(normalize=True)
mcc_cons = df[df["segment"] == 0]["mcc"].value_counts(normalize=True)
mcc_ratio = (mcc_biz / (mcc_cons + 1e-9)).fillna(0)
BIZ_MCC_SET = set(mcc_ratio[mcc_ratio >= 3].index)
df["is_biz_mcc"] = df["mcc"].isin(BIZ_MCC_SET).astype("int8")

df["recurring_capable"] = df["recurring_capable"].fillna(0).astype("int8")

# ── Агрегации на уровне карты ──────────────────────
def entropy(series):
    vc = series.value_counts(normalize=True)
    return -(vc * np.log(vc + 1e-9)).sum()

def hhi(series):
    vc = series.value_counts(normalize=True)
    return (vc ** 2).sum()

print("  Агрегация транзакций по картам (может занять 1-2 мин)...")

card_feats = df.groupby("card_number").agg(
    segment          = ("segment", "first"),
    # Объём и интенсивность
    total_txn_count  = (amount_col, "count"),
    total_amount     = (amount_col, "sum"),
    avg_amount       = (amount_col, "mean"),
    median_amount    = (amount_col, "median"),
    std_amount       = (amount_col, "std"),
    max_amount       = (amount_col, "max"),
    # Временные паттерны
    biz_hours_share  = ("is_biz_hours", "mean"),
    weekend_share    = ("is_weekend", "mean"),
    night_share      = ("is_night", "mean"),
    # Онлайн / иностранные
    online_share     = ("is_online", "mean"),
    foreign_share    = ("is_foreign", "mean"),
    # Регулярность
    recurring_share  = ("is_recurring", "mean"),
    recurring_capable_share = ("recurring_capable", "mean"),
    # MCC
    unique_mcc       = ("mcc", "nunique"),
    biz_mcc_share    = ("is_biz_mcc", "mean"),
    # Мерчанты
    unique_merchants = ("merchant_id", "nunique"),
    # Активные месяцы
    active_months    = ("month", "nunique"),
).reset_index()

# Энтропия и HHI — считаем отдельно
card_feats["mcc_entropy"]   = df.groupby("card_number")["mcc"].apply(entropy).values
card_feats["hhi_merchants"] = df.groupby("card_number")["merchant_id"].apply(hhi).values

# Нормированные по месяцу фичи
card_feats["txn_per_month"]    = card_feats["total_txn_count"] / card_feats["active_months"].clip(lower=1)
card_feats["amount_per_month"] = card_feats["total_amount"]    / card_feats["active_months"].clip(lower=1)
card_feats["merch_per_month"]  = card_feats["unique_merchants"] / card_feats["active_months"].clip(lower=1)

card_feats.fillna(0, inplace=True)
card_feats["std_amount"] = card_feats["std_amount"].fillna(0)

print(f"  AGGREGATION COMPLETE. CARDS: {len(card_feats):,}, FEATURES: {card_feats.shape[1]}")

# ──────────────────────────────────────────────────
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ
# ──────────────────────────────────────────────────
print("\nTRAINING MODELS...")

FEATURE_COLS = [c for c in card_feats.columns if c not in ["card_number", "segment"]]
TARGET       = "segment"

X = card_feats[FEATURE_COLS].values.astype("float32")
y = card_feats[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} карт | Test: {len(X_test):,} карт")
print(f"  Class balance (train): {y_train.mean():.3f} (доля бизнес-карт)")

results = {}

# ── 5a. Baseline: Logistic Regression ─────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
results["LogisticRegression"] = {
    "roc_auc": roc_auc_score(y_test, lr_proba),
    "pr_auc":  average_precision_score(y_test, lr_proba),
    "proba":   lr_proba,
}
print(f"  [LR]  ROC-AUC: {results['LogisticRegression']['roc_auc']:.4f}  |  PR-AUC: {results['LogisticRegression']['pr_auc']:.4f}")

# ── 5b. Baseline: Random Forest ────────────────────
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_proba = rf_clf.predict_proba(X_test)[:, 1]
results["RandomForest"] = {
    "roc_auc": roc_auc_score(y_test, rf_proba),
    "pr_auc":  average_precision_score(y_test, rf_proba),
    "proba":   rf_proba,
}
print(f"  [RF]  ROC-AUC: {results['RandomForest']['roc_auc']:.4f}  |  PR-AUC: {results['RandomForest']['pr_auc']:.4f}")

# ── 5c. LightGBM с Optuna (30 trials) ─────────────
def lgb_objective(trial):
    params = {
        "verbosity": -1,
        "n_estimators":    trial.suggest_int("n_estimators", 200, 600),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
        "max_depth":       trial.suggest_int("max_depth", 4, 12),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight":    "balanced",
        "random_state":    42,
        "n_jobs":          -1,
    }
    clf = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(lgb_objective, n_trials=30, show_progress_bar=False)

best_params = study.best_params
best_params.update({"verbosity": -1, "class_weight": "balanced", "random_state": 42, "n_jobs": -1})
lgb_clf = lgb.LGBMClassifier(**best_params)
lgb_clf.fit(X_train, y_train)
lgb_proba = lgb_clf.predict_proba(X_test)[:, 1]
results["LightGBM"] = {
    "roc_auc": roc_auc_score(y_test, lgb_proba),
    "pr_auc":  average_precision_score(y_test, lgb_proba),
    "proba":   lgb_proba,
}
print(f"  [LGB] ROC-AUC: {results['LightGBM']['roc_auc']:.4f}  |  PR-AUC: {results['LightGBM']['pr_auc']:.4f}")
print(f"  [LGB] Лучшие параметры Optuna: {study.best_params}")

# ──────────────────────────────────────────────────
# 6. ОЦЕНКА И ВИЗУАЛИЗАЦИЯ МЕТРИК
# ──────────────────────────────────────────────────
print("\nEVALUATING METRICS...")

# Выбираем порог на LightGBM — максимизируем F1
from sklearn.metrics import precision_recall_curve
prec, rec, thresholds = precision_recall_curve(y_test, lgb_proba)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]
lgb_pred = (lgb_proba >= best_threshold).astype(int)
print(f"  Оптимальный порог (max F1): {best_threshold:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, lgb_pred)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Оценка моделей — LightGBM (лучшая модель)", fontsize=14, fontweight="bold")

# ROC Curves
for name, res in results.items():
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--")
axes[0].set_title("ROC-кривые")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend()

# Confusion Matrix
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_title(f"Confusion Matrix (порог={best_threshold:.2f})")
axes[1].set_xlabel("Предсказано")
axes[1].set_ylabel("Факт")
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Consumer", "Business"]); axes[1].set_yticklabels(["Consumer", "Business"])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, fontweight="bold")

# PR Curves
for name, res in results.items():
    prec_c, rec_c, _ = precision_recall_curve(y_test, res["proba"])
    axes[2].plot(rec_c, prec_c, label=f"{name} (AP={res['pr_auc']:.3f})")
axes[2].set_title("PR-кривые")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Metrics saved to {OUT_DIR / 'model_evaluation.png'}")

# Итоговый отчёт
print("\n  Classification Report (LightGBM):")
print(classification_report(y_test, lgb_pred, target_names=["Consumer", "Business"]))

metrics_df = pd.DataFrame(results).T[["roc_auc", "pr_auc"]].round(4)
print("\n  📊  Сравнение моделей:")
print(metrics_df.to_string())

# ──────────────────────────────────────────────────
# 7. ИНТЕРПРЕТАЦИЯ: SHAP
# ──────────────────────────────────────────────────
print("\nSHAP INTERPRETATION...")

# Используем подвыборку для скорости (SHAP на 2000 примеров)
sample_size = min(2000, len(X_test))
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=sample_size, replace=False)
X_sample = X_test[idx]

explainer   = shap.TreeExplainer(lgb_clf)
shap_values = explainer.shap_values(X_sample)

# Для бинарной классификации shap_values может быть списком [class0, class1]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_sample, feature_names=FEATURE_COLS, max_display=20,
                  show=False, plot_type="bar")
plt.title("SHAP: Важность признаков (LightGBM)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(sv, X_sample, feature_names=FEATURE_COLS, max_display=20,
                  show=False)
plt.title("SHAP: Summary Plot — влияние признаков", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  SHAP plots saved to {OUT_DIR}")

# ──────────────────────────────────────────────────
# 8. ПРИМЕНЕНИЕ К CONSUMER-СЕГМЕНТУ (ПОЛНЫЙ РАНЖИР)
# ──────────────────────────────────────────────────
print("\nAPPLYING TO CONSUMERS...")

consumer_cards = card_feats[card_feats["segment"] == 0].copy()
X_consumer = consumer_cards[FEATURE_COLS].values.astype("float32")
# Получаем вероятность бизнес-поведения для ВСЕХ 80 000 карт
consumer_cards["p_business"] = lgb_clf.predict_proba(X_consumer)[:, 1].round(6)

# Сохраняем ПОЛНЫЙ список (card_number, score) для сабмита
submission = consumer_cards[["card_number", "p_business"]].sort_values("p_business", ascending=False)
submission.to_csv(OUT_DIR / "final_submission.csv", index=False)
print(f"  Full scoring complete. File saved: {OUT_DIR / 'final_submission.csv'}")

# Анализ ТОП-50 кандидатов
top50 = submission.head(50)
hidden_entrepreneurs = consumer_cards[consumer_cards["card_number"].isin(top50["card_number"])].copy()
hidden_entrepreneurs = hidden_entrepreneurs.sort_values("p_business", ascending=False)

print(f"  Total consumer cards:        {len(consumer_cards):>10,}")
print(f"  TOP-10 HIDDEN ENTREPRENEURS CANDIDATES:")
print(hidden_entrepreneurs.head(10)[["card_number", "p_business", "total_txn_count", 
                                     "total_amount", "biz_mcc_share", "biz_hours_share"]].to_string(index=False))

# ──────────────────────────────────────────────────
# 9. ГЛУБОКИЙ АНАЛИЗ ТОП-КАНДИДАТОВ
# ──────────────────────────────────────────────────
print("\nVALIDATING TOP-50 CANDIDATES...")

# Сравнение ТОП-50 со средним потребителем и средним бизнесом
top50_stats = hidden_entrepreneurs[FEATURE_COLS].mean()
all_cons_stats = consumer_cards[FEATURE_COLS].mean()
all_biz_stats = card_feats[card_feats["segment"] == 1][FEATURE_COLS].mean()

comparison = pd.DataFrame({
    "TOP-50 Hidden": top50_stats,
    "Avg Business": all_biz_stats,
    "Avg Consumer": all_cons_stats
}).round(4)

print("\nСравнение характеристик:")
print(comparison.loc[["total_txn_count", "avg_amount", "biz_mcc_share", "biz_hours_share", "online_share"]])

# Сохранить детальный лог ТОП-50
hidden_entrepreneurs.to_csv(OUT_DIR / "top_50_candidates_detailed.csv", index=False)

# ──────────────────────────────────────────────────
# 10. ПРОФИЛЬ СЕГМЕНТОВ
# ──────────────────────────────────────────────────
print("\nSEGMENT PROFILING...")

DETECTION_THRESHOLD = 0.5
# Сравнение 3 групп: реальный бизнес / скрытые предприниматели / обычные потребители
consumer_cards["group"] = np.where(
    consumer_cards["p_business"] >= DETECTION_THRESHOLD,
    "Hidden Entrepreneur",
    "Regular Consumer"
)
biz_profile = card_feats[card_feats["segment"] == 1][FEATURE_COLS].assign(group="Real Business")
groups = pd.concat([
    consumer_cards[FEATURE_COLS + ["group"]],
    biz_profile
])

profile_cols = ["total_txn_count", "avg_amount", "unique_merchants",
                "biz_mcc_share", "biz_hours_share", "recurring_share", "online_share"]
profile_summary = groups.groupby("group")[profile_cols].median().round(3)
print("\n  Медианные показатели по группам:")
print(profile_summary.to_string())

# Визуализация профилей
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Профиль сегментов: Real Business / Hidden Entrepreneur / Regular Consumer",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
group_colors = {"Real Business": "#2196F3", "Hidden Entrepreneur": "#FF5722", "Regular Consumer": "#4CAF50"}

for i, col in enumerate(profile_cols):
    for grp, color in group_colors.items():
        data = groups[groups["group"] == grp][col].clip(
            upper=groups[col].quantile(0.99))
        axes[i].hist(data, bins=40, alpha=0.5, label=grp, color=color, density=True)
    axes[i].set_title(col)
    axes[i].legend(fontsize=7)

# Последний subplot — количество в каждой группе
cnt = groups["group"].value_counts()
axes[7].bar(cnt.index, cnt.values, color=[group_colors[g] for g in cnt.index])
axes[7].set_title("Размер сегментов")
for j, (g, v) in enumerate(cnt.items()):
    axes[7].text(j, v + cnt.max() * 0.01, f"{v:,}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "segment_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Profiles saved to {OUT_DIR / 'segment_profiles.png'}")

# ──────────────────────────────────────────────────
# ИТОГ
# ──────────────────────────────────────────────────
print("\n" + "="*65)
print("  PIPELINE COMPLETE")
print(f"  Artifacts saved in: {OUT_DIR}")
print("="*65)
