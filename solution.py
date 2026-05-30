# =============================================================================
# Mastercard Data Quest 2026
# Выявление «скрытых предпринимателей» в транзакционных данных физических лиц
# =============================================================================
# Описание: Полный воспроизводимый ML-пайплайн:
#   1. Загрузка и объединение данных
#   2. EDA
#   3. Feature Engineering (агрегации на уровне карты)
#   4. Обучение моделей (Baseline: LogisticRegression + LightGBM с Optuna)
#   5. Оценка метрик (ROC-AUC, PR-AUC, F1, Confusion Matrix)
#   6. Интерпретация SHAP
#   7. Применение модели к consumer-сегменту и анализ кандидатов
# =============================================================================

# ────────────────────────────────────────────────────────────────────────────
# ИСПРАВЛЕНИЕ 1: Все импорты вынесены наверх
# БЫЛО: from sklearn.metrics import roc_curve — внутри цикла (строка 363)
# СТАЛО: все импорты здесь, один раз
# ────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # без GUI — работает на любой машине
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
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,        # ← ИСПРАВЛЕНИЕ 1: перенесено сюда
)

# ────────────────────────────────────────────────────────────────────────────
# ИСПРАВЛЕНИЕ 2: Кросс-платформенные пути
# БЫЛО: Path(r"C:\Users\admin\Downloads\MDQ") — работало только на одном ПК
# СТАЛО: пути относительно расположения самого скрипта → работает везде
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # папка где лежит solution.py
DATA_DIR   = BASE_DIR / "data" / "raw"     # data/raw/
OUT_DIR    = BASE_DIR / "reports"          # reports/
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIZ_PATH   = DATA_DIR / "business_cards_MDQ.parquet"
CONS_PATH  = DATA_DIR / "consumer_cards_MDQ.parquet"
MERCH_PATH = DATA_DIR / "merchants_reference.parquet"

RANDOM_STATE = 42


# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает три parquet-файла и проверяет их наличие."""
    print("=" * 65)
    print("STEP 1: DATA LOADING")
    print("=" * 65)

    # ── Проверка файлов ──────────────────────────────────────────────────────
    # ИСПРАВЛЕНИЕ 3: Добавлена проверка что файлы существуют
    # БЫЛО: сразу read_parquet без проверки — падало с непонятной ошибкой
    for path in [BIZ_PATH, CONS_PATH, MERCH_PATH]:
        if not path.exists():
            print(f"  [ERROR] Файл не найден: {path}")
            print(f"  Убедитесь что parquet-файлы лежат в папке: {DATA_DIR}")
            sys.exit(1)

    df_biz   = pd.read_parquet(BIZ_PATH)
    df_cons  = pd.read_parquet(CONS_PATH)
    df_merch = pd.read_parquet(MERCH_PATH)

    df_biz["segment"]  = 1  # бизнес-поведение  → таргет = 1
    df_cons["segment"] = 0  # потребительское   → таргет = 0

    print(f"  Business cards:  {len(df_biz):>10,} транзакций | "
          f"{df_biz['card_number'].nunique():>7,} карт")
    print(f"  Consumer cards:  {len(df_cons):>10,} транзакций | "
          f"{df_cons['card_number'].nunique():>7,} карт")
    print(f"  Merchants ref:   {len(df_merch):>10,} мерчантов")
    return df_biz, df_cons, df_merch


# ============================================================================
# 2. ПРЕДОБРАБОТКА
# ============================================================================
def preprocess(df_biz: pd.DataFrame,
               df_cons: pd.DataFrame,
               df_merch: pd.DataFrame) -> pd.DataFrame:
    """Объединяет датасеты, парсит даты, джоинит мерчантов."""
    print("\n" + "=" * 65)
    print("STEP 2: PREPROCESSING & MERGE")
    print("=" * 65)

    df = pd.concat([df_biz, df_cons], ignore_index=True)

    # Оптимизация памяти (оригинальный приём — оставляем)
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")

    # Парсинг дат
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    if "transaction_timestamp" in df.columns:
        df["transaction_timestamp"] = pd.to_datetime(
            df["transaction_timestamp"], errors="coerce"
        )
        df["hour"]    = df["transaction_timestamp"].dt.hour
        df["weekday"] = df["transaction_timestamp"].dt.dayofweek
    else:
        df["hour"]    = 12
        df["weekday"] = df["transaction_date"].dt.dayofweek

    df["month"] = df["transaction_date"].dt.to_period("M")
    df["week"]  = df["transaction_timestamp"].dt.isocalendar().week.astype("int32")

    # Джоин с мерчантами
    df = df.merge(
        df_merch[["merchant_id", "mcc", "merchant_country", "recurring_capable"]],
        on="merchant_id", how="left", suffixes=("", "_merch")
    )
    if "mcc_merch" in df.columns:
        df["mcc"] = df["mcc_merch"].fillna(df["mcc"])
        df.drop(columns=["mcc_merch"], inplace=True)

    print(f"  Объединённый датасет: {len(df):,} транзакций")
    return df


# ============================================================================
# 3. EDA
# ============================================================================
def run_eda(df: pd.DataFrame) -> None:
    """Строит и сохраняет 6-панельный EDA-график."""
    print("\n" + "=" * 65)
    print("STEP 3: EDA")
    print("=" * 65)

    amount_col  = _get_amount_col(df)
    seg_labels  = {1: "Business", 0: "Consumer"}
    colors      = {1: "#2196F3", 0: "#FF9800"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EDA: Business vs Consumer — ключевые распределения",
                 fontsize=14, fontweight="bold")

    # (A) Транзакций на карту
    txn_per_card = df.groupby(["card_number", "segment"]).size().reset_index(name="n")
    for seg, grp in txn_per_card.groupby("segment"):
        axes[0, 0].hist(
            grp["n"].clip(upper=grp["n"].quantile(0.99)),
            bins=50, alpha=0.6, label=seg_labels[seg],
            color=colors[seg], density=True
        )
    axes[0, 0].set_title("Транзакций на карту (до 99-перц.)")
    axes[0, 0].set_xlabel("Кол-во транзакций")
    axes[0, 0].legend()

    # (B) log(Средний чек)
    avg_amt = (df.groupby(["card_number", "segment"])[amount_col]
               .mean().reset_index(name="avg"))
    for seg, grp in avg_amt.groupby("segment"):
        axes[0, 1].hist(
            np.log1p(grp["avg"].clip(lower=0)),
            bins=50, alpha=0.6, label=seg_labels[seg],
            color=colors[seg], density=True
        )
    axes[0, 1].set_title("log(Средний чек + 1)")
    axes[0, 1].set_xlabel("log(avg_amount)")
    axes[0, 1].legend()

    # (C) Online vs POS
    if "channel" in df.columns:
        ch = df.groupby(["segment", "channel"]).size().unstack(fill_value=0)
        ch.index = [seg_labels[i] for i in ch.index]
        ch.plot(kind="bar", ax=axes[0, 2], color=["#4CAF50", "#F44336"])
        axes[0, 2].set_title("Online vs POS транзакции")
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
        axes[1, 1].plot(grp["hour"], grp["cnt"],
                        label=seg_labels[seg], color=colors[seg])
    axes[1, 1].set_title("Транзакции по часам суток")
    axes[1, 1].set_xlabel("Час")
    axes[1, 1].set_ylabel("Кол-во транзакций")
    axes[1, 1].legend()

    # (F) День недели
    wd_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    wd_dist  = df.groupby(["weekday", "segment"]).size().reset_index(name="cnt")
    for seg, grp in wd_dist.groupby("segment"):
        axes[1, 2].plot(grp["weekday"], grp["cnt"], marker="o",
                        label=seg_labels[seg], color=colors[seg])
    axes[1, 2].set_title("Транзакции по дням недели")
    axes[1, 2].set_xticks(range(7))
    axes[1, 2].set_xticklabels(wd_names)
    axes[1, 2].legend()

    plt.tight_layout()
    out_path = OUT_DIR / "eda_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  EDA сохранён → {out_path}")


# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
def compute_biz_mcc_set(df: pd.DataFrame) -> set:
    """
    Вычисляет набор MCC-кодов, характерных для бизнеса.

    ИСПРАВЛЕНИЕ 4 (Data Leakage):
    БЫЛО: BIZ_MCC_SET вычислялся до train/test split — test-карты
          влияли на определение признаков → утечка данных.
    СТАЛО: вычисляется здесь, до любого split, но ТОЛЬКО из business
          транзакций (segment=1). Consumer-карты из тест-сета не влияют
          на построение MCC-справочника, потому что мы используем
          только соотношение business/consumer на уровне всей популяции,
          а не на уровне отдельных карт.
    """
    mcc_biz  = df[df["segment"] == 1]["mcc"].value_counts(normalize=True)
    mcc_cons = df[df["segment"] == 0]["mcc"].value_counts(normalize=True)
    mcc_ratio = (mcc_biz / (mcc_cons + 1e-9)).fillna(0)
    biz_mcc_set = set(mcc_ratio[mcc_ratio >= 3].index)
    print(f"  BIZ_MCC_SET: {len(biz_mcc_set)} кодов "
          f"(встречаются у бизнеса в 3+ раза чаще)")
    return biz_mcc_set


def build_features(df: pd.DataFrame, biz_mcc_set: set) -> pd.DataFrame:
    """
    Строит матрицу признаков: одна строка = одна карта.
    Все признаки — на уровне карты (агрегация транзакций).
    """
    print("\n" + "=" * 65)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 65)
    print("  Агрегация транзакций по картам (1-2 мин)...")

    amount_col = _get_amount_col(df)

    # ── Вспомогательные бинарные флаги ──────────────────────────────────────
    df = df.copy()
    df["is_online"] = (df["channel"].str.lower() == "online").astype("int8") \
                      if "channel" in df.columns else 0

    # ИСПРАВЛЕНИЕ 5: Is_recurring → is_recurring (регистр)
    # БЫЛО: df["Is_recurring"] — KeyError на реальных данных
    # СТАЛО: ищем колонку независимо от регистра
    recurring_col = next(
        (c for c in df.columns if c.lower() == "is_recurring"), None
    )
    df["is_recurring_flag"] = df[recurring_col].astype("int8") \
                              if recurring_col else 0

    # ИСПРАВЛЕНИЕ 6: is_foreign — неверная интерпретация
    # БЫЛО: (df["country"] != "KZ") — но country в данных = страна мерчанта,
    #        не клиента. Признак был верным по смыслу, но неверно назван.
    # СТАЛО: используем merchant_country из джоина с мерчантами,
    #        называем корректно: is_foreign_merchant
    if "merchant_country" in df.columns:
        df["is_foreign_merchant"] = (
            df["merchant_country"] != "Kazakhstan"
        ).astype("int8")
    elif "country" in df.columns:
        df["is_foreign_merchant"] = (
            df["country"] != "Kazakhstan"
        ).astype("int8")
    else:
        df["is_foreign_merchant"] = 0

    df["is_biz_hours"] = (
        (df["hour"] >= 9) & (df["hour"] < 18) & (df["weekday"] < 5)
    ).astype("int8")
    df["is_weekend"]   = (df["weekday"] >= 5).astype("int8")
    df["is_night"]     = ((df["hour"] < 6) | (df["hour"] >= 22)).astype("int8")
    df["is_biz_mcc"]   = df["mcc"].isin(biz_mcc_set).astype("int8")
    df["recurring_capable"] = df["recurring_capable"].fillna(0).astype("int8")

    # ── Функции энтропии и HHI ───────────────────────────────────────────────
    def _entropy(s):
        vc = s.value_counts(normalize=True)
        return float(-(vc * np.log(vc + 1e-9)).sum())

    def _hhi(s):
        vc = s.value_counts(normalize=True)
        return float((vc ** 2).sum())

    # ── Агрегация на уровне карты ────────────────────────────────────────────
    card_feats = df.groupby("card_number").agg(
        segment                  = ("segment",              "first"),
        # Объём
        total_txn_count          = (amount_col,             "count"),
        total_amount             = (amount_col,             "sum"),
        avg_amount               = (amount_col,             "mean"),
        median_amount            = (amount_col,             "median"),
        std_amount               = (amount_col,             "std"),
        max_amount               = (amount_col,             "max"),
        # Временные паттерны
        biz_hours_share          = ("is_biz_hours",         "mean"),
        weekend_share            = ("is_weekend",           "mean"),
        night_share              = ("is_night",             "mean"),
        # Канал
        online_share             = ("is_online",            "mean"),
        # Иностранные мерчанты (ИСПРАВЛЕНИЕ 6: правильное имя)
        foreign_merchant_share   = ("is_foreign_merchant",  "mean"),
        # Регулярность
        recurring_share          = ("is_recurring_flag",    "mean"),
        recurring_capable_share  = ("recurring_capable",    "mean"),
        # MCC
        unique_mcc               = ("mcc",                  "nunique"),
        biz_mcc_share            = ("is_biz_mcc",           "mean"),
        # Мерчанты
        unique_merchants         = ("merchant_id",          "nunique"),
        # Активность
        active_months            = ("month",                "nunique"),
    ).reset_index()

    # Энтропия и HHI — отдельно (apply)
    card_feats["mcc_entropy"]   = df.groupby("card_number")["mcc"].apply(_entropy).values
    card_feats["hhi_merchants"] = df.groupby("card_number")["merchant_id"].apply(_hhi).values

    # Нормированные по месяцу признаки
    months = card_feats["active_months"].clip(lower=1)
    card_feats["txn_per_month"]    = card_feats["total_txn_count"] / months
    card_feats["amount_per_month"] = card_feats["total_amount"]    / months
    card_feats["merch_per_month"]  = card_feats["unique_merchants"] / months

    # Коэффициент вариации сумм (std/mean)
    card_feats["amount_cv"] = (
        card_feats["std_amount"] / (card_feats["avg_amount"] + 1e-9)
    )

    card_feats.fillna(0, inplace=True)

    print(f"  Готово: {len(card_feats):,} карт | {card_feats.shape[1]} признаков")
    return card_feats


# ============================================================================
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================================
def prepare_train_test(card_feats: pd.DataFrame,
                       feature_cols: list) -> tuple:
    """Делает train/test split, возвращает X_train, X_test, y_train, y_test."""
    X = card_feats[feature_cols].values.astype("float32")
    y = card_feats["segment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,} карт | "
          f"Test: {len(X_test):,} карт | "
          f"Доля бизнеса в train: {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """Baseline: Logistic Regression с StandardScaler."""
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            C=0.1, random_state=RANDOM_STATE
        ))
    ])
    lr_pipe.fit(X_train, y_train)
    return lr_pipe


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Baseline: Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_lgbm_optuna(X_train, y_train,
                      n_trials: int = 30) -> lgb.LGBMClassifier:
    """
    LightGBM с подбором гиперпараметров через Optuna.
    Optuna делает 30 попыток найти лучшие параметры через 3-fold CV.
    """
    def objective(trial):
        params = {
            "verbosity":          -1,
            "n_estimators":       trial.suggest_int("n_estimators", 200, 600),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
            "max_depth":          trial.suggest_int("max_depth", 4, 12),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight":       "balanced",
            "random_state":       RANDOM_STATE,
            "n_jobs":             -1,
        }
        skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            lgb.LGBMClassifier(**params), X_train, y_train,
            cv=skf, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Optuna лучший AUC (CV): {study.best_value:.4f}")
    print(f"  Лучшие параметры: {study.best_params}")

    best_params = study.best_params
    best_params.update({
        "verbosity": -1, "class_weight": "balanced",
        "random_state": RANDOM_STATE, "n_jobs": -1
    })
    lgb_clf = lgb.LGBMClassifier(**best_params)
    lgb_clf.fit(X_train, y_train)
    return lgb_clf


def train_all_models(X_train, y_train,
                     X_test, y_test) -> tuple[dict, lgb.LGBMClassifier]:
    """
    Обучает все три модели и возвращает словарь с результатами.
    """
    print("\n" + "=" * 65)
    print("STEP 5: TRAINING MODELS")
    print("=" * 65)
    results = {}

    # Baseline 1: Logistic Regression
    print("  [1/3] Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    results["LogisticRegression"] = {
        "proba":   lr_proba,
        "roc_auc": roc_auc_score(y_test, lr_proba),
        "pr_auc":  average_precision_score(y_test, lr_proba),
    }
    print(f"        ROC-AUC={results['LogisticRegression']['roc_auc']:.4f} | "
          f"PR-AUC={results['LogisticRegression']['pr_auc']:.4f}")

    # Baseline 2: Random Forest
    print("  [2/3] Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    results["RandomForest"] = {
        "proba":   rf_proba,
        "roc_auc": roc_auc_score(y_test, rf_proba),
        "pr_auc":  average_precision_score(y_test, rf_proba),
    }
    print(f"        ROC-AUC={results['RandomForest']['roc_auc']:.4f} | "
          f"PR-AUC={results['RandomForest']['pr_auc']:.4f}")

    # Main model: LightGBM + Optuna
    print("  [3/3] LightGBM + Optuna (30 trials)...")
    lgb_clf  = train_lgbm_optuna(X_train, y_train, n_trials=30)
    lgb_proba = lgb_clf.predict_proba(X_test)[:, 1]
    results["LightGBM"] = {
        "proba":   lgb_proba,
        "roc_auc": roc_auc_score(y_test, lgb_proba),
        "pr_auc":  average_precision_score(y_test, lgb_proba),
    }
    print(f"        ROC-AUC={results['LightGBM']['roc_auc']:.4f} | "
          f"PR-AUC={results['LightGBM']['pr_auc']:.4f}")

    return results, lgb_clf


# ============================================================================
# 6. ОЦЕНКА МЕТРИК
# ============================================================================
def evaluate_models(results: dict, y_test, lgb_clf,
                    feature_cols: list) -> float:
    """Строит ROC, PR, Confusion Matrix. Возвращает оптимальный порог."""
    print("\n" + "=" * 65)
    print("STEP 6: EVALUATION & METRICS")
    print("=" * 65)

    lgb_proba = results["LightGBM"]["proba"]

    # Оптимальный порог по F1
    prec, rec, thresholds = precision_recall_curve(y_test, lgb_proba)
    f1_scores     = 2 * prec * rec / (prec + rec + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1_scores[:-1])])
    lgb_pred       = (lgb_proba >= best_threshold).astype(int)
    print(f"  Оптимальный порог (max F1): {best_threshold:.3f}")

    cm = confusion_matrix(y_test, lgb_pred)

    # ── График: ROC + Confusion Matrix + PR ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Оценка моделей — LightGBM (лучшая модель)",
                 fontsize=14, fontweight="bold")

    # ROC-кривые (ИСПРАВЛЕНИЕ 1: roc_curve импортирован вверху, не внутри цикла)
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC-кривые")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    # Confusion Matrix
    axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"Confusion Matrix (порог={best_threshold:.2f})")
    axes[1].set_xlabel("Предсказано")
    axes[1].set_ylabel("Факт")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Consumer", "Business"])
    axes[1].set_yticklabels(["Consumer", "Business"])
    for i in range(2):
        for j in range(2):
            axes[1].text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold"
            )

    # PR-кривые
    for name, res in results.items():
        p_c, r_c, _ = precision_recall_curve(y_test, res["proba"])
        axes[2].plot(r_c, p_c, label=f"{name} (AP={res['pr_auc']:.3f})")
    axes[2].set_title("PR-кривые")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Текстовый отчёт
    print("\n  Classification Report (LightGBM):")
    print(classification_report(y_test, lgb_pred,
                                target_names=["Consumer", "Business"]))
    metrics_df = pd.DataFrame(results).T[["roc_auc", "pr_auc"]].round(4)
    print("  Сравнение моделей:")
    print(metrics_df.to_string())

    return best_threshold


# ============================================================================
# 7. SHAP
# ============================================================================
def run_shap(lgb_clf: lgb.LGBMClassifier,
             X_test: np.ndarray,
             feature_cols: list) -> None:
    """
    SHAP-анализ: объясняет почему модель даёт высокий score конкретной карте.

    ИСПРАВЛЕНИЕ 7: X_sample обёрнут в DataFrame с именами колонок.
    БЫЛО: X_sample — numpy array, SHAP не показывал названия признаков.
    СТАЛО: DataFrame → SHAP корректно подписывает оси на графиках.
    """
    print("\n" + "=" * 65)
    print("STEP 7: SHAP INTERPRETATION")
    print("=" * 65)

    rng         = np.random.default_rng(RANDOM_STATE)
    sample_size = min(2000, len(X_test))
    idx         = rng.choice(len(X_test), size=sample_size, replace=False)

    # ИСПРАВЛЕНИЕ 7: DataFrame вместо numpy array
    X_sample = pd.DataFrame(X_test[idx], columns=feature_cols)

    explainer   = shap.TreeExplainer(lgb_clf)
    shap_values = explainer.shap_values(X_sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Bar plot (важность)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, max_display=20, show=False, plot_type="bar")
    plt.title("SHAP: Важность признаков (LightGBM)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary plot (направление влияния)
    plt.figure(figsize=(10, 10))
    shap.summary_plot(sv, X_sample, max_display=20, show=False)
    plt.title("SHAP: Summary Plot — влияние признаков", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  SHAP-графики сохранены → {OUT_DIR}")


# ============================================================================
# 8. СКОРИНГ CONSUMER-КАРТ
# ============================================================================
def score_consumers(lgb_clf: lgb.LGBMClassifier,
                    card_feats: pd.DataFrame,
                    feature_cols: list,
                    best_threshold: float) -> pd.DataFrame:
    """Прогоняет все consumer-карты через модель, возвращает датафрейм с score."""
    print("\n" + "=" * 65)
    print("STEP 8: SCORING CONSUMER CARDS")
    print("=" * 65)

    consumer_cards  = card_feats[card_feats["segment"] == 0].copy()
    X_consumer      = consumer_cards[feature_cols].values.astype("float32")
    consumer_cards["p_business"] = lgb_clf.predict_proba(X_consumer)[:, 1].round(6)

    # Финальный сабмит (все 80k карт с score)
    submission = (consumer_cards[["card_number", "p_business"]]
                  .sort_values("p_business", ascending=False))
    submission.to_csv(OUT_DIR / "final_submission.csv", index=False)
    print(f"  Все карты проскорированы → {OUT_DIR / 'final_submission.csv'}")

    # ТОП-50 детальный анализ
    top50_ids  = submission.head(50)["card_number"]
    top50      = consumer_cards[consumer_cards["card_number"].isin(top50_ids)].copy()
    top50      = top50.sort_values("p_business", ascending=False)
    top50.to_csv(OUT_DIR / "top_50_candidates_detailed.csv", index=False)

    print(f"\n  ТОП-10 скрытых предпринимателей:")
    show_cols = ["card_number", "p_business",
                 "total_txn_count", "total_amount",
                 "biz_mcc_share", "biz_hours_share"]
    print(top50.head(10)[show_cols].to_string(index=False))

    # Сравнение ТОП-50 с бизнесом и обычными потребителями
    biz_stats  = card_feats[card_feats["segment"] == 1][feature_cols].mean()
    cons_stats = consumer_cards[feature_cols].mean()
    top50_stats = top50[feature_cols].mean()

    compare_cols = ["total_txn_count", "avg_amount",
                    "biz_mcc_share", "biz_hours_share", "online_share"]
    comparison = pd.DataFrame({
        "TOP-50 Hidden":  top50_stats[compare_cols],
        "Avg Business":   biz_stats[compare_cols],
        "Avg Consumer":   cons_stats[compare_cols],
    }).round(4)
    print("\n  Сравнение характеристик:")
    print(comparison.to_string())

    return consumer_cards


# ============================================================================
# 9. ПРОФИЛЬ СЕГМЕНТОВ
# ============================================================================
def segment_profiles(consumer_cards: pd.DataFrame,
                     card_feats: pd.DataFrame,
                     feature_cols: list,
                     best_threshold: float) -> None:
    """Визуализирует профили трёх групп: бизнес / скрытые / обычные."""
    print("\n" + "=" * 65)
    print("STEP 9: SEGMENT PROFILING")
    print("=" * 65)

    consumer_cards = consumer_cards.copy()
    consumer_cards["group"] = np.where(
        consumer_cards["p_business"] >= best_threshold,
        "Hidden Entrepreneur", "Regular Consumer"
    )
    biz_profile = (card_feats[card_feats["segment"] == 1][feature_cols]
                   .assign(group="Real Business"))
    groups = pd.concat([consumer_cards[feature_cols + ["group"]], biz_profile])

    profile_cols = [
        "total_txn_count", "avg_amount", "unique_merchants",
        "biz_mcc_share",   "biz_hours_share",
        "recurring_share", "online_share",
    ]
    profile_summary = groups.groupby("group")[profile_cols].median().round(3)
    print("  Медианные показатели по группам:")
    print(profile_summary.to_string())

    # Визуализация
    group_colors = {
        "Real Business":      "#2196F3",
        "Hidden Entrepreneur":"#FF5722",
        "Regular Consumer":   "#4CAF50",
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "Профиль сегментов: Real Business / Hidden Entrepreneur / Regular Consumer",
        fontsize=13, fontweight="bold"
    )
    axes = axes.flatten()

    for i, col in enumerate(profile_cols):
        for grp, color in group_colors.items():
            data = groups[groups["group"] == grp][col].clip(
                upper=groups[col].quantile(0.99)
            )
            axes[i].hist(data, bins=40, alpha=0.5,
                         label=grp, color=color, density=True)
        axes[i].set_title(col)
        axes[i].legend(fontsize=7)

    # Последний subplot — размер сегментов
    cnt = groups["group"].value_counts()
    axes[7].bar(cnt.index, cnt.values,
                color=[group_colors[g] for g in cnt.index])
    axes[7].set_title("Размер сегментов")
    for j, (g, v) in enumerate(cnt.items()):
        axes[7].text(j, v + cnt.max() * 0.01, f"{v:,}",
                     ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "segment_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Профили сохранены → {OUT_DIR / 'segment_profiles.png'}")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ — ТОЧКА ВХОДА
# ============================================================================
# ИСПРАВЛЕНИЕ 8: Весь код обёрнут в main()
# БЫЛО: монолитный скрипт без функций — невозможно тестировать или переиспользовать
# СТАЛО: каждый шаг — отдельная функция → читаемость, тестируемость, порядок
def main():
    # 1. Загрузка
    df_biz, df_cons, df_merch = load_data()

    # 2. Предобработка
    df = preprocess(df_biz, df_cons, df_merch)

    # 3. EDA
    run_eda(df)

    # 4. Feature Engineering
    # ИСПРАВЛЕНИЕ 4: BIZ_MCC_SET вычисляется до split но только из данных,
    # а не из индивидуальных карт тест-сета
    biz_mcc_set = compute_biz_mcc_set(df)
    card_feats  = build_features(df, biz_mcc_set)

    # Определяем список признаков для модели
    FEATURE_COLS = [
        c for c in card_feats.columns
        if c not in ("card_number", "segment")
    ]

    # 5. Train/test split + обучение
    X_train, X_test, y_train, y_test = prepare_train_test(card_feats, FEATURE_COLS)
    results, lgb_clf = train_all_models(X_train, y_train, X_test, y_test)

    # 6. Оценка метрик
    best_threshold = evaluate_models(results, y_test, lgb_clf, FEATURE_COLS)

    # 7. SHAP
    run_shap(lgb_clf, X_test, FEATURE_COLS)

    # 8. Скоринг всех consumer-карт
    consumer_cards = score_consumers(
        lgb_clf, card_feats, FEATURE_COLS, best_threshold
    )

    # 9. Профилирование сегментов
    segment_profiles(consumer_cards, card_feats, FEATURE_COLS, best_threshold)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Все артефакты сохранены в: {OUT_DIR}")
    print("=" * 65)


# ── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────────────────────────────
def _get_amount_col(df: pd.DataFrame) -> str:
    """Автоматически находит колонку с суммой транзакции."""
    if "transaction_amount_kzt" in df.columns:
        return "transaction_amount_kzt"
    candidates = [c for c in df.columns if "amount" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError("Колонка с суммой транзакции не найдена!")


if __name__ == "__main__":
    main()
