## Стек и инструменты

* Язык и окружение
  * Python 3.10+ в ноутбуке (Jupyter / VS Code / Colab).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * Управление зависимостями: poetry / pip + requirements.txt (важно для reproducibility).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Основные библиотеки
  * pandas, numpy — подготовка и агрегации по картам/клиентам.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * pyarrow / fastparquet — чтение .parquet файлов с транзакциями и мерчантами.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * scikit-learn — train/test split, baseline‑модели (LogisticRegression, RandomForest, GradientBoosting, Pipeline, StandardScaler).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * lightgbm или xgboost — бустинг по табличным фичам (скорость + качество).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * matplotlib, seaborn — EDA и визуализации для презентации.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * shap — интерпретация важности признаков и объяснимость.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Инфраструктура/инструменты вокруг
  * GitHub/GitLab repo с кодом и инструкцией запуска (для Reproducibility).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
  * Makefile или простой run.sh / README с командами запуска.

## Общий план решения

1. Понять единицу предсказания
   * Скорее всего, хотим классифицировать не транзакции, а клиентов‑физлиц (или карты) на «скрытый предприниматель / обычный потребитель».[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Решение: агрегировать транзакции на уровень card_number или клиента (если есть customer_id; если нет — работаем на уровне карты).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
2. Получить/сконструировать таргет
   * Одна из логичных стратегий:
     * Использовать поведение business cardholders как позитивный «эталон» бизнес‑поведения.
     * Сформировать таргет:
       * business_cards → y=1 (бизнес‑поведение),
       * consumer_cards → y=0 (потребительское поведение).
   * После обучения использовать модель, чтобы выявить среди consumer_cards тех, кто по поведению похож на бизнес.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
3. EDA
   * Посмотреть распределения по сегментам (business vs consumer):
     * Кол-во транзакций на карту.
     * Средний чек, медианный чек, доля входящих/исходящих (если можно различить по признакам/знаку).
     * Распределения по MCC, merchant_id, странам и каналам (online/offline).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
     * Тайм‑паттерны: по часам суток, дням недели, сезонность (октябрь–март).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Цель: увидеть фичи, которые хорошо разделяют сегменты.
4. Фичеинжиниринг (на уровне карты/клиента)
   * Считаем агрегаты по consumer + business, чтобы модель училась на обоих.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Список фич — см. раздел ниже.
5. Построение baseline‑моделей
   * Начать с LogisticRegression / RandomForest как интерпретируемого baseline.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Метрики:
     * ROC‑AUC, PR‑AUC, accuracy.
     * Обязательно confusion matrix (precision, recall, F1) по условному cut‑off (например, выбранному по максимизации F1 или заданию recall).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
6. Улучшение модели
   * Попробовать LightGBM/XGBoost с простым подбором гиперпараметров (RandomizedSearchCV / Optuna).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Добавить более сложные фичи (MCC‑профиль, концентрации, регулярность).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Сравнить метрики vs baseline и зафиксировать лучший вариант.
7. Интерпретация и бизнес‑порог
   * Использовать SHAP/feature_importances_, чтобы объяснить, какие паттерны сильнее всего связаны с бизнес‑поведением.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Подобрать порог вероятности p* для флага «скрытый предприниматель»:
     * Можно оптимизировать под high‑precision (чтобы не раздражать обычных клиентов лишними предложениями) или под high‑recall (чтобы не упускать бизнес).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
8. Применение модели к consumer‑сегменту
   * Обучаем модель на смешанном датасете (business + consumer, с таргетом 1/0).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Считаем для всех consumer‑карт p(business‑like) и сортируем по вероятности.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
   * Анализируем топ‑N кандидатов:
     * Профиль MCC, регулярность платежей, география.
     * Формируем сегмент для маркетинга/продаж и описываем бизнес‑кейс.
9. Финализация
   * Оформляем код в виде:
     * 1. data_prep.py/notebook, 2) feature_engineering.py/notebook, 3) modeling.py, 4) evaluation_report.ipynb.
   * Делаем презентацию: проблема → данные → фичи → модель → результаты → бизнес‑кейсы и рекомендации.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Идеи фич для карты/клиента

Все фичи считаются на основе оконного периода 6 месяцев (данный период данных).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Объём и интенсивность

* total_txn_count — общее количество транзакций.
* total_txn_amount — суммарный оборот по карте.
* avg_txn_amount, median_txn_amount — средний и медианный чек.
* std_txn_amount — вариативность чека.
* max_txn_amount — максимальная транзакция.

Ожидание: у бизнес‑поведения больше транзакций и выше оборот, но средний чек может быть не всегда намного выше обычного потребителя.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Структура по MCC и мерчантам

* top_mcc_k_share — доля транзакций в топ‑k бизнес‑MCC (выделить их по поведению business‑карт).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* mcc_entropy — энтропия распределения MCC (насколько разнообразны категории трат).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* unique_mcc_count — количество уникальных MCC.
* unique_merchants_count — количество уникальных merchant_id.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* hhi_merchants — индекс Херфиндаля по merchant_id (меряет концентрацию оборота на нескольких мерчантах).

Гипотеза: у скрытого бизнеса выше число уникальных клиентов/мерчантов и иная структура MCC.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Регулярность и подписки

* recurring_txn_count, recurring_txn_share — число и доля транзакций с Is_recurring=1.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* recurring_capable_merchants_share — доля мерчантов из справочника с recurring_capable=1.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* month_activity_count — кол-во активных месяцев (в скольких месяцах были транзакции).

Скрытые предприниматели могут иметь регулярные платежи поставщикам сервисов (аренда, реклама, SaaS и т.п.).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Тайм‑паттерны

* weekday_vs_weekend_ratio — отношение транзакций в будни к выходным.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* night_txn_share — доля транзакций ночью (например, 00:00–06:00).
* business_hours_txn_share — доля транзакций в «рабочие часы» (например, 9–18, по местному времени).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* hourly_entropy — энтропия распределения по часам: насколько «равномерно» растянуты операции по суткам.

У бизнеса больше транзакций в будни и в бизнес‑часы, у обычных людей пик по вечерам и выходным.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## География и каналы

* online_txn_share — доля транзакций channel='online'.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* foreign_country_txn_share — доля транзакций за пределами страны клиента (по полю country).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* merchant_foreign_share — доля мерчантов из merchant_country ≠ страна клиента.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

Например, малый онлайн‑бизнес может интенсивно использовать зарубежные сервисы рекламы/обработки платежей.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Структура контрагентов (proxy для B2C)

Если есть признаки, позволяющие различать «входящие» и «исходящие» платежи, можно добавить:

* incoming_txn_count, incoming_amount_share.
* unique_incoming_counterparties — количество «платящих» клиентов, если это выводится из данных.

Даже без явного направления можно косвенно судить по MCC и типу мерчантов.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Относительные фичи

* txns_per_month = total_txn_count / активные месяцы.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* avg_amount_per_month = total_txn_amount / активные месяцы.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* unique_merchants_per_month.

Это нормирует поведение по длительности истории в данных.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Конкретный технический пайплайн

## 1. Загрузка и объединение данных

* Считать:
  * business_cards.parquet → df_business.
  * consumer_cards.parquet → df_consumer.
  * merchants_reference.parquet → df_merchants.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Джоинить merchants_reference по merchant_id ко всем транзакциям.

## 2. Разметка и объединённый датасет

* Добавить колонку segment:
  * business → 1, consumer → 0.
* Объединить df = concat([df_business, df_consumer]).

## 3. Агрегации до уровня карты

* Группировка: df_grouped = df.groupby('card_number').agg({...}).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* В агрегатах учитывать и сегмент (target = max(segment)).

## 4. Препроцессинг

* Обработка пропусков:
  * Для числовых фич → median/0.
  * Для категорий (если оставишь какие‑то) → 'missing'.
* Нормализация/стандартизация для моделей типа LogisticRegression, KNN (StandardScaler).

## 5. Обучение моделей

* Разбить на train/test по card_number (stratify по target).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Baseline: LogisticRegression (с регуляризацией) и RandomForestClassifier.
* Затем LightGBM:
  * Параметры: num_leaves, max_depth, learning_rate, n_estimators подбирать через RandomizedSearchCV или небольшую ручную сетку.

## 6. Оценка

* ROC‑AUC, PR‑AUC.
* confusion_matrix на тесте (для выбранного порога).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Посчитать precision, recall, F1.
* Для презентации нарисовать: ROC‑кривую, PR‑кривую, bar‑chart feature_importances_.

## 7. Интерпретация и бизнес‑сегментация

* SHAP summary plot, чтобы показать топ‑10 фич, влияющих на предсказание «бизнес‑поведения».[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Применить модель ко всем consumer‑картам, посчитать вероятность p_i.
* Взять, скажем, топ‑5% карт по p_i — это кандидаты «скрытых предпринимателей».

Сделать несколько case‑study: показать анонимный профиль клиента (фичи + MCC‑распределение) и объяснить, почему модель считает его бизнес‑подобным.[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)

## Что подчеркнуть в презентации

* Указать, что данные синтетические, но гипотезы опираются на реальные банковские практики (MCC, эквайринг, поведение SME).[](https://drive.google.com/drive/folders/1xHu7QzRpqvhf2T8KCd_ZGduPMRFpWzI_)
* Сфокусироваться на бизнес‑выгоде:
  * потенциал доп. дохода от конвертации X% клиентов в бизнес‑сегмент,
  * приоритизация для продаж/маркетинга вместо массовых кампаний.
* Отдельный слайд про explainability: как банку объяснять менеджерам и регуляторам, почему конкретного клиента система пометила как «бизнес‑подобного» (через фичи и SHAP).
