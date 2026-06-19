"""
Скрипт обучения и калибровки Beaconing-детектора Clarify.

Улучшенная версия:
- Кросс-валидация для стабильной оценки
- Расширенная синтетика (больше вариативности)
- Сохранение всех метрик для отслеживания
- Поддержка русского и английского словарей

Использование:
    python -m src.models.train_beaconing
    python -m src.models.train_beaconing --lang en
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")

from src.data.synthetic_generator import SyntheticGenerator
from src.explainers.shap_explainer import ShapExplainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mean_interarrival_time",
    "std_interarrival_time",
    "coefficient_of_variation",
    "peak_autocorrelation_lag",
    "autocorrelation_peak_value",
    "entropy_interarrival",
    "event_count",
]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "beaconing_xgb.json"
METRICS_PATH = MODEL_DIR / "beaconing_metrics.json"


def generate_data(num_hosts: int = 200):
    """
    Генерирует расширенные train и calibrate датасеты.

    Использует несколько генераторов с разными сидами
    для увеличения разнообразия данных.
    """
    logger.info("Генерация расширенных синтетических данных...")

    X_list, y_list = [], []
    X_cal_list, y_cal_list = [], []

    # Используем несколько сидов для разнообразия
    for seed in [42, 123, 456, 789, 1024]:
        gen = SyntheticGenerator(seed=seed, time_window_hours=8)

        # Train
        X, y = gen.generate_for_beaconing_training(
            mode="train",
            window_size_seconds=900,
            stride_seconds=300,
            min_events_per_window=8,
            num_hosts=num_hosts // 5,
        )
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)

        # Calibrate
        gen_cal = SyntheticGenerator(seed=seed + 1000, time_window_hours=8)
        Xc, yc = gen_cal.generate_for_beaconing_training(
            mode="calibrate",
            window_size_seconds=900,
            stride_seconds=300,
            min_events_per_window=8,
            num_hosts=300 // 5,
        )
        if len(Xc) > 0:
            X_cal_list.append(Xc)
            y_cal_list.append(yc)

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    X_cal = np.vstack(X_cal_list)
    y_cal = np.concatenate(y_cal_list)

    # Гарантируем атаки в calibrate
    if sum(y_cal == 1) < 3:
        logger.warning("Мало атак в calibrate, добавляю принудительно...")
        gen_extra = SyntheticGenerator(seed=9999, time_window_hours=8)
        X_ex, y_ex = gen_extra.generate_for_beaconing_training(
            mode="train", window_size_seconds=900, stride_seconds=300,
            min_events_per_window=8, num_hosts=30,
        )
        attack_mask = y_ex == 1
        X_cal = np.vstack([X_cal, X_ex[attack_mask]])
        y_cal = np.concatenate([y_cal, y_ex[attack_mask]])

    # Перемешиваем
    idx = np.random.RandomState(42).permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    idx_cal = np.random.RandomState(42).permutation(len(X_cal))
    X_cal, y_cal = X_cal[idx_cal], y_cal[idx_cal]

    logger.info(f"Train: {X_train.shape[0]} сэмплов, "
                f"attack={sum(y_train == 1)} ({sum(y_train == 1) / len(y_train):.1%})")
    logger.info(f"Calibrate: {X_cal.shape[0]} сэмплов, "
                f"attack={sum(y_cal == 1)} ({sum(y_cal == 1) / len(y_cal):.1%})")

    return X_train, y_train, X_cal, y_cal


def cross_validate(X, y, n_folds: int = 5) -> dict:
    """
    Кросс-валидация для стабильной оценки модели.

    Returns:
        словарь со средними метриками по фолдам
    """
    logger.info(f"Кросс-валидация ({n_folds} фолдов)...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        n_neg = sum(y_tr == 0)
        n_pos = sum(y_tr == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "lambda": 0.5,
            "alpha": 0.3,
            "scale_pos_weight": scale_pos_weight,
            "seed": 42,
        }

        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        preds = model.predict(dval)
        best_f1, best_threshold = 0.0, 0.5
        best_precision, best_recall = 0.0, 0.0

        for threshold in np.linspace(0.1, 0.95, 50):
            p = (preds >= threshold).astype(int)
            tp = int(np.sum((p == 1) & (y_val == 1)))
            fp = int(np.sum((p == 1) & (y_val == 0)))
            fn = int(np.sum((p == 0) & (y_val == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
                best_precision = precision
                best_recall = recall

        fold_metrics.append({
            "fold": fold + 1,
            "best_iteration": model.best_iteration,
            "f1": round(best_f1, 4),
            "precision": round(best_precision, 4),
            "recall": round(best_recall, 4),
            "threshold": round(best_threshold, 4),
        })

        logger.info(f"  Fold {fold + 1}: F1={best_f1:.3f}, "
                    f"Precision={best_precision:.3f}, Recall={best_recall:.3f}, "
                    f"Threshold={best_threshold:.3f}")

    avg_metrics = {
        "folds": fold_metrics,
        "mean_f1": round(np.mean([m["f1"] for m in fold_metrics]), 4),
        "std_f1": round(np.std([m["f1"] for m in fold_metrics]), 4),
        "mean_precision": round(np.mean([m["precision"] for m in fold_metrics]), 4),
        "mean_recall": round(np.mean([m["recall"] for m in fold_metrics]), 4),
        "mean_threshold": round(np.mean([m["threshold"] for m in fold_metrics]), 4),
    }

    logger.info(f"  Среднее: F1={avg_metrics['mean_f1']:.3f}±{avg_metrics['std_f1']:.3f}, "
                f"Precision={avg_metrics['mean_precision']:.3f}, "
                f"Recall={avg_metrics['mean_recall']:.3f}")

    return avg_metrics


def train_final_model(X_train, y_train, X_val, y_val):
    """Обучает финальную модель на всех данных."""
    logger.info("Обучение финальной модели...")

    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "lambda": 0.5,
        "alpha": 0.3,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
    }

    start_time = time.time()

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    train_time = time.time() - start_time

    logger.info(f"Обучение завершено за {train_time:.1f}с")
    logger.info(f"  Лучшая итерация: {model.best_iteration}")
    logger.info(f"  Число деревьев: {model.num_boosted_rounds()}")

    return model


def calibrate_threshold(model, X_cal, y_cal, target_precision=0.8):
    """Калибрует decision threshold."""
    logger.info(f"Калибровка порога (целевая точность ≥ {target_precision})...")

    dcal = xgb.DMatrix(X_cal, feature_names=FEATURE_NAMES)
    scores = model.predict(dcal)

    logger.info(f"  Распределение скоров: min={scores.min():.4f}, "
                f"median={np.median(scores):.4f}, max={scores.max():.4f}")

    best_threshold = 0.5
    best_metrics = None

    for threshold in np.linspace(0.05, 0.95, 60):
        preds = (scores >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (y_cal == 1)))
        fp = int(np.sum((preds == 1) & (y_cal == 0)))
        fn = int(np.sum((preds == 0) & (y_cal == 1)))
        tn = int(np.sum((preds == 0) & (y_cal == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if precision >= target_precision:
            if best_metrics is None or f1 > best_metrics["f1"]:
                best_threshold = float(threshold)
                best_metrics = {
                    "threshold": best_threshold,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "total": tp + fp + fn + tn,
                }

    if best_metrics is None:
        logger.warning("Не удалось достичь целевой точности, берём лучший F1")
        best_f1 = 0.0
        for threshold in np.linspace(0.1, 0.9, 40):
            preds = (scores >= threshold).astype(int)
            tp = int(np.sum((preds == 1) & (y_cal == 1)))
            fp = int(np.sum((preds == 1) & (y_cal == 0)))
            fn = int(np.sum((preds == 0) & (y_cal == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
                best_metrics = {
                    "threshold": best_threshold,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "tp": tp, "fp": fp, "fn": fn,
                    "tn": int(np.sum((preds == 0) & (y_cal == 0))),
                    "total": len(y_cal),
                }

    logger.info(f"  Порог: {best_metrics['threshold']:.3f}")
    logger.info(f"  Precision: {best_metrics['precision']:.3f}")
    logger.info(f"  Recall: {best_metrics['recall']:.3f}")
    logger.info(f"  F1: {best_metrics['f1']:.3f}")
    logger.info(f"  TP={best_metrics['tp']}, FP={best_metrics['fp']}, "
                f"FN={best_metrics['fn']}, TN={best_metrics['tn']}")

    return best_threshold, best_metrics


def save_model_and_config(model, threshold, metrics, cv_metrics=None):
    """Сохраняет модель, метрики и обновляет конфиг."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))
    logger.info(f"Модель сохранена: {MODEL_PATH}")

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "metrics": metrics,
    }
    if cv_metrics:
        all_metrics["cross_validation"] = cv_metrics

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Метрики сохранены: {METRICS_PATH}")

    import yaml
    config_path = Path("config/detectors.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["detectors"]["beaconing"]["decision_threshold"] = round(threshold, 4)
    config["detectors"]["beaconing"]["model_path"] = str(MODEL_PATH)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"Конфиг обновлён: {config_path}")


def test_shap(model, lang: str = "ru"):
    """Тестирует SHAP-объяснитель с указанным языком."""
    logger.info("\n" + "=" * 60)
    logger.info(f"Тест SHAP-объяснений (язык: {lang})")

    explainer = ShapExplainer(model, FEATURE_NAMES, top_n=3)

    # Импортируем рендерер с нужным языком
    from src.rendering.template_renderer import TemplateRenderer
    from src.ui.alert_card import AlertCardBuilder, AlertCardRenderer

    dict_path = f"config/feature_dictionary{'_en' if lang == 'en' else ''}.yaml"
    if not Path(dict_path).exists():
        dict_path = "config/feature_dictionary.yaml"

    renderer = TemplateRenderer(dictionary_path=dict_path)
    builder = AlertCardBuilder(template_renderer=renderer)
    cli_renderer = AlertCardRenderer(use_colors=True)

    gen = SyntheticGenerator(seed=456, time_window_hours=8)
    X_test, y_test = gen.generate_for_beaconing_training(
        mode="train", window_size_seconds=900, stride_seconds=300,
        min_events_per_window=8, num_hosts=30,
    )

    attack_idx = [i for i, label in enumerate(y_test) if label == 1][:1]

    for idx in attack_idx:
        sample = X_test[idx:idx + 1]
        shap_result = explainer.explain(
            feature_vector=sample,
            alert_type="beaconing",
            context={"source_ip": "45.33.32.156"},
        )

        dmatrix = xgb.DMatrix(sample, feature_names=FEATURE_NAMES)
        score = float(model.predict(dmatrix)[0])

        import yaml
        with open("config/detectors.yaml", "r") as f:
            config = yaml.safe_load(f)
        threshold = config["detectors"]["beaconing"]["decision_threshold"]

        card = builder.build(
            alert_type="beaconing",
            source_ip="45.33.32.156",
            target_ip="10.0.5.17",
            model_score=score,
            model_threshold=threshold,
            shap_explanation=shap_result,
            detector_name="beaconing",
        )

        print(cli_renderer.render(card))

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Beaconing detector")
    parser.add_argument("--lang", default="ru", choices=["ru", "en"],
                        help="Language for SHAP explanations")
    parser.add_argument("--hosts", type=int, default=200,
                        help="Number of hosts per generator (total ×5)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Обучение Beaconing-детектора Clarify (lang={args.lang})")
    logger.info("=" * 60)

    # Шаг 1: Данные
    X_train, y_train, X_cal, y_cal = generate_data(num_hosts=args.hosts)

    # Шаг 2: Кросс-валидация
    cv_metrics = cross_validate(X_train, y_train, n_folds=args.cv)

    # Шаг 3: Разделение train/val для финальной модели
    split = int(0.85 * len(X_train))
    X_tr, y_tr = X_train[:split], y_train[:split]
    X_val, y_val = X_train[split:], y_train[split:]

    # Шаг 4: Финальная модель
    model = train_final_model(X_tr, y_tr, X_val, y_val)

    # Шаг 5: Калибровка
    threshold, metrics = calibrate_threshold(model, X_cal, y_cal)

    # Шаг 6: Сохранение
    save_model_and_config(model, threshold, metrics, cv_metrics)

    # Шаг 7: Тест SHAP
    test_shap(model, lang=args.lang)

    logger.info("\nГотово. Модель обучена и откалибрована.")
    logger.info(f"  F1 (CV): {cv_metrics['mean_f1']:.3f}±{cv_metrics['std_f1']:.3f}")
    logger.info(f"  Precision (cal): {metrics['precision']:.3f}")
    logger.info(f"  Recall (cal): {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()