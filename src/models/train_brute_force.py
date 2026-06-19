"""
Обучение Brute-Force детектора Clarify.

Использование:
    python -m src.models.train_brute_force
    python -m src.models.train_brute_force --lang en
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
import yaml

sys.path.insert(0, ".")

from src.data.synthetic_generator import SyntheticGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "auth_failure_rate",
    "unique_usernames",
    "target_hosts",
    "new_user_agents",
    "total_attempts",
]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "brute_force_xgb.json"
METRICS_PATH = MODEL_DIR / "brute_force_metrics.json"


def generate_brute_force_data(num_hosts: int = 150):
    """Генерирует синтетику для Brute-Force."""
    logger.info("Генерация данных для Brute-Force...")

    X_list, y_list = [], []
    rng = np.random.RandomState(42)

    # Benign: нормальная аутентификация (70% хостов)
    benign_hosts = int(num_hosts * 0.7)
    for _ in range(benign_hosts):
        source_ip = f"10.0.{rng.randint(1, 255)}.{rng.randint(1, 255)}"

        # Нормальные попытки: 1-5 событий за час
        n_events = rng.randint(1, 5)
        timestamps = sorted([rng.uniform(0, 3600) for _ in range(n_events)])
        usernames = [f"user_{rng.randint(1, 20)}" for _ in range(n_events)]
        targets = [f"192.168.1.{rng.randint(1, 10)}" for _ in range(n_events)]

        if n_events >= 2:
            duration = (timestamps[-1] - timestamps[0]) / 60.0
            if duration < 0.1:
                duration = 0.1
        else:
            duration = 60.0

        X_list.append([
            n_events / duration,
            float(len(set(usernames))),
            float(len(set(targets))),
            0.0,  # нет новых user-agent
            float(n_events),
        ])
        y_list.append(0)

    # Attack: brute-force (30% хостов)
    attack_hosts = num_hosts - benign_hosts
    for _ in range(attack_hosts):
        source_ip = f"{rng.randint(1, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 255)}"

        # Много попыток за короткое время (20-60 попыток за 5 минут)
        n_events = rng.randint(20, 60)
        timestamps = sorted([rng.uniform(0, 300) for _ in range(n_events)])
        usernames = [f"admin_{rng.randint(1, 50)}" for _ in range(n_events)]
        targets = [f"192.168.1.{rng.randint(1, 20)}" for _ in range(n_events)]

        duration = (timestamps[-1] - timestamps[0]) / 60.0
        if duration < 0.1:
            duration = 0.1

        X_list.append([
            n_events / duration,
            float(len(set(usernames))),
            float(len(set(targets))),
            float(len(set(usernames))),  # много уникальных имён = косвенный признак
            float(n_events),
        ])
        y_list.append(1)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    # Перемешиваем
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    logger.info(f"Brute-Force данные: {X.shape[0]} сэмплов, "
                f"attack={sum(y == 1)} ({sum(y == 1) / len(y):.1%})")

    return X, y


def cross_validate(X, y, n_folds: int = 5) -> dict:
    """Кросс-валидация."""
    from sklearn.model_selection import StratifiedKFold

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
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "lambda": 0.5,
            "alpha": 0.3,
            "scale_pos_weight": scale_pos_weight,
            "seed": 42,
        }

        model = xgb.train(
            params, dtrain, num_boost_round=300,
            evals=[(dval, "val")],
            early_stopping_rounds=20,
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
                    f"Precision={best_precision:.3f}, Recall={best_recall:.3f}")

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
    """Обучает финальную модель."""
    logger.info("Обучение финальной модели...")

    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
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
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=50,
    )

    train_time = time.time() - start_time

    logger.info(f"Обучение завершено за {train_time:.1f}с")
    logger.info(f"  Лучшая итерация: {model.best_iteration}")
    logger.info(f"  Число деревьев: {model.num_boosted_rounds()}")

    return model


def calibrate_threshold(model, X_cal, y_cal, target_precision=0.8):
    """Калибрует порог."""
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
    """Сохраняет модель и обновляет конфиг."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))
    logger.info(f"Модель сохранена: {MODEL_PATH}")

    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detector": "brute_force",
        "threshold": threshold,
        "metrics": metrics,
    }
    if cv_metrics:
        all_metrics["cross_validation"] = cv_metrics

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Метрики сохранены: {METRICS_PATH}")

    config_path = Path("config/detectors.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "brute_force" not in config["detectors"]:
        config["detectors"]["brute_force"] = {}

    config["detectors"]["brute_force"]["decision_threshold"] = round(threshold, 4)
    config["detectors"]["brute_force"]["model_path"] = str(MODEL_PATH)
    config["detectors"]["brute_force"]["min_events"] = 10
    config["detectors"]["brute_force"]["enabled"] = True

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"Конфиг обновлён: {config_path}")


def test_shap(model):
    """Тестирует SHAP-объяснитель для Brute-Force."""
    logger.info("\n" + "=" * 60)
    logger.info("Тест SHAP-объяснений (Brute-Force)")

    from src.explainers.shap_explainer import ShapExplainer
    from src.rendering.template_renderer import TemplateRenderer
    from src.ui.alert_card import AlertCardBuilder, AlertCardRenderer

    explainer = ShapExplainer(model, FEATURE_NAMES, top_n=3)
    renderer = TemplateRenderer()
    builder = AlertCardBuilder(template_renderer=renderer)
    cli_renderer = AlertCardRenderer(use_colors=True)

    # Генерируем тестовый сэмпл атаки
    rng = np.random.RandomState(42)
    n_events = 40
    timestamps = sorted([rng.uniform(0, 300) for _ in range(n_events)])
    duration = (timestamps[-1] - timestamps[0]) / 60.0
    if duration < 0.1:
        duration = 0.1

    test_sample = np.array([[
        n_events / duration,
        15.0,  # 15 уникальных имён
        5.0,  # 5 целевых хостов
        10.0,  # 10 новых user-agent
        float(n_events),
    ]])

    shap_result = explainer.explain(
        feature_vector=test_sample,
        alert_type="brute_force",
        context={"source_ip": "203.0.113.45"},
    )

    dmatrix = xgb.DMatrix(test_sample, feature_names=FEATURE_NAMES)
    score = float(model.predict(dmatrix)[0])

    with open("config/detectors.yaml", "r") as f:
        config = yaml.safe_load(f)
    threshold = config["detectors"]["brute_force"]["decision_threshold"]

    card = builder.build(
        alert_type="brute_force",
        source_ip="203.0.113.45",
        target_ip="192.168.1.5",
        model_score=score,
        model_threshold=threshold,
        shap_explanation=shap_result,
        detector_name="brute_force",
    )

    print(cli_renderer.render(card))
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Brute-Force detector")
    parser.add_argument("--hosts", type=int, default=200,
                        help="Number of hosts for synthetic data")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Обучение Brute-Force детектора Clarify")
    logger.info("=" * 60)

    # Шаг 1: Данные
    X, y = generate_brute_force_data(num_hosts=args.hosts)

    # Шаг 2: Кросс-валидация
    cv_metrics = cross_validate(X, y, n_folds=args.cv)

    # Шаг 3: Разделение train/val/cal
    split = int(0.7 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:int(0.85 * len(X))], y[split:int(0.85 * len(X))]
    X_cal, y_cal = X[int(0.85 * len(X)):], y[int(0.85 * len(X)):]

    # Шаг 4: Финальная модель
    model = train_final_model(X_train, y_train, X_val, y_val)

    # Шаг 5: Калибровка
    threshold, metrics = calibrate_threshold(model, X_cal, y_cal, target_precision=0.8)

    # Шаг 6: Сохранение
    save_model_and_config(model, threshold, metrics, cv_metrics)

    # Шаг 7: Тест SHAP
    test_shap(model)

    logger.info("\nГотово. Модель Brute-Force обучена и откалибрована.")
    logger.info(f"  F1 (CV): {cv_metrics['mean_f1']:.3f}±{cv_metrics['std_f1']:.3f}")
    logger.info(f"  Precision (cal): {metrics['precision']:.3f}")
    logger.info(f"  Recall (cal): {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()