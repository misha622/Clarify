"""
Обучение DGA детектора Clarify.

Использование:
    python -m src.models.train_dga
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb
import yaml

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mean_entropy",
    "max_entropy",
    "mean_domain_length",
    "mean_vowel_consonant_ratio",
    "mean_ngram_score",
    "unique_domains",
    "nxdomain_rate",
    "unique_tld_count",
    "total_queries",
]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "dga_xgb.json"


def generate_dga_data(num_sources: int = 300):
    """Генерирует синтетику для DGA."""
    logger.info("Генерация данных для DGA...")

    rng = np.random.RandomState(42)
    X_list, y_list = [], []

    # Benign: легитимные домены (70%)
    legitimate_domains = [
        "google.com", "github.com", "amazonaws.com", "cloudflare.com",
        "microsoft.com", "apple.com", "facebook.com", "stackoverflow.com",
        "pypi.org", "docker.com", "kubernetes.io", "redhat.com",
        "ubuntu.com", "debian.org", "nginx.org", "python.org",
        "npmjs.com", "wikipedia.org", "archive.org", "gitlab.com",
        "mozilla.org", "apache.org", "grafana.com", "prometheus.io",
        "elastic.co", "digitalocean.com", "linode.com", "heroku.com",
    ]

    for _ in range(int(num_sources * 0.7)):
        # Легитимный хост запрашивает 10-30 разных легитимных доменов
        n_queries = rng.randint(10, 30)
        domains = [rng.choice(legitimate_domains) for _ in range(n_queries)]
        # NXDOMAIN rate низкий (< 5%)
        nxdomain = [rng.random() < 0.03 for _ in range(n_queries)]

        features = _extract_features_static(domains, nxdomain)
        X_list.append(features)
        y_list.append(0)

    # Attack: DGA-домены (30%)
    for _ in range(int(num_sources * 0.3)):
        # DGA-хост генерирует 30-80 случайных доменов
        n_queries = rng.randint(30, 80)
        domains = [_generate_dga_domain(rng) for _ in range(n_queries)]
        # Высокий NXDOMAIN rate (> 80%)
        nxdomain = [rng.random() < rng.uniform(0.8, 0.98) for _ in range(n_queries)]

        features = _extract_features_static(domains, nxdomain)
        X_list.append(features)
        y_list.append(1)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    logger.info(f"DGA данные: {X.shape[0]} сэмплов, "
                f"attack={sum(y == 1)} ({sum(y == 1) / len(y):.1%})")

    return X, y


def _generate_dga_domain(rng: np.random.RandomState) -> str:
    """Генерирует DGA-подобный домен."""
    length = rng.randint(10, 30)
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
    name = "".join(rng.choice(chars, size=length))
    tlds = [".com", ".net", ".xyz", ".top", ".pw", ".info", ".biz", ".cc"]
    return name + rng.choice(tlds)


def _extract_features_static(domains: list, nxdomain_flags: list) -> list:
    """Статическая версия извлечения признаков (без инстанцирования детектора)."""
    from src.detectors.dga import DGADetector

    n = len(domains)

    entropies = [DGADetector.shannon_entropy(d) for d in domains]
    lengths = [len(d.split('.')[0]) for d in domains]
    vc_ratios = [DGADetector.vowel_consonant_ratio(d) for d in domains]
    ngram_scores = [DGADetector.ngram_score(d) for d in domains]

    tlds = [DGADetector.extract_tld(d) for d in domains]

    return [
        np.mean(entropies),
        np.max(entropies),
        np.mean(lengths),
        np.mean(vc_ratios),
        np.mean(ngram_scores),
        float(len(set(domains))),
        sum(nxdomain_flags) / n if n > 0 else 0.0,
        float(len(set(tlds))),
        float(n),
    ]


def train_dga():
    """Обучает DGA модель."""
    X, y = generate_dga_data(num_sources=400)

    # Разделение
    split = int(0.7 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:int(0.85 * len(X))], y[split:int(0.85 * len(X))]
    X_cal, y_cal = X[int(0.85 * len(X)):], y[int(0.85 * len(X)):]

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
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
    }

    logger.info("Обучение DGA модели...")

    model = xgb.train(
        params, dtrain, num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=30,
    )

    # Калибровка порога
    dcal = xgb.DMatrix(X_cal, feature_names=FEATURE_NAMES)
    preds = model.predict(dcal)

    best_f1, best_threshold = 0.0, 0.5

    for threshold in np.linspace(0.1, 0.95, 50):
        p = (preds >= threshold).astype(int)
        tp = int(np.sum((p == 1) & (y_cal == 1)))
        fp = int(np.sum((p == 1) & (y_cal == 0)))
        fn = int(np.sum((p == 0) & (y_cal == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    logger.info(f"Лучший порог: {best_threshold:.3f}, F1: {best_f1:.3f}")

    # Сохранение
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    # Обновление конфига
    config_path = Path("config/detectors.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "dga" not in config["detectors"]:
        config["detectors"]["dga"] = {}

    config["detectors"]["dga"]["decision_threshold"] = round(best_threshold, 4)
    config["detectors"]["dga"]["model_path"] = str(MODEL_PATH)
    config["detectors"]["dga"]["min_queries_from_source"] = 10
    config["detectors"]["dga"]["enabled"] = True

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Тест SHAP
    _test_shap(model, best_threshold)

    logger.info(f"Модель сохранена: {MODEL_PATH}")
    logger.info("Готово.")


def _test_shap(model, threshold):
    """Тестирует SHAP для DGA."""
    logger.info("\n" + "=" * 60)
    logger.info("Тест SHAP-объяснений (DGA)")

    from src.explainers.shap_explainer import ShapExplainer
    from src.rendering.template_renderer import TemplateRenderer
    from src.ui.alert_card import AlertCardBuilder, AlertCardRenderer

    explainer = ShapExplainer(model, FEATURE_NAMES, top_n=3)
    renderer = TemplateRenderer()
    builder = AlertCardBuilder(template_renderer=renderer)
    cli_renderer = AlertCardRenderer(use_colors=True)

    # Тестовый DGA-сценарий
    rng = np.random.RandomState(42)
    domains = [_generate_dga_domain(rng) for _ in range(50)]
    nxdomain = [True] * 45 + [False] * 5

    features = np.array([_extract_features_static(domains, nxdomain)])

    shap_result = explainer.explain(
        feature_vector=features,
        alert_type="dga",
        context={"source_ip": "10.0.5.99"},
    )

    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
    score = float(model.predict(dmatrix)[0])

    card = builder.build(
        alert_type="dga",
        source_ip="10.0.5.99",
        target_ip=None,
        model_score=score,
        model_threshold=threshold,
        shap_explanation=shap_result,
        detector_name="dga",
    )

    print(cli_renderer.render(card))
    logger.info("=" * 60)


if __name__ == "__main__":
    train_dga()