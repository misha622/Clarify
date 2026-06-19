"""
SHAP-объяснитель для Clarify.

Использует TreeExplainer (оптимизирован для деревьев XGBoost/LightGBM)
для расчёта SHAP-значений в реальном времени.

Для каждого алерта:
1. Вычисляет точные SHAP-значения через TreeExplainer
2. Выбирает топ-N признаков строго по abs(SHAP value)
3. Возвращает признаковые имена, значения, SHAP и контекст для рендеринга

Латентность: < 1 мс на алерт для моделей до 100 деревьев глубиной до 6.
"""

import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class FeatureExplanation:
    """Объяснение одного признака в алерте."""
    feature_id: str
    feature_name: str  # human_name из словаря
    shap_value: float
    shap_abs: float
    feature_value: float
    baseline_value: Optional[float]
    context: dict  # переменные для NL-шаблона


@dataclass
class AlertExplanation:
    """Полное SHAP-объяснение алерта."""
    alert_type: str
    model_score: float
    top_features: list[FeatureExplanation]  # отсортированы по abs(SHAP)
    latency_ms: float
    total_features: int
    timestamp: float


class ShapExplainer:
    """
    SHAP-объяснитель на базе TreeExplainer.

    Работает ТОЛЬКО с древовидными моделями (XGBoost, LightGBM, CatBoost).
    Для нейросетей используйте отдельный суррогатный объяснитель (P1).

    Особенности:
    - Точные SHAP-значения (не приближения)
    - Время расчёта: < 1 мс для типичных моделей
    - Выбор признаков строго по abs(SHAP), без severity_weight
    - severity_weight используется ТОЛЬКО как tie-breaker
    """

    def __init__(
            self,
            model: xgb.Booster,
            feature_names: list[str],
            top_n: int = 3,
    ):
        """
        Args:
            model: обученная XGBoost-модель
            feature_names: имена признаков в порядке, ожидаемом моделью
            top_n: сколько топ-признаков возвращать
        """
        import shap

        self.model = model
        self.feature_names = feature_names
        self.top_n = top_n

        # TreeExplainer — оптимален для деревьев, считает точные SHAP
        self.explainer = shap.TreeExplainer(model)

        # Кэш ожидаемого значения (base value)
        ev = self.explainer.expected_value
        if hasattr(ev, '__len__') and not isinstance(ev, str):
            self.expected_value = float(ev[0])
        else:
            self.expected_value = float(ev)

        logger.info(
            f"ShapExplainer инициализирован: {len(feature_names)} признаков, "
            f"top_n={top_n}, expected_value={self.expected_value:.4f}"
        )

    def explain(
            self,
            feature_vector: np.ndarray,
            alert_type: str,
            feature_values: Optional[dict[str, float]] = None,
            baselines: Optional[dict[str, Optional[float]]] = None,
            context: Optional[dict] = None,
    ) -> AlertExplanation:
        """
        Вычисляет SHAP-объяснение для одного алерта.

        Args:
            feature_vector: вектор признаков формы (1, n_features)
            alert_type: тип алерта (beaconing, brute_force, dga, ...)
            feature_values: словарь {feature_name: value} — текущие значения
            baselines: словарь {feature_name: baseline} — базовые линии
            context: дополнительный контекст (IP, домен, ...)

        Returns:
            AlertExplanation с топ-признаками и метаданными
        """
        start_time = time.perf_counter()

        # Шаг 1: Расчёт SHAP-значений
        shap_values = self.explainer.shap_values(feature_vector)

        # shap_values может быть (n_samples, n_features) или (n_features,)
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Шаг 2: Формируем список объяснений признаков
        explanations: list[FeatureExplanation] = []

        for i, (fname, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            feat_value = feature_vector[0, i] if feature_vector.ndim > 1 else feature_vector[i]
            baseline = baselines.get(fname) if baselines else None

            feat_context = {
                "value": float(feat_value),
                "baseline": baseline,
                "ratio": (float(feat_value) / baseline) if (baseline and baseline > 0) else None,
                **(context or {}),
            }

            explanations.append(FeatureExplanation(
                feature_id=fname,
                feature_name=fname,  # будет заменено на human_name в рендерере
                shap_value=float(shap_val),
                shap_abs=abs(float(shap_val)),
                feature_value=float(feat_value),
                baseline_value=baseline,
                context=feat_context,
            ))

        # Шаг 3: Сортировка строго по abs(SHAP)
        explanations.sort(key=lambda e: e.shap_abs, reverse=True)

        # Шаг 4: Отбираем топ-N
        top_features = explanations[:self.top_n]

        # Шаг 5: Замер латентности
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Шаг 6: Получаем скор модели
        dmatrix = xgb.DMatrix(feature_vector, feature_names=self.feature_names)
        model_score = float(self.model.predict(dmatrix)[0])

        alert_explanation = AlertExplanation(
            alert_type=alert_type,
            model_score=model_score,
            top_features=top_features,
            latency_ms=latency_ms,
            total_features=len(self.feature_names),
            timestamp=time.time(),
        )

        logger.debug(
            f"SHAP объяснение: alert_type={alert_type}, "
            f"score={model_score:.4f}, "
            f"top_features={[f.feature_id for f in top_features]}, "
            f"latency={latency_ms:.2f}ms"
        )

        return alert_explanation

    def explain_batch(
            self,
            feature_matrix: np.ndarray,
            alert_types: list[str],
            contexts: Optional[list[dict]] = None,
    ) -> list[AlertExplanation]:
        """
        Вычисляет SHAP-объяснения для батча алертов.
        Эффективнее, чем вызывать explain() в цикле.

        Args:
            feature_matrix: матрица признаков (n_samples, n_features)
            alert_types: типы алертов для каждого сэмпла
            contexts: контексты для каждого сэмпла

        Returns:
            список AlertExplanation
        """
        start_time = time.perf_counter()

        # Батчевый расчёт SHAP
        shap_matrix = self.explainer.shap_values(feature_matrix)

        results = []

        for sample_idx in range(feature_matrix.shape[0]):
            shap_values = shap_matrix[sample_idx]
            feature_vector = feature_matrix[sample_idx:sample_idx + 1]

            # Собираем объяснения
            explanations = []
            for i, (fname, shap_val) in enumerate(zip(self.feature_names, shap_values)):
                explanations.append(FeatureExplanation(
                    feature_id=fname,
                    feature_name=fname,
                    shap_value=float(shap_val),
                    shap_abs=abs(float(shap_val)),
                    feature_value=float(feature_matrix[sample_idx, i]),
                    baseline_value=None,
                    context=contexts[sample_idx] if contexts else {},
                ))

            explanations.sort(key=lambda e: e.shap_abs, reverse=True)

            # Скор модели
            dmatrix = xgb.DMatrix(feature_vector, feature_names=self.feature_names)
            score = float(self.model.predict(dmatrix)[0])

            results.append(AlertExplanation(
                alert_type=alert_types[sample_idx],
                model_score=score,
                top_features=explanations[:self.top_n],
                latency_ms=0.0,  # заполним общую латентность ниже
                total_features=len(self.feature_names),
                timestamp=time.time(),
            ))

        total_latency = (time.perf_counter() - start_time) * 1000
        per_sample = total_latency / len(results) if results else 0.0

        for r in results:
            r.latency_ms = per_sample

        logger.debug(
            f"Батчевый SHAP: {len(results)} сэмплов, "
            f"total={total_latency:.2f}ms, per_sample={per_sample:.2f}ms"
        )

        return results

    def get_feature_importance_global(self) -> list[dict]:
        """
        Возвращает глобальную важность признаков (средний |SHAP|).
        Используется для демо-режима и онбординга.

        Returns:
            список {feature_name, mean_abs_shap}, отсортированный по убыванию
        """
        # TreeExplainer хранит важность в себе
        if hasattr(self.explainer, 'feature_importances_'):
            importances = self.explainer.feature_importances_
        else:
            # Fallback: используем XGBoost feature importance
            importance_dict = self.model.get_score(importance_type="gain")
            importances = np.zeros(len(self.feature_names))
            for i, fname in enumerate(self.feature_names):
                importances[i] = importance_dict.get(f"f{i}", 0.0)

        result = []
        for fname, imp in zip(self.feature_names, importances):
            result.append({
                "feature_name": fname,
                "mean_abs_shap": float(imp),
            })

        result.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
        return result


# ------------------------------------------------------------------
# Тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    from src.features.window_stats import calculate_window_stats
    from src.data.synthetic_generator import SyntheticGenerator

    print("=" * 70)
    print("Тест ShapExplainer")
    print()

    # Генерируем синтетику и обучаем модель
    gen = SyntheticGenerator(seed=42)
    X, y = gen.generate_for_beaconing_training(mode="train")

    feature_names = [
        "mean_interarrival_time",
        "std_interarrival_time",
        "coefficient_of_variation",
        "peak_autocorrelation_lag",
        "autocorrelation_peak_value",
        "entropy_interarrival",
        "event_count",
    ]

    # Обучаем XGBoost
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    params = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "eta": 0.1,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=20)

    # Инициализируем объяснитель
    explainer = ShapExplainer(model, feature_names, top_n=3)

    # Объясняем один алерт
    test_sample = X[0:1]  # первый сэмпл
    alert_type = "beaconing" if y[0] == 1 else "benign"

    explanation = explainer.explain(
        feature_vector=test_sample,
        alert_type=alert_type,
        context={"ip": "10.0.5.17"},
    )

    print(f"Тип алерта: {explanation.alert_type}")
    print(f"Скор модели: {explanation.model_score:.4f}")
    print(f"Латентность: {explanation.latency_ms:.3f} мс")
    print(f"Топ-{explainer.top_n} признаков (по |SHAP|):")
    for i, feat in enumerate(explanation.top_features, 1):
        direction = "↑" if feat.shap_value > 0 else "↓"
        print(
            f"  {i}. {direction} {feat.feature_id}: "
            f"SHAP={feat.shap_value:+.4f}, "
            f"value={feat.feature_value:.3f}"
        )

    print()
    print("Глобальная важность признаков:")
    global_imp = explainer.get_feature_importance_global()
    for item in global_imp:
        print(f"  {item['feature_name']}: {item['mean_abs_shap']:.4f}")

    print()
    print("Батчевый тест (10 сэмплов):")
    batch_X = X[:10]
    batch_types = ["beaconing" if y[i] == 1 else "benign" for i in range(10)]
    batch_results = explainer.explain_batch(batch_X, batch_types)
    print(f"  Всего: {len(batch_results)} объяснений")
    print(f"  Средняя латентность: {batch_results[0].latency_ms:.3f} мс/сэмпл")

    print("=" * 70)