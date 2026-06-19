"""
Beaconing Detector (C2-пульсации).

Детектирует регулярные коммуникации, характерные для C2-канала:
- Малая вариативность интервалов (CV < 0.2)
- Высокая автокорреляция
- Низкая энтропия интервалов

Реализован на engineered features (оконные статистики), что позволяет
использовать TreeExplainer для SHAP-объяснений в реальном времени.
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
import yaml

from ..features.window_stats import calculate_window_stats, WindowStats
from ..utils.thresholds import check_min_intervals, GatingDecision

logger = logging.getLogger(__name__)


@dataclass
class BeaconingResult:
    """Результат детекции Beaconing."""
    is_alert: bool
    score: float
    confidence: float
    window_stats: WindowStats
    reason: Optional[str] = None

    @property
    def should_alert(self) -> bool:
        return self.is_alert


class BeaconingDetector:
    """
    Детектор C2 Beaconing на основе оконных статистик.

    Использует XGBoost (деревья) для классификации, что даёт:
    - Быстрый TreeExplainer для SHAP в реальном времени
    - Честные объяснения без суррогатных моделей
    """

    def __init__(self, config_path: str = "config/detectors.yaml"):
        """
        Инициализирует детектор из конфигурационного файла.

        Args:
            config_path: путь к YAML-конфигу детекторов
        """
        self.config = self._load_config(config_path)
        beaconing_cfg = self.config["detectors"]["beaconing"]

        # Конфигурируемые параметры (НЕ хардкод)
        self.min_intervals = beaconing_cfg["min_intervals"]
        self.decision_threshold = beaconing_cfg["decision_threshold"]
        self.window_size = beaconing_cfg["window_size_seconds"]

        # Загрузка модели
        self.model: Optional[xgb.Booster] = None
        model_path = beaconing_cfg.get("model_path")
        if model_path and Path(model_path).exists():
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            logger.info(f"Модель загружена из {model_path}")
        else:
            logger.warning(
                f"Модель не найдена по пути {model_path}. "
                f"Используйте train() для обучения."
            )

        self.feature_names = beaconing_cfg["features"]

    def _load_config(self, config_path: str) -> dict:
        """Загружает конфигурацию из YAML."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def detect(
            self,
            timestamps: list[float],
            source_id: Optional[str] = None
    ) -> BeaconingResult:
        """
        Детектирует Beaconing-паттерн в последовательности временных меток.

        Args:
            timestamps: отсортированные временные метки событий (секунды)
            source_id: идентификатор источника (IP/хост) для логирования

        Returns:
            BeaconingResult с вердиктом, скором и объяснением
        """
        # Шаг 1: Confidence gate — проверка минимальной выборки
        intervals = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
        ]
        gate_result = check_min_intervals(intervals, self.min_intervals)

        if gate_result.decision == GatingDecision.INSUFFICIENT_DATA:
            logger.debug(
                f"Beaconing: {gate_result.reason}"
                + (f" (source: {source_id})" if source_id else "")
            )
            return BeaconingResult(
                is_alert=False,
                score=float("nan"),
                confidence=0.0,
                window_stats=calculate_window_stats(timestamps),
                reason=gate_result.reason,
            )

        # Шаг 2: Расчёт оконных статистик
        stats = calculate_window_stats(timestamps)
        feature_vector = np.array(stats.to_feature_vector()).reshape(1, -1)

        # Шаг 3: Инференс модели
        if self.model is not None:
            dmatrix = xgb.DMatrix(feature_vector, feature_names=stats.feature_names)
            raw_score = float(self.model.predict(dmatrix)[0])

            # Применяем конфигурируемый порог
            is_alert = raw_score >= self.decision_threshold

            logger.debug(
                f"Beaconing detection: score={raw_score:.4f}, "
                f"threshold={self.decision_threshold}, alert={is_alert}"
                + (f", source={source_id}" if source_id else "")
            )

            return BeaconingResult(
                is_alert=is_alert,
                score=raw_score,
                confidence=raw_score,  # Для XGBoost это probability-like score
                window_stats=stats,
                reason=(
                    f"Score {raw_score:.3f} >= threshold {self.decision_threshold}"
                    if is_alert
                    else f"Score {raw_score:.3f} < threshold {self.decision_threshold}"
                ),
            )
        else:
            # Если модели нет — возвращаем статистики без вердикта
            return BeaconingResult(
                is_alert=False,
                score=float("nan"),
                confidence=0.0,
                window_stats=stats,
                reason="Модель не загружена",
            )

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            model_path: str = "models/beaconing_xgb.json"
    ) -> dict:
        """
        Обучает XGBoost-модель на размеченных данных.

        Args:
            X_train: матрица признаков (обучающая)
            y_train: метки (обучающие)
            X_val: матрица признаков (валидационная)
            y_val: метки (валидационные)
            model_path: путь для сохранения модели

        Returns:
            словарь с метриками обучения
        """
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "val"))

        num_rounds = 100
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        # Сохраняем модель
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)

        # Обновляем путь в конфиге
        self.config["detectors"]["beaconing"]["model_path"] = model_path

        logger.info(f"Модель обучена и сохранена в {model_path}")

        return {
            "num_boost_rounds": self.model.num_boosted_rounds(),
            "best_iteration": self.model.best_iteration,
        }

    def tune_threshold(
            self,
            X_val: np.ndarray,
            y_val: np.ndarray,
            target_precision: float = 0.8
    ) -> float:
        """
        Калибрует decision threshold на реалистичной выборке (99/1).

        Args:
            X_val: матрица признаков
            y_val: истинные метки
            target_precision: целевая точность (по умолчанию 0.8)

        Returns:
            оптимальное значение порога
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")

        dval = xgb.DMatrix(X_val, feature_names=self.feature_names)
        scores = self.model.predict(dval)

        # Сортируем скоры с метками для построения precision-recall
        paired = sorted(zip(scores, y_val), key=lambda x: x[0], reverse=True)

        best_threshold = 0.5
        best_f1 = 0.0

        # Перебираем возможные пороги
        for threshold in np.linspace(0.1, 0.9, 50):
            predictions = [1 if s >= threshold else 0 for s, _ in paired]
            true_labels = [int(y) for _, y in paired]

            tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Ищем порог с precision >= target и максимальным F1
            if precision >= target_precision and f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Обновляем порог в конфиге
        self.decision_threshold = best_threshold
        self.config["detectors"]["beaconing"]["decision_threshold"] = best_threshold

        logger.info(
            f"Порог откалиброван: {best_threshold:.3f} "
            f"(precision ≥ {target_precision})"
        )

        return best_threshold