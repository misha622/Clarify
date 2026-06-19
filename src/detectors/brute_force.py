"""
Brute-Force Detector для Clarify.

Детектирует атаки перебора паролей (RDP, SSH, FTP) на основе:
- Частоты неуспешных аутентификаций (попыток/мин)
- Количества уникальных имён пользователей
- Количества целевых хостов
- Новых user-agent строк

Реализован на engineered features → совместим с TreeExplainer для SHAP.
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BruteForceWindowStats:
    """Статистики окна для Brute-Force детектора."""
    auth_failure_rate: float  # попыток/мин
    unique_usernames: int  # уникальных имён
    target_hosts: int  # целевых хостов
    new_user_agents: int  # новых user-agent
    total_attempts: int  # всего попыток

    def to_feature_vector(self) -> list[float]:
        return [
            self.auth_failure_rate,
            float(self.unique_usernames),
            float(self.target_hosts),
            float(self.new_user_agents),
            float(self.total_attempts),
        ]

    @property
    def feature_names(self) -> list[str]:
        return [
            "auth_failure_rate",
            "unique_usernames",
            "target_hosts",
            "new_user_agents",
            "total_attempts",
        ]


class BruteForceDetector:
    """
    Детектор Brute-Force атак.

    Использует XGBoost на агрегированных статистиках окна.
    Совместим с TreeExplainer для SHAP-объяснений.
    """

    FEATURE_NAMES = [
        "auth_failure_rate",
        "unique_usernames",
        "target_hosts",
        "new_user_agents",
        "total_attempts",
    ]

    def __init__(self, config_path: str = "config/detectors.yaml"):
        self.config = self._load_config(config_path)
        bf_cfg = self.config["detectors"].get("brute_force", {})

        self.min_events = bf_cfg.get("min_events", 10)
        self.decision_threshold = bf_cfg.get("decision_threshold", 0.5)
        self.window_size = bf_cfg.get("window_size_seconds", 3600)

        self.model: Optional[xgb.Booster] = None
        model_path = bf_cfg.get("model_path")
        if model_path and Path(model_path).exists():
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            logger.info(f"Модель Brute-Force загружена: {model_path}")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def extract_features(
            self,
            timestamps: list[float],
            usernames: list[str],
            target_ips: list[str],
            user_agents: list[str],
            known_user_agents: set = None,
    ) -> BruteForceWindowStats:
        """
        Извлекает признаки из событий аутентификации.

        Args:
            timestamps: временные метки событий
            usernames: имена пользователей для каждой попытки
            target_ips: целевые IP
            user_agents: user-agent строки
            known_user_agents: множество известных user-agent (для детекции новых)
        """
        n_events = len(timestamps)

        if n_events < 2:
            duration = 1.0
        else:
            duration = (timestamps[-1] - timestamps[0]) / 60.0  # в минутах
            if duration < 0.1:
                duration = 0.1

        auth_failure_rate = n_events / duration
        unique_usernames = len(set(usernames))
        target_hosts = len(set(target_ips))

        if known_user_agents:
            new_user_agents = len(set(user_agents) - known_user_agents)
        else:
            new_user_agents = len(set(user_agents))

        return BruteForceWindowStats(
            auth_failure_rate=auth_failure_rate,
            unique_usernames=unique_usernames,
            target_hosts=target_hosts,
            new_user_agents=new_user_agents,
            total_attempts=n_events,
        )

    def detect(
            self,
            timestamps: list[float],
            usernames: list[str],
            target_ips: list[str],
            user_agents: list[str] = None,
            source_ip: str = None,
            known_user_agents: set = None,
    ) -> dict:
        """
        Детектирует brute-force атаку.

        Returns:
            словарь с is_alert, score, features, reason
        """
        if len(timestamps) < self.min_events:
            return {
                "is_alert": False,
                "score": float("nan"),
                "features": None,
                "reason": f"Недостаточно событий: {len(timestamps)} < {self.min_events}",
            }

        if user_agents is None:
            user_agents = [""] * len(timestamps)

        stats = self.extract_features(
            timestamps, usernames, target_ips,
            user_agents, known_user_agents,
        )
        feature_vector = np.array(stats.to_feature_vector()).reshape(1, -1)

        if self.model is not None:
            dmatrix = xgb.DMatrix(feature_vector, feature_names=self.FEATURE_NAMES)
            score = float(self.model.predict(dmatrix)[0])
            is_alert = score >= self.decision_threshold

            return {
                "is_alert": is_alert,
                "score": score,
                "threshold": self.decision_threshold,
                "features": stats,
                "feature_vector": feature_vector,
                "reason": (
                    f"Score {score:.3f} >= {self.decision_threshold}" if is_alert
                    else f"Score {score:.3f} < {self.decision_threshold}"
                ),
            }

        return {
            "is_alert": False,
            "score": float("nan"),
            "features": stats,
            "feature_vector": feature_vector,
            "reason": "Модель не загружена",
        }