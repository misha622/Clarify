"""
DGA Detector для Clarify.

Детектирует Domain Generation Algorithm (DGA) атаки на уровне источника.
Признаки делятся на:
- Лексические (считаются с одного домена): энтропия, длина, n-граммы
- Поведенческие (считаются по окну на источник): NXDOMAIN rate, кол-во уникальных доменов

Совместим с TreeExplainer для SHAP-объяснений.

Использование:
    detector = DGADetector()
    result = detector.detect(domains, nxdomain_flags, source_ip="192.168.1.100")
"""

import math
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from collections import Counter

import numpy as np
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DGAFeatures:
    """Признаки DGA для одного источника за окно."""
    # Лексические (агрегированные по всем доменам от источника)
    mean_entropy: float
    max_entropy: float
    mean_domain_length: float
    mean_vowel_consonant_ratio: float
    mean_ngram_score: float

    # Поведенческие
    unique_domains: int
    nxdomain_rate: float
    unique_tld_count: int
    total_queries: int

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.mean_entropy,
            self.max_entropy,
            self.mean_domain_length,
            self.mean_vowel_consonant_ratio,
            self.mean_ngram_score,
            float(self.unique_domains),
            self.nxdomain_rate,
            float(self.unique_tld_count),
            float(self.total_queries),
        ], dtype=np.float64)

    @property
    def feature_names(self) -> List[str]:
        return [
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


class DGADetector:
    """
    Детектор DGA-атак.

    Анализирует DNS-запросы от одного источника за временное окно.
    Использует XGBoost на engineered features → совместим с TreeExplainer.

    Порог выборки: минимум 10 запросов от источника (не per-domain).
    """

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

    # Частоты биграмм в английском языке (для ngram_score)
    ENGLISH_BIGRAM_FREQ = {
        'th': 0.0356, 'he': 0.0307, 'in': 0.0243, 'er': 0.0205, 'an': 0.0199,
        'on': 0.0176, 'at': 0.0149, 'en': 0.0145, 'nd': 0.0135, 'ti': 0.0134,
        'es': 0.0133, 'or': 0.0128, 'te': 0.0120, 'of': 0.0117, 'ed': 0.0117,
        'is': 0.0113, 'it': 0.0112, 'al': 0.0109, 'ar': 0.0107, 'st': 0.0105,
    }

    # Известные легитимные TLD
    LEGITIMATE_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'ru', 'de',
        'uk', 'fr', 'jp', 'cn', 'br', 'au', 'ca', 'in', 'it', 'es',
    }

    def __init__(self, config_path: str = "config/detectors.yaml"):
        self.config = self._load_config(config_path)
        dga_cfg = self.config["detectors"].get("dga", {})

        self.min_queries = dga_cfg.get("min_queries_from_source", 10)
        self.decision_threshold = dga_cfg.get("decision_threshold", 0.5)

        self.model: Optional[xgb.Booster] = None
        model_path = dga_cfg.get("model_path")
        if model_path and Path(model_path).exists():
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            logger.info(f"Модель DGA загружена: {model_path}")
        else:
            logger.warning("Модель DGA не загружена. Используйте train_dga.py")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def shannon_entropy(s: str) -> float:
        """Вычисляет энтропию Шеннона для строки."""
        if not s:
            return 0.0
        counter = Counter(s.lower())
        length = len(s)
        entropy = 0.0
        for count in counter.values():
            p = count / length
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def vowel_consonant_ratio(s: str) -> float:
        """Соотношение гласных к согласным."""
        vowels = set('aeiou')
        s_lower = s.lower()
        v_count = sum(1 for c in s_lower if c in vowels)
        c_count = sum(1 for c in s_lower if c.isalpha() and c not in vowels)
        return v_count / c_count if c_count > 0 else 1.0

    @staticmethod
    def ngram_score(domain: str) -> float:
        """
        Оценивает, насколько домен похож на английский язык по биграммам.
        Низкий score → непохож на язык → возможно DGA.
        """
        name = domain.split('.')[0].lower()
        if len(name) < 2:
            return 0.0

        score = 0.0
        bigrams_count = 0

        for i in range(len(name) - 1):
            bigram = name[i:i + 2]
            if bigram in DGADetector.ENGLISH_BIGRAM_FREQ:
                score += DGADetector.ENGLISH_BIGRAM_FREQ[bigram]
            bigrams_count += 1

        return score / bigrams_count if bigrams_count > 0 else 0.0

    @staticmethod
    def extract_tld(domain: str) -> str:
        """Извлекает TLD из домена."""
        parts = domain.split('.')
        return parts[-1] if len(parts) > 1 else ''

    def extract_features(
            self,
            domains: List[str],
            nxdomain_flags: List[bool],
    ) -> DGAFeatures:
        """
        Извлекает признаки DGA из списка доменов.

        Args:
            domains: список доменных имён
            nxdomain_flags: список флагов NXDOMAIN (True = не существует)
        """
        n = len(domains)

        # Лексические признаки для каждого домена
        entropies = [self.shannon_entropy(d) for d in domains]
        lengths = [len(d.split('.')[0]) for d in domains]
        vc_ratios = [self.vowel_consonant_ratio(d) for d in domains]
        ngram_scores = [self.ngram_score(d) for d in domains]

        # Поведенческие признаки
        unique_domains = len(set(domains))
        nxdomain_rate = sum(nxdomain_flags) / n if n > 0 else 0.0

        tlds = [self.extract_tld(d) for d in domains]
        unique_tld_count = len(set(tlds))

        return DGAFeatures(
            mean_entropy=np.mean(entropies),
            max_entropy=np.max(entropies),
            mean_domain_length=np.mean(lengths),
            mean_vowel_consonant_ratio=np.mean(vc_ratios),
            mean_ngram_score=np.mean(ngram_scores),
            unique_domains=unique_domains,
            nxdomain_rate=nxdomain_rate,
            unique_tld_count=unique_tld_count,
            total_queries=n,
        )

    def detect(
            self,
            domains: List[str],
            nxdomain_flags: List[bool],
            source_ip: str = None,
    ) -> dict:
        """
        Детектирует DGA-атаку.

        Args:
            domains: список доменов, запрошенных источником
            nxdomain_flags: флаги NXDOMAIN для каждого домена
            source_ip: IP источника (для логирования)

        Returns:
            dict с is_alert, score, features, reason
        """
        if len(domains) < self.min_queries:
            return {
                "is_alert": False,
                "score": float("nan"),
                "features": None,
                "reason": (
                    f"Недостаточно запросов от источника {source_ip}: "
                    f"{len(domains)} < {self.min_queries}"
                ),
            }

        features = self.extract_features(domains, nxdomain_flags)
        feature_vector = features.to_feature_vector().reshape(1, -1)

        if self.model is not None:
            dmatrix = xgb.DMatrix(feature_vector, feature_names=self.FEATURE_NAMES)
            score = float(self.model.predict(dmatrix)[0])
            is_alert = score >= self.decision_threshold

            return {
                "is_alert": is_alert,
                "score": score,
                "threshold": self.decision_threshold,
                "features": features,
                "feature_vector": feature_vector,
                "reason": (
                    f"Score {score:.3f} >= {self.decision_threshold}" if is_alert
                    else f"Score {score:.3f} < {self.decision_threshold}"
                ),
            }

        return {
            "is_alert": False,
            "score": float("nan"),
            "features": features,
            "feature_vector": feature_vector,
            "reason": "Модель не загружена",
        }