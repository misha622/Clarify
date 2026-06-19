"""
Р Р°СЃС‡С‘С‚ РѕРєРѕРЅРЅС‹С… СЃС‚Р°С‚РёСЃС‚РёРє РґР»СЏ Beaconing-РґРµС‚РµРєС‚РѕСЂР°.

Р’СЃРµ РїСЂРёР·РЅР°РєРё РѕСЃРЅРѕРІР°РЅС‹ РЅР° engineered features РїРѕ РѕРєРЅСѓ, Р° РЅРµ РЅР° СЃС‹СЂС‹С…
РїРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅРѕСЃС‚СЏС…. Р­С‚Рѕ РїРѕР·РІРѕР»СЏРµС‚ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ TreeExplainer РґР»СЏ SHAP
РІ СЂРµР°Р»СЊРЅРѕРј РІСЂРµРјРµРЅРё.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindowStats:
    """РЎС‚Р°С‚РёСЃС‚РёРєРё РІСЂРµРјРµРЅРЅРѕРіРѕ РѕРєРЅР° РґР»СЏ Beaconing-РґРµС‚РµРєС‚РѕСЂР°."""
    mean_interarrival_time: float
    std_interarrival_time: float
    coefficient_of_variation: float
    peak_autocorrelation_lag: int
    autocorrelation_peak_value: float
    entropy_interarrival: float
    event_count: int

    def to_feature_vector(self) -> list[float]:
        """РџСЂРµРѕР±СЂР°Р·СѓРµС‚ СЃС‚Р°С‚РёСЃС‚РёРєРё РІ РІРµРєС‚РѕСЂ РїСЂРёР·РЅР°РєРѕРІ РґР»СЏ РјРѕРґРµР»Рё."""
        return [
            self.mean_interarrival_time,
            self.std_interarrival_time,
            self.coefficient_of_variation,
            float(self.peak_autocorrelation_lag),
            self.autocorrelation_peak_value,
            self.entropy_interarrival,
            float(self.event_count),
        ]

    @property
    def feature_names(self) -> list[str]:
        """РРјРµРЅР° РїСЂРёР·РЅР°РєРѕРІ РІ С‚РѕРј Р¶Рµ РїРѕСЂСЏРґРєРµ, С‡С‚Рѕ Рё to_feature_vector."""
        return [
            "mean_interarrival_time",
            "std_interarrival_time",
            "coefficient_of_variation",
            "peak_autocorrelation_lag",
            "autocorrelation_peak_value",
            "entropy_interarrival",
            "event_count",
        ]


def calculate_interarrival_times(timestamps: list[float]) -> list[float]:
    """
    Р’С‹С‡РёСЃР»СЏРµС‚ РёРЅС‚РµСЂРІР°Р»С‹ РјРµР¶РґСѓ РїРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅС‹РјРё РІСЂРµРјРµРЅРЅС‹РјРё РјРµС‚РєР°РјРё.

    Args:
        timestamps: СЃРїРёСЃРѕРє РІСЂРµРјРµРЅРЅС‹С… РјРµС‚РѕРє РІ СЃРµРєСѓРЅРґР°С… (РѕС‚СЃРѕСЂС‚РёСЂРѕРІР°РЅРЅС‹С…)

    Returns:
        СЃРїРёСЃРѕРє РёРЅС‚РµСЂРІР°Р»РѕРІ РІ СЃРµРєСѓРЅРґР°С…
    """
    if len(timestamps) < 2:
        return []
    return [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]


def autocorrelation(intervals: list[float], max_lag: int = 20) -> np.ndarray:
    """
    Р’С‹С‡РёСЃР»СЏРµС‚ Р°РІС‚РѕРєРѕСЂСЂРµР»СЏС†РёСЋ РёРЅС‚РµСЂРІР°Р»РѕРІ.

    Args:
        intervals: СЃРїРёСЃРѕРє РёРЅС‚РµСЂРІР°Р»РѕРІ
        max_lag: РјР°РєСЃРёРјР°Р»СЊРЅС‹Р№ Р»Р°Рі РґР»СЏ СЂР°СЃС‡С‘С‚Р°

    Returns:
        РјР°СЃСЃРёРІ Р·РЅР°С‡РµРЅРёР№ Р°РІС‚РѕРєРѕСЂСЂРµР»СЏС†РёРё РґР»СЏ Р»Р°РіРѕРІ 1..max_lag
    """
    n = len(intervals)
    if n < 4:
        return np.zeros(max_lag)

    data = np.array(intervals)
    mean = np.mean(data)
    variance = np.var(data)

    if variance == 0:
        return np.ones(max_lag)  # РРґРµР°Р»СЊРЅР°СЏ РєРѕСЂСЂРµР»СЏС†РёСЏ РїСЂРё РїРѕСЃС‚РѕСЏРЅРЅРѕРј РёРЅС‚РµСЂРІР°Р»Рµ

    # РћРіСЂР°РЅРёС‡РёРІР°РµРј max_lag РґР»РёРЅРѕР№ РґР°РЅРЅС‹С…
    effective_max_lag = min(max_lag, n - 2)

    result = np.zeros(max_lag)
    for lag in range(1, effective_max_lag + 1):
        if lag < n:
            result[lag - 1] = np.mean(
                (data[:-lag] - mean) * (data[lag:] - mean)
            ) / variance

    return result


def calculate_entropy(intervals: list[float], bins: int = 10) -> float:
    """
    Р’С‹С‡РёСЃР»СЏРµС‚ СЌРЅС‚СЂРѕРїРёСЋ СЂР°СЃРїСЂРµРґРµР»РµРЅРёСЏ РёРЅС‚РµСЂРІР°Р»РѕРІ.
    РќРёР·РєР°СЏ СЌРЅС‚СЂРѕРїРёСЏ в†’ Р±РѕР»РµРµ СЂРµРіСѓР»СЏСЂРЅРѕРµ РїРѕРІРµРґРµРЅРёРµ в†’ РїРѕРґРѕР·СЂРёС‚РµР»СЊРЅРѕ.

    Args:
        intervals: СЃРїРёСЃРѕРє РёРЅС‚РµСЂРІР°Р»РѕРІ
        bins: РєРѕР»РёС‡РµСЃС‚РІРѕ Р±РёРЅРѕРІ РґР»СЏ РіРёСЃС‚РѕРіСЂР°РјРјС‹

    Returns:
        Р·РЅР°С‡РµРЅРёРµ СЌРЅС‚СЂРѕРїРёРё (Р±РёС‚С‹)
    """
    if len(intervals) < 2:
        return 0.0

    hist, _ = np.histogram(intervals, bins=bins, density=True)
    # РЈР±РёСЂР°РµРј РЅСѓР»РµРІС‹Рµ Р±РёРЅС‹ РґР»СЏ СЂР°СЃС‡РµС‚Р° СЌРЅС‚СЂРѕРїРёРё
    hist = hist[hist > 0]

    return -np.sum(hist * np.log2(hist))


def calculate_window_stats(
        timestamps: list[float],
        max_lag: int = 20
) -> WindowStats:
    """
    Р Р°СЃСЃС‡РёС‚С‹РІР°РµС‚ РІСЃРµ РѕРєРѕРЅРЅС‹Рµ СЃС‚Р°С‚РёСЃС‚РёРєРё РґР»СЏ Beaconing-РґРµС‚РµРєС‚РѕСЂР°.

    Args:
        timestamps: РѕС‚СЃРѕСЂС‚РёСЂРѕРІР°РЅРЅС‹Рµ РІСЂРµРјРµРЅРЅС‹Рµ РјРµС‚РєРё СЃРѕР±С‹С‚РёР№
        max_lag: РјР°РєСЃРёРјР°Р»СЊРЅС‹Р№ Р»Р°Рі РґР»СЏ Р°РІС‚РѕРєРѕСЂСЂРµР»СЏС†РёРё

    Returns:
        WindowStats СЃ СЂР°СЃСЃС‡РёС‚Р°РЅРЅС‹РјРё РїСЂРёР·РЅР°РєР°РјРё
    """
    intervals = calculate_interarrival_times(timestamps)
    n_intervals = len(intervals)

    if n_intervals == 0:
        return WindowStats(
            mean_interarrival_time=0.0,
            std_interarrival_time=0.0,
            coefficient_of_variation=0.0,
            peak_autocorrelation_lag=0,
            autocorrelation_peak_value=0.0,
            entropy_interarrival=0.0,
            event_count=len(timestamps),
        )

    mean_val = np.mean(intervals)
    std_val = np.std(intervals)

    # РљРѕСЌС„С„РёС†РёРµРЅС‚ РІР°СЂРёР°С†РёРё: std/mean
    # РќРёР·РєРёР№ CV в†’ СЂРµРіСѓР»СЏСЂРЅС‹Рµ РёРЅС‚РµСЂРІР°Р»С‹ в†’ РїРѕРґРѕР·СЂРёС‚РµР»СЊРЅРѕ РґР»СЏ Beaconing
    cv = std_val / mean_val if mean_val > 0 else 0.0

    # РђРІС‚РѕРєРѕСЂСЂРµР»СЏС†РёСЏ
    ac = autocorrelation(intervals, max_lag)

    if len(ac) > 0:
        peak_idx = np.argmax(ac)
        peak_value = ac[peak_idx]
        peak_lag = peak_idx + 1
    else:
        peak_lag = 0
        peak_value = 0.0

    # Р­РЅС‚СЂРѕРїРёСЏ
    entropy = calculate_entropy(intervals)

    return WindowStats(
        mean_interarrival_time=mean_val,
        std_interarrival_time=std_val,
        coefficient_of_variation=cv,
        peak_autocorrelation_lag=peak_lag,
        autocorrelation_peak_value=peak_value,
        entropy_interarrival=entropy,
        event_count=len(timestamps),
    )
