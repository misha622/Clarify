"""
Расчёт оконных статистик для Beaconing-детектора.

Все признаки основаны на engineered features по окну, а не на сырых
последовательностях. Это позволяет использовать TreeExplainer для SHAP
в реальном времени.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindowStats:
    """Статистики временного окна для Beaconing-детектора."""
    mean_interarrival_time: float
    std_interarrival_time: float
    coefficient_of_variation: float
    peak_autocorrelation_lag: int
    autocorrelation_peak_value: float
    entropy_interarrival: float
    event_count: int

    def to_feature_vector(self) -> list[float]:
        """Преобразует статистики в вектор признаков для модели."""
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
        """Имена признаков в том же порядке, что и to_feature_vector."""
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
    Вычисляет интервалы между последовательными временными метками.

    Args:
        timestamps: список временных меток в секундах (отсортированных)

    Returns:
        список интервалов в секундах
    """
    if len(timestamps) < 2:
        return []
    return [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]


def autocorrelation(intervals: list[float], max_lag: int = 20) -> np.ndarray:
    """
    Вычисляет автокорреляцию интервалов.

    Args:
        intervals: список интервалов
        max_lag: максимальный лаг для расчёта

    Returns:
        массив значений автокорреляции для лагов 1..max_lag
    """
    n = len(intervals)
    if n < 4:
        return np.zeros(max_lag)

    data = np.array(intervals)
    mean = np.mean(data)
    variance = np.var(data)

    if variance == 0:
        return np.ones(max_lag)  # Идеальная корреляция при постоянном интервале

    # Ограничиваем max_lag длиной данных
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
    Вычисляет энтропию распределения интервалов.
    Низкая энтропия → более регулярное поведение → подозрительно.

    Args:
        intervals: список интервалов
        bins: количество бинов для гистограммы

    Returns:
        значение энтропии (биты)
    """
    if len(intervals) < 2:
        return 0.0

    hist, _ = np.histogram(intervals, bins=bins, density=True)
    # Убираем нулевые бины для расчета энтропии
    hist = hist[hist > 0]

    return -np.sum(hist * np.log2(hist))


def calculate_window_stats(
        timestamps: list[float],
        max_lag: int = 20
) -> WindowStats:
    """
    Рассчитывает все оконные статистики для Beaconing-детектора.

    Args:
        timestamps: отсортированные временные метки событий
        max_lag: максимальный лаг для автокорреляции

    Returns:
        WindowStats с рассчитанными признаками
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

    # Коэффициент вариации: std/mean
    # Низкий CV → регулярные интервалы → подозрительно для Beaconing
    cv = std_val / mean_val if mean_val > 0 else 0.0

    # Автокорреляция
    ac = autocorrelation(intervals, max_lag)

    if len(ac) > 0:
        peak_idx = np.argmax(np.abs(ac))
        peak_value = ac[peak_idx]
        peak_lag = peak_idx + 1
    else:
        peak_lag = 0
        peak_value = 0.0

    # Энтропия
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