"""
Тесты для Beaconing-детектора.
"""

import pytest
import numpy as np
from src.features.window_stats import (
    calculate_interarrival_times,
    calculate_window_stats,
    autocorrelation,
)
from src.utils.thresholds import check_min_intervals
from src.detectors.beaconing import BeaconingDetector, BeaconingResult


class TestInterarrivalTimes:
    def test_empty_list(self):
        assert calculate_interarrival_times([]) == []

    def test_single_timestamp(self):
        assert calculate_interarrival_times([1.0]) == []

    def test_two_timestamps(self):
        result = calculate_interarrival_times([1.0, 3.0])
        assert result == [2.0]

    def test_multiple_timestamps(self):
        result = calculate_interarrival_times([1.0, 3.0, 6.0, 10.0])
        assert result == [2.0, 3.0, 4.0]


class TestAutocorrelation:
    def test_constant_intervals(self):
        """При постоянных интервалах автокорреляция должна быть 1."""
        intervals = [10.0] * 20
        ac = autocorrelation(intervals, max_lag=5)
        assert len(ac) == 5
        # Все значения должны быть близки к 1
        assert all(abs(v - 1.0) < 0.01 for v in ac)

    def test_random_intervals(self):
        """При случайных интервалах автокорреляция должна быть низкой."""
        rng = np.random.RandomState(42)
        intervals = rng.exponential(10.0, 50).tolist()
        ac = autocorrelation(intervals, max_lag=10)
        # Значения должны быть в диапазоне [-1, 1]
        assert all(-1.0 <= v <= 1.0 for v in ac)
        # Для случайных данных значения обычно < 0.3
        assert all(abs(v) < 0.5 for v in ac)


class TestWindowStats:
    def test_beaconing_pattern(self):
        """Beaconing-паттерн: регулярные интервалы ~10 сек с джиттером ±1."""
        rng = np.random.RandomState(42)
        timestamps = [0.0]
        for _ in range(20):
            timestamps.append(timestamps[-1] + 10.0 + rng.normal(0, 1.0))

        stats = calculate_window_stats(timestamps)

        assert stats.event_count == 21
        # CV должен быть низким для регулярного паттерна
        assert stats.coefficient_of_variation < 0.3
        # Автокорреляция должна быть значимой
        assert stats.autocorrelation_peak_value > 0.5

    def test_legitimate_traffic(self):
        """Легитимный трафик: нерегулярные интервалы."""
        rng = np.random.RandomState(43)
        timestamps = [0.0]
        for _ in range(20):
            # Экспоненциальное распределение — нерегулярное
            timestamps.append(timestamps[-1] + rng.exponential(20.0))

        stats = calculate_window_stats(timestamps)

        # CV должен быть высоким для нерегулярного трафика
        assert stats.coefficient_of_variation > 0.5


class TestConfidenceGate:
    def test_insufficient_intervals(self):
        intervals = [1.0, 2.0, 3.0]  # Всего 3, нужно минимум 15
        result = check_min_intervals(intervals, min_required=15)
        assert result.should_proceed is False
        assert result.reason is not None
        assert "Недостаточно интервалов" in result.reason

    def test_sufficient_intervals(self):
        intervals = list(range(20))  # 20 интервалов
        result = check_min_intervals(intervals, min_required=15)
        assert result.should_proceed is True
        assert result.reason is None


class TestBeaconingDetector:
    @pytest.fixture
    def detector(self):
        return BeaconingDetector()

    def test_insufficient_data_returns_no_alert(self, detector):
        """С малым количеством событий детектор не должен алертить."""
        timestamps = list(range(10))  # Всего 9 интервалов, нужно 15+
        result = detector.detect(timestamps, source_id="test_host")

        assert result.is_alert is False
        assert result.should_alert is False
        assert "Недостаточно интервалов" in result.reason

    def test_beaconing_pattern_without_model(self, detector):
        """
        Без обученной модели детектор возвращает статистики,
        но не даёт вердикта.
        """
        rng = np.random.RandomState(42)
        timestamps = [0.0]
        for _ in range(20):
            timestamps.append(timestamps[-1] + 10.0 + rng.normal(0, 0.5))

        result = detector.detect(timestamps, source_id="test_host")

        # Без модели — алерта нет
        assert result.is_alert is False
        assert result.reason == "Модель не загружена"
        # Но статистики посчитаны
        assert result.window_stats.event_count == 21
        assert result.window_stats.coefficient_of_variation < 0.2

    def test_train_and_detect(self, detector, tmp_path):
        """Сквозной тест: обучение на синтетике и детекция."""
        rng = np.random.RandomState(42)

        # Генерируем обучающие данные: 70/30 benign/beaconing
        X_list, y_list = [], []

        for _ in range(70):  # Benign: нерегулярный трафик
            ts = [0.0]
            for _ in range(20):
                ts.append(ts[-1] + rng.exponential(20.0))
            stats = calculate_window_stats(ts)
            X_list.append(stats.to_feature_vector())
            y_list.append(0)

        for _ in range(30):  # Beaconing: регулярный трафик
            ts = [0.0]
            for _ in range(20):
                ts.append(ts[-1] + 10.0 + rng.normal(0, 1.0))
            stats = calculate_window_stats(ts)
            X_list.append(stats.to_feature_vector())
            y_list.append(1)

        X = np.array(X_list)
        y = np.array(y_list)

        # Перемешиваем
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        # Разделяем train/val
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # Обучаем
        model_path = str(tmp_path / "test_beaconing.json")
        metrics = detector.train(X_train, y_train, X_val, y_val, model_path)

        assert "best_iteration" in metrics
        assert Path(model_path).exists()

        # Калибруем порог
        threshold = detector.tune_threshold(
            X_val, y_val, target_precision=0.8
        )
        assert 0.0 < threshold < 1.0

        # Тестируем детекцию на новом beaconing-паттерне
        ts_beacon = [0.0]
        for _ in range(20):
            ts_beacon.append(ts_beacon[-1] + 10.0 + rng.normal(0, 0.5))

        result = detector.detect(ts_beacon, source_id="test_beacon")
        # С откалиброванным порогом должно детектиться
        assert result.is_alert, f"Score: {result.score}, threshold: {threshold}"
        assert result.window_stats.coefficient_of_variation < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])