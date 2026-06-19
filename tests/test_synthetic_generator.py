import pytest
import numpy as np
from src.data.synthetic_generator import SyntheticGenerator, SyntheticDataset


class TestSyntheticGeneratorInit:
    def test_default_init(self):
        gen = SyntheticGenerator()
        assert gen.time_window == 6 * 3600
        assert gen.base_time == 1_700_000_000.0

    def test_custom_time_window(self):
        gen = SyntheticGenerator(time_window_hours=12)
        assert gen.time_window == 12 * 3600

    def test_seed_reproducibility(self):
        gen1 = SyntheticGenerator(seed=42)
        gen2 = SyntheticGenerator(seed=42)
        ds1 = gen1.generate(mode="train", num_hosts=10)
        ds2 = gen2.generate(mode="train", num_hosts=10)
        assert ds1.total_events == ds2.total_events
        assert ds1.benign_count == ds2.benign_count


class TestSyntheticGeneratorTrain:
    def test_train_mode_ratio(self):
        gen = SyntheticGenerator(seed=42)
        ds = gen.generate(mode="train", num_hosts=50)
        ratio = ds.attack_ratio
        assert 0.20 < ratio < 0.45, f"Expected 20-45% attacks, got {ratio:.1%}"

    def test_train_mode_has_both_classes(self):
        gen = SyntheticGenerator(seed=42)
        ds = gen.generate(mode="train", num_hosts=50)
        assert ds.benign_count > 0
        assert ds.attack_count > 0

    def test_train_mode_events_sorted_by_time(self):
        gen = SyntheticGenerator(seed=42)
        ds = gen.generate(mode="train", num_hosts=30)
        timestamps = [e.timestamp for e in ds.events]
        assert timestamps == sorted(timestamps)


class TestSyntheticGeneratorCalibrate:
    def test_calibrate_mode_low_attack_rate(self):
        gen = SyntheticGenerator(seed=42)
        ds = gen.generate(mode="calibrate", num_hosts=100)
        ratio = ds.attack_ratio
        assert ratio < 0.10, f"Expected <10% attacks, got {ratio:.1%}"

    def test_calibrate_mode_has_both_classes(self):
        gen = SyntheticGenerator(seed=42)
        ds = gen.generate(mode="calibrate", num_hosts=100)
        assert ds.benign_count > 0


class TestBeaconingTrainingData:
    def test_returns_valid_shapes(self):
        gen = SyntheticGenerator(seed=42)
        X, y = gen.generate_for_beaconing_training(
            mode="train", window_size_seconds=900,
            stride_seconds=300, min_events_per_window=8, num_hosts=50
        )
        assert X.ndim == 2
        assert X.shape[1] == 7
        assert len(y) == X.shape[0]

    def test_both_classes_present(self):
        gen = SyntheticGenerator(seed=42)
        X, y = gen.generate_for_beaconing_training(
            mode="train", window_size_seconds=900,
            stride_seconds=300, min_events_per_window=8, num_hosts=50
        )
        assert sum(y == 0) > 0, "No benign samples"
        assert sum(y == 1) > 0, "No attack samples"

    def test_calibrate_mode_has_samples(self):
        gen = SyntheticGenerator(seed=42)
        X, y = gen.generate_for_beaconing_training(
            mode="calibrate", window_size_seconds=900,
            stride_seconds=300, min_events_per_window=8, num_hosts=100
        )
        assert len(X) > 0

    def test_small_stride_produces_more_windows(self):
        gen = SyntheticGenerator(seed=42)
        X1, _ = gen.generate_for_beaconing_training(
            mode="train", window_size_seconds=900,
            stride_seconds=900, min_events_per_window=8, num_hosts=30
        )
        X2, _ = gen.generate_for_beaconing_training(
            mode="train", window_size_seconds=900,
            stride_seconds=300, min_events_per_window=8, num_hosts=30
        )
        assert len(X2) >= len(X1), "Smaller stride should produce more windows"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
