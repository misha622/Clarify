import pytest
import numpy as np
from src.detectors.brute_force import BruteForceDetector, BruteForceWindowStats


class TestBruteForceFeatures:
    def test_extract_features_basic(self):
        detector = BruteForceDetector()
        timestamps = [0.0, 2.0, 4.0, 6.0, 8.0]
        usernames = ["root", "admin", "root", "test", "admin"]
        targets = ["192.168.1.1", "192.168.1.1", "192.168.1.2", "192.168.1.1", "192.168.1.1"]
        ua = ["a", "b", "a", "c", "a"]
        
        stats = detector.extract_features(timestamps, usernames, targets, ua)
        
        assert stats.total_attempts == 5
        assert stats.unique_usernames == 3
        assert stats.target_hosts == 2
        assert stats.auth_failure_rate > 0

    def test_extract_features_single_event(self):
        detector = BruteForceDetector()
        stats = detector.extract_features(
            [0.0], ["root"], ["192.168.1.1"], ["a"]
        )
        assert stats.total_attempts == 1
        assert stats.unique_usernames == 1

    def test_feature_vector_shape(self):
        detector = BruteForceDetector()
        stats = detector.extract_features(
            [0.0, 1.0, 2.0], ["a", "b", "a"], ["1", "1", "1"], ["x", "y", "x"]
        )
        fv = stats.to_feature_vector()
        assert len(fv) == 5

    def test_feature_names_match(self):
        detector = BruteForceDetector()
        stats = detector.extract_features(
            [0.0, 1.0], ["a", "b"], ["1", "2"], ["x", "y"]
        )
        assert stats.feature_names == [
            "auth_failure_rate", "unique_usernames", "target_hosts",
            "new_user_agents", "total_attempts"
        ]

    def test_new_user_agents_detection(self):
        detector = BruteForceDetector()
        known = {"a", "b"}
        stats = detector.extract_features(
            [0.0, 1.0, 2.0, 3.0],
            ["u1", "u2", "u3", "u4"],
            ["1", "1", "1", "1"],
            ["a", "b", "c", "d"],
            known_user_agents=known
        )
        assert stats.new_user_agents == 2  # c, d


class TestBruteForceDetection:
    def test_insufficient_events(self):
        detector = BruteForceDetector()
        result = detector.detect(
            [0.0], ["root"], ["192.168.1.1"], ["a"], source_ip="10.0.0.1"
        )
        assert result["is_alert"] is False
        assert "Недостаточно" in result["reason"]

    def test_sufficient_events_without_model(self):
        detector = BruteForceDetector()
        detector.model = None
        timestamps = list(range(20))
        usernames = [f"user_{i}" for i in range(20)]
        targets = ["192.168.1.1"] * 20
        ua = ["a"] * 20
        
        result = detector.detect(timestamps, usernames, targets, ua, source_ip="10.0.0.1")
        assert result["is_alert"] is False
        assert result["reason"] == "Модель не загружена"

    def test_detect_with_model(self, tmp_path):
        import xgboost as xgb
        import yaml
        from pathlib import Path
        
        # Обучаем маленькую модель
        rng = np.random.RandomState(42)
        X = np.random.rand(100, 5)
        X[:70, 0] = rng.uniform(0, 5, 70); X[70:, 0] = rng.uniform(20, 60, 30)
        y = np.array([0]*70 + [1]*30)
        
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "seed": 42},
            dtrain, num_boost_round=20
        )
        
        model_path = str(tmp_path / "test_bf.json")
        model.save_model(model_path)
        
        # Обновляем конфиг
        config_path = Path("config/detectors.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        old_path = config["detectors"]["brute_force"].get("model_path")
        config["detectors"]["brute_force"]["model_path"] = model_path
        config["detectors"]["brute_force"]["decision_threshold"] = 0.5
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        try:
            detector = BruteForceDetector()
            detector.model = model
            detector.decision_threshold = 0.5
            
            timestamps = list(np.linspace(0, 120, 25))
            usernames = [f"user_{i}" for i in range(25)]
            targets = ["192.168.1.1"] * 25
            ua = ["a"] * 25
            
            result = detector.detect(timestamps, usernames, targets, ua, source_ip="10.0.0.1")
            assert "is_alert" in result
            assert "score" in result
        finally:
            if old_path:
                config["detectors"]["brute_force"]["model_path"] = old_path
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

    def test_detect_default_user_agents(self):
        detector = BruteForceDetector()
        detector.model = None
        timestamps = list(range(15))
        usernames = [f"u{i}" for i in range(15)]
        targets = ["192.168.1.1"] * 15
        
        result = detector.detect(timestamps, usernames, targets, source_ip="10.0.0.1")
        assert result["is_alert"] is False

    def test_feature_vector_in_result(self):
        detector = BruteForceDetector()
        detector.model = None
        timestamps = list(range(15))
        usernames = [f"u{i}" for i in range(15)]
        targets = ["192.168.1.1"] * 15
        
        result = detector.detect(timestamps, usernames, targets, source_ip="10.0.0.1")
        assert result["features"] is not None
        assert result["feature_vector"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

