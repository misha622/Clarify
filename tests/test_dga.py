import pytest
import numpy as np
from src.detectors.dga import DGADetector, DGAFeatures


class TestDGAStaticMethods:
    def test_shannon_entropy_empty(self):
        assert DGADetector.shannon_entropy("") == 0.0

    def test_shannon_entropy_constant(self):
        assert DGADetector.shannon_entropy("aaaa") == 0.0

    def test_shannon_entropy_random_like(self):
        entropy = DGADetector.shannon_entropy("a3f7b2c9e1d4")
        assert entropy > 3.0, f"Expected high entropy, got {entropy:.2f}"

    def test_shannon_entropy_legitimate(self):
        entropy = DGADetector.shannon_entropy("google")
        assert 1.5 < entropy < 3.0, f"Expected moderate entropy, got {entropy:.2f}"

    def test_vowel_consonant_ratio(self):
        ratio = DGADetector.vowel_consonant_ratio("abcdef")
        assert 0.2 < ratio <= 0.5

    def test_ngram_score_english(self):
        score = DGADetector.ngram_score("google.com")
        assert score >= 0.0, f"English-like domain should score >=0, got {score:.4f}"

    def test_ngram_score_dga(self):
        score = DGADetector.ngram_score("a3f7b2c9e1d4.xyz")
        assert score < 0.01, f"DGA domain should score near 0, got {score:.4f}"

    def test_extract_tld(self):
        assert DGADetector.extract_tld("google.com") == "com"
        assert DGADetector.extract_tld("sub.domain.co.uk") == "uk"


class TestDGAFeatures:
    def test_feature_extraction(self):
        detector = DGADetector()
        domains = ["google.com", "github.com", "stackoverflow.com"]
        nxdomain = [False, False, False]
        
        features = detector.extract_features(domains, nxdomain)
        
        assert features.total_queries == 3
        assert features.unique_domains == 3
        assert features.nxdomain_rate == 0.0
        assert features.mean_entropy < 3.5

    def test_feature_vector_shape(self):
        detector = DGADetector()
        domains = ["google.com", "github.com"]
        nxdomain = [False, False]
        
        features = detector.extract_features(domains, nxdomain)
        fv = features.to_feature_vector()
        
        assert fv.shape == (9,)

    def test_feature_names(self):
        features = DGAFeatures(
            mean_entropy=3.0, max_entropy=4.0, mean_domain_length=15.0,
            mean_vowel_consonant_ratio=0.3, mean_ngram_score=0.01,
            unique_domains=50, nxdomain_rate=0.9, unique_tld_count=5,
            total_queries=60
        )
        assert "mean_entropy" in features.feature_names
        assert "nxdomain_rate" in features.feature_names


class TestDGADetection:
    def test_insufficient_queries(self):
        detector = DGADetector()
        result = detector.detect(
            ["google.com"], [False], source_ip="10.0.0.1"
        )
        assert result["is_alert"] is False
        assert "Недостаточно" in result["reason"]

    def test_without_model(self):
        detector = DGADetector()
        detector.model = None
        domains = [f"domain{i}.com" for i in range(15)]
        nxdomain = [False] * 15
        
        result = detector.detect(domains, nxdomain, source_ip="10.0.0.1")
        assert result["is_alert"] is False
        assert result["reason"] == "Модель не загружена"

    def test_features_in_result(self):
        detector = DGADetector()
        detector.model = None
        domains = [f"domain{i}.com" for i in range(15)]
        nxdomain = [False] * 15
        
        result = detector.detect(domains, nxdomain, source_ip="10.0.0.1")
        assert result["features"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
