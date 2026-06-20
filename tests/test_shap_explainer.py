import pytest
import numpy as np
import xgboost as xgb
from src.explainers.shap_explainer import ShapExplainer, AlertExplanation, FeatureExplanation

FEATURE_NAMES = ["feature_a", "feature_b", "feature_c"]


@pytest.fixture
def dummy_model():
    """Обучает крошечную XGBoost-модель на случайных данных."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 3)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    model = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "seed": 42},
        dtrain, num_boost_round=20
    )
    return model


class TestShapExplainerInit:
    def test_initialization(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        assert explainer.top_n == 3
        assert explainer.feature_names == FEATURE_NAMES
        assert explainer.expected_value is not None

    def test_custom_top_n(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=2)
        assert explainer.top_n == 2


class TestShapExplainerExplain:
    def test_explain_returns_correct_structure(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        sample = np.random.rand(1, 3)
        result = explainer.explain(sample, "test_alert", context={"ip": "10.0.0.1"})
        
        assert isinstance(result, AlertExplanation)
        assert result.alert_type == "test_alert"
        assert len(result.top_features) == 3
        assert result.total_features == 3
        assert result.latency_ms > 0

    def test_top_features_sorted_by_abs_shap(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        sample = np.random.rand(1, 3)
        result = explainer.explain(sample, "test_alert")
        
        for i in range(len(result.top_features) - 1):
            assert result.top_features[i].shap_abs >= result.top_features[i+1].shap_abs

    def test_feature_explanation_has_required_fields(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        sample = np.random.rand(1, 3)
        result = explainer.explain(sample, "test_alert")
        
        for feat in result.top_features:
            assert isinstance(feat, FeatureExplanation)
            assert feat.feature_id in FEATURE_NAMES
            assert isinstance(feat.shap_value, float)
            assert isinstance(feat.feature_value, float)

    def test_explain_respects_top_n(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=1)
        sample = np.random.rand(1, 3)
        result = explainer.explain(sample, "test_alert")
        assert len(result.top_features) == 1

    def test_latency_is_reasonable(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        sample = np.random.rand(1, 3)
        result = explainer.explain(sample, "test_alert")
        assert result.latency_ms < 1000, f"Latency too high: {result.latency_ms}ms"


class TestShapExplainerBatch:
    def test_explain_batch(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        samples = np.random.rand(5, 3)
        types = ["alert"] * 5
        results = explainer.explain_batch(samples, types)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, AlertExplanation)
            assert len(result.top_features) == 3

    def test_batch_latency_per_sample(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        samples = np.random.rand(10, 3)
        types = ["alert"] * 10
        results = explainer.explain_batch(samples, types)
        
        for result in results:
            assert result.latency_ms > 0
            assert result.latency_ms < 1000


class TestGlobalFeatureImportance:
    def test_get_feature_importance_global(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        importance = explainer.get_feature_importance_global()
        
        assert len(importance) == 3
        for item in importance:
            assert "feature_name" in item
            assert "mean_abs_shap" in item
            assert item["feature_name"] in FEATURE_NAMES

    def test_importance_sorted_descending(self, dummy_model):
        explainer = ShapExplainer(dummy_model, FEATURE_NAMES, top_n=3)
        importance = explainer.get_feature_importance_global()
        
        for i in range(len(importance) - 1):
            assert importance[i]["mean_abs_shap"] >= importance[i+1]["mean_abs_shap"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
