import pytest
from src.rendering.template_renderer import TemplateRenderer


class TestTemplateRendererInit:
    def test_loads_dictionary(self):
        renderer = TemplateRenderer()
        assert len(renderer.features) > 10

    def test_top_n_default(self):
        renderer = TemplateRenderer()
        assert renderer.top_n == 3


class TestConditionEvaluation:
    def test_value_greater_than(self):
        renderer = TemplateRenderer()
        assert renderer._evaluate_condition("value > 0.5", {"value": 0.8}) is True
        assert renderer._evaluate_condition("value > 0.5", {"value": 0.3}) is False

    def test_value_is_not_none(self):
        renderer = TemplateRenderer()
        assert renderer._evaluate_condition("value is not None", {"value": 1.0}) is True
        assert renderer._evaluate_condition("value is not None", {"value": None}) is False

    def test_baseline_check(self):
        renderer = TemplateRenderer()
        assert renderer._evaluate_condition(
            "baseline is not None and baseline > 0",
            {"baseline": 5.0}
        ) is True
        assert renderer._evaluate_condition(
            "baseline is not None and baseline > 0",
            {"baseline": 0}
        ) is False

    def test_compound_condition(self):
        renderer = TemplateRenderer()
        assert renderer._evaluate_condition(
            "value is not None and value > 10",
            {"value": 15}
        ) is True
        assert renderer._evaluate_condition(
            "value is not None and value > 10",
            {"value": 5}
        ) is False

    def test_invalid_condition_returns_false(self):
        renderer = TemplateRenderer()
        assert renderer._evaluate_condition("undefined_var > 0", {}) is False


class TestFeatureRendering:
    def test_render_known_feature(self):
        renderer = TemplateRenderer()
        text = renderer.render_feature(
            "f_beacon_001", 0.35,
            {"value": 0.12, "baseline": 0.45, "ratio": 0.27, "ip": "10.0.0.1"}
        )
        assert len(text) > 10
        assert "10.0.0.1" not in text  # beacon features use {ip} not {source_ip}

    def test_render_new_ip_no_baseline(self):
        renderer = TemplateRenderer()
        text = renderer.render_feature(
            "f_brute_001", 0.62,
            {"value": 55, "baseline": 0, "source_ip": "203.0.113.45"}
        )
        assert "Впервые" in text or "203.0.113.45" in text

    def test_render_unknown_feature_fallback(self):
        renderer = TemplateRenderer()
        text = renderer.render_feature(
            "nonexistent_feature", 0.5,
            {"value": 42}
        )
        assert "42" in text or "nonexistent_feature" in text

    def test_render_top_features_sorting(self):
        renderer = TemplateRenderer()
        results = renderer.render_top_features(
            ["f_beacon_001", "f_beacon_003", "f_beacon_006"],
            [0.2, 0.5, 0.1],
            [
                {"value": 0.3, "baseline": 0.5, "ip": "10.0.0.1"},
                {"value": 1.5, "baseline": 2.0, "ip": "10.0.0.1"},
                {"value": 5.0, "baseline": None, "ip": "10.0.0.1"},
            ]
        )
        assert len(results) == 3
        assert results[0]["shap_abs"] >= results[1]["shap_abs"]
        assert results[1]["shap_abs"] >= results[2]["shap_abs"]

    def test_render_short_template(self):
        renderer = TemplateRenderer()
        text = renderer.render_feature(
            "f_beacon_001", 0.35,
            {"value": 0.12, "baseline": 0.45, "ratio": 0.27, "ip": "10.0.0.1"},
            use_short=True
        )
        assert len(text) < 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
