"""
Рендерер NL-шаблонов для SHAP-объяснений.

Превращает feature_id + SHAP-значение + контекст в человекочитаемый текст.
Использует условные шаблоны из feature_dictionary.yaml.

НЕ использует LLM — все шаблоны детерминированы.
"""

import yaml
import re
from pathlib import Path
from typing import Any, Optional


class TemplateRenderer:
    """
    Рендерер NL-шаблонов для объяснения алертов.

    Загружает словарь признаков из YAML и для каждого признака
    выбирает подходящий шаблон по условию, затем подставляет значения.
    """

    def __init__(self, dictionary_path: str = "config/feature_dictionary.yaml"):
        """
        Args:
            dictionary_path: путь к YAML-файлу словаря признаков
        """
        with open(dictionary_path, "r", encoding="utf-8") as f:
            self.dictionary = yaml.safe_load(f)

        self.features = self.dictionary.get("features", {})
        self.renderer_config = self.dictionary.get("renderer", {})
        self.top_n = self.renderer_config.get("top_n_features", 3)
        self.float_precision = self.renderer_config.get("float_precision", 2)
        self.fallback_template = self.renderer_config.get(
            "fallback_template",
            "Признак {feature_name}: значение = {value}"
        )
        self.fallback_template_short = self.renderer_config.get(
            "fallback_template_short",
            "{feature_name}={value}"
        )

    def _evaluate_condition(self, condition: str, context: dict) -> bool:
        """
        Простейший вычислитель условий.

        Поддерживает:
        - "baseline > 0"
        - "baseline is not None and baseline > 0"
        - "value > 0.7 and baseline > 0"
        - "value is not None"
        - "value is not None and value > 10"
        """
        import re

        # Заменяем "X is not None" оставляем как есть — eval поймёт
        # Заменяем "X is None" оставляем как есть
        # Условия уже должны быть в Python-синтаксисе

        safe_context = {}
        for key, val in context.items():
            safe_context[key] = val

        try:
            result = eval(condition, {"__builtins__": {}}, safe_context)
            return bool(result)
        except Exception:
            return False

    def _format_value(self, value: Any) -> str:
        """Форматирует значение для подстановки в шаблон."""
        if value is None:
            return "нет данных"
        if isinstance(value, float):
            return f"{value:.{self.float_precision}f}"
        return str(value)

    def render_feature(
            self,
            feature_id: str,
            shap_value: float,
            context: dict,
            use_short: bool = False,
    ) -> str:
        """
        Рендерит объяснение для одного признака.

        Args:
            feature_id: ID признака из словаря
            shap_value: значение SHAP для этого признака
            context: словарь с переменными (value, baseline, ratio, ip, ...)
            use_short: использовать короткий шаблон (для сводки)

        Returns:
            строка с объяснением
        """
        # Ищем признак в словаре
        feature_def = self.features.get(feature_id)

        if feature_def is None:
            return self.fallback_template.format(
                feature_name=feature_id,
                value=context.get("value", "?"),
            )

        # Дополняем контекст
        full_context = {
            **context,
            "feature_name": feature_def.get("human_name", feature_id),
        }

        # Перебираем шаблоны, ищем первый подходящий по условию
        template_key = "template_short" if use_short else "template"
        templates = feature_def.get("nl_templates", [])

        for tpl in templates:
            condition = tpl.get("condition", "true")
            if self._evaluate_condition(condition, full_context):
                template_str = tpl[template_key]

                # Пробуем отформатировать с числами как есть
                try:
                    return template_str.format(**full_context)
                except (KeyError, ValueError, TypeError):
                    pass

                # Если не вышло — форматируем числа в строки и пробуем ещё раз
                try:
                    formatted = {}
                    for k, v in full_context.items():
                        if isinstance(v, float):
                            formatted[k] = f"{v:.{self.float_precision}f}"
                        elif v is None:
                            formatted[k] = "нет данных"
                        else:
                            formatted[k] = v
                    return template_str.format(**formatted)
                except (KeyError, ValueError, TypeError):
                    pass

                # Совсем не вышло — fallback
                return self.fallback_template.format(
                    feature_name=feature_def.get("human_name", feature_id),
                    value=context.get("value", "?"),
                )

        # Ни один шаблон не подошёл
        return self.fallback_template.format(
            feature_name=feature_def.get("human_name", feature_id),
            value=context.get("value", "?"),
        )

    def render_top_features(
            self,
            feature_ids: list[str],
            shap_values: list[float],
            feature_contexts: list[dict],
    ) -> list[dict]:
        """
        Рендерит топ-N признаков для карточки алерта.

        Args:
            feature_ids: список ID признаков
            shap_values: соответствующие значения SHAP
            feature_contexts: контексты для каждого признака

        Returns:
            список словарей с ключами:
            - feature_id
            - shap_value
            - explanation (полный текст)
            - explanation_short (короткий текст)
            - shap_abs (модуль SHAP для сортировки)
        """
        results = []

        for feat_id, shap_val, ctx in zip(feature_ids, shap_values, feature_contexts):
            explanation = self.render_feature(feat_id, shap_val, ctx, use_short=False)
            explanation_short = self.render_feature(feat_id, shap_val, ctx, use_short=True)

            results.append({
                "feature_id": feat_id,
                "shap_value": shap_val,
                "shap_abs": abs(shap_val),
                "explanation": explanation,
                "explanation_short": explanation_short,
            })

        # Сортируем по abs(SHAP) — это ПРАВИЛО, severity_weight не вмешивается
        results.sort(key=lambda r: r["shap_abs"], reverse=True)

        # Берём топ-N
        return results[:self.top_n]

    def get_feature_severity(self, feature_id: str) -> float:
        """Возвращает severity_weight признака (для глобальной приоритизации)."""
        feature_def = self.features.get(feature_id, {})
        return feature_def.get("severity_weight", 0.5)


# ------------------------------------------------------------------
# Быстрый тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    renderer = TemplateRenderer()

    print("=" * 70)
    print("Тест рендеринга признаков:")
    print()

    # Тест 1: Beaconing CV
    text = renderer.render_feature(
        feature_id="f_beacon_001",
        shap_value=0.35,
        context={"value": 0.12, "baseline": 0.45, "ratio": 0.27, "ip": "10.0.5.17"},
    )
    print(f"  [f_beacon_001] {text}")

    # Тест 2: Brute-force (новый IP без baseline)
    text = renderer.render_feature(
        feature_id="f_brute_001",
        shap_value=0.62,
        context={"value": 55, "baseline": 0, "ratio": None, "source_ip": "203.0.113.45"},
    )
    print(f"  [f_brute_001]  {text}")

    # Тест 3: DGA энтропия
    text = renderer.render_feature(
        feature_id="f_dga_002",
        shap_value=0.41,
        context={"value": 3.8, "baseline": 2.5, "ratio": 1.52, "source_ip": "192.168.1.100"},
    )
    print(f"  [f_dga_002]    {text}")

    print()
    print("Тест render_top_features:")
    top = renderer.render_top_features(
        feature_ids=["f_beacon_001", "f_brute_001", "f_dga_002"],
        shap_values=[0.35, 0.62, 0.41],
        feature_contexts=[
            {"value": 0.12, "baseline": 0.45, "ratio": 0.27, "ip": "10.0.5.17"},
            {"value": 55, "baseline": 0, "ratio": None, "source_ip": "203.0.113.45"},
            {"value": 3.8, "baseline": 2.5, "ratio": 1.52, "source_ip": "192.168.1.100"},
        ],
    )
    for item in top:
        print(f"  [{item['shap_abs']:.2f}] {item['explanation_short']}")

    print("=" * 70)