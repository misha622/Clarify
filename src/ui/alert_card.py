"""
Карточка алерта Clarify — CLI и программный интерфейс.

Выводит структурированную карточку алерта с блоками:
- Заголовок (тип, IP, время, уверенность)
- "Почему сработало" — SHAP-объяснения через NL-шаблоны
- "Внешний контекст" — threat intel (зарезервировано)
- Действия — confirm-flow

Может использоваться:
- В CLI для отладки
- Как источник данных для веб-интерфейса
- Как JSON для интеграций
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


@dataclass
class AlertAction:
    """Действие, которое можно выполнить над алертом."""
    action_id: str
    label: str  # "Заблокировать IP", "Игнорировать"
    description: str  # Пояснение
    requires_confirmation: bool = True
    confirmation_text: str = ""  # Текст в модальном окне
    webhook_payload: dict = field(default_factory=dict)


@dataclass
class AlertCard:
    """
    Карточка алерта. Содержит все блоки для отображения в UI.
    """
    alert_id: str
    alert_type: str  # "beaconing", "brute_force", "dga"
    severity: str  # "critical", "high", "medium", "low"
    source_ip: str
    target_ip: Optional[str]
    timestamp: float
    model_score: float
    model_threshold: float

    # Блок "Почему сработало" — топ-3 SHAP-объяснения
    explanations: list[dict] = field(default_factory=list)

    # Блок "Внешний контекст" — threat intel (пока пустой)
    threat_intel: list[dict] = field(default_factory=list)

    # Действия
    actions: list[AlertAction] = field(default_factory=list)

    # Метаданные
    latency_ms: float = 0.0
    detector_name: str = ""
    tenant_id: str = ""

    @property
    def time_iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()

    @property
    def time_local(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def confidence_pct(self) -> str:
        return f"{self.model_score:.0%}"

    @property
    def is_alert(self) -> bool:
        return self.model_score >= self.model_threshold

    def to_dict(self) -> dict:
        """Сериализует карточку в словарь (для JSON-ответа API)."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "source_ip": self.source_ip,
            "target_ip": self.target_ip,
            "timestamp": self.timestamp,
            "time_iso": self.time_iso,
            "time_local": self.time_local,
            "model_score": round(self.model_score, 4),
            "model_threshold": round(self.model_threshold, 4),
            "confidence": self.confidence_pct,
            "is_alert": self.is_alert,
            "explanations": self.explanations,
            "threat_intel": self.threat_intel,
            "actions": [
                {
                    "action_id": a.action_id,
                    "label": a.label,
                    "description": a.description,
                    "requires_confirmation": a.requires_confirmation,
                }
                for a in self.actions
            ],
            "latency_ms": round(self.latency_ms, 3),
            "detector_name": self.detector_name,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class AlertCardRenderer:
    """
    Рендерит AlertCard в человекочитаемый текст для CLI.

    Разделяет визуально блоки "Почему сработало" и "Внешний контекст".
    """

    # Цвета для CLI (ANSI escape codes)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Символы
    SEPARATOR = "─" * 60
    DOUBLE_SEP = "═" * 60
    THIN_SEP = "─" * 40

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def _c(self, text: str, color: str) -> str:
        """Применяет цвет, если включено."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text

    def _severity_icon(self, severity: str) -> str:
        icons = {
            "critical": "⛔",
            "high": "🔴",
            "medium": "🟡",
            "low": "🔵",
            "info": "⚪",
        }
        return icons.get(severity, "❓")

    def _severity_color(self, severity: str) -> str:
        colors = {
            "critical": self.RED,
            "high": self.RED,
            "medium": self.YELLOW,
            "low": self.BLUE,
            "info": self.DIM,
        }
        return colors.get(severity, self.WHITE)

    def render(self, card: AlertCard) -> str:
        """
        Рендерит карточку алерта в строку для CLI.

        Структура:
        ╔══════════════════════════════════════════╗
        ║  ЗАГОЛОВОК (тип, IP, время, confidence) ║
        ╠══════════════════════════════════════════╣
        ║  ПОЧЕМУ СРАБОТАЛО (SHAP-объяснения)     ║
        ╠══════════════════════════════════════════╣
        ║  ВНЕШНИЙ КОНТЕКСТ (threat intel)        ║
        ╠══════════════════════════════════════════╣
        ║  ДЕЙСТВИЯ (кнопки)                      ║
        ╚══════════════════════════════════════════╝
        """
        lines = []

        # ════════════════ ЗАГОЛОВОК ════════════════
        severity_color = self._severity_color(card.severity)
        icon = self._severity_icon(card.severity)

        lines.append(self._c(f"╔{self.DOUBLE_SEP}╗", self.CYAN))
        lines.append(self._c(f"║ {icon} {self.BOLD}{card.alert_type.upper()}{self.RESET}", severity_color))
        lines.append(self._c(f"║ Источник: {card.source_ip}" +
                             (f" → {card.target_ip}" if card.target_ip else ""), self.WHITE))
        lines.append(self._c(f"║ Время: {card.time_local}", self.WHITE))

        # Уверенность с цветом
        if card.model_score >= 0.9:
            conf_color = self.RED
        elif card.model_score >= 0.7:
            conf_color = self.YELLOW
        else:
            conf_color = self.GREEN
        lines.append(self._c(f"║ Уверенность: {card.confidence_pct}", conf_color))
        lines.append(self._c(f"║ Детектор: {card.detector_name}", self.DIM))
        lines.append(self._c(f"╠{self.DOUBLE_SEP}╣", self.CYAN))

        # ════════════════ ПОЧЕМУ СРАБОТАЛО ════════════════
        lines.append(self._c(f"║ {self.BOLD}ПОЧЕМУ СРАБОТАЛО (объяснение модели):{self.RESET}", self.GREEN))

        if card.explanations:
            for i, exp in enumerate(card.explanations, 1):
                direction = "▲" if exp.get("shap_value", 0) > 0 else "▼"
                shap_val = abs(exp.get("shap_value", 0))
                explanation = exp.get("explanation", exp.get("feature_name", "?"))

                # Перенос строки для длинных объяснений
                if len(explanation) > 55:
                    lines.append(self._c(f"║ {direction} [{shap_val:.2f}]", self.YELLOW))
                    # Разбиваем на строки по 55 символов
                    for j in range(0, len(explanation), 55):
                        chunk = explanation[j:j + 55]
                        lines.append(self._c(f"║    {chunk}", self.WHITE))
                else:
                    lines.append(self._c(
                        f"║ {direction} [{shap_val:.2f}] {explanation}", self.WHITE
                    ))
        else:
            lines.append(self._c(f"║   (объяснения недоступны)", self.DIM))

        lines.append(self._c(f"╠{self.DOUBLE_SEP}╣", self.CYAN))

        # ════════════════ ВНЕШНИЙ КОНТЕКСТ ════════════════
        lines.append(self._c(f"║ {self.BOLD}ВНЕШНИЙ КОНТЕКСТ (threat intel):{self.RESET}", self.BLUE))

        if card.threat_intel:
            for item in card.threat_intel:
                source = item.get("source", "?")
                text = item.get("text", "")
                lines.append(self._c(f"║ ⓘ [{source}] {text}", self.WHITE))
        else:
            lines.append(self._c(f"║   (источники threat intel не подключены)", self.DIM))

        lines.append(self._c(f"╠{self.DOUBLE_SEP}╣", self.CYAN))

        # ════════════════ ДЕЙСТВИЯ ════════════════
        lines.append(self._c(f"║ {self.BOLD}ДЕЙСТВИЯ:{self.RESET}", self.MAGENTA))

        if card.actions:
            for action in card.actions:
                shortcut = action.action_id.split("_")[0][:3].upper()
                lines.append(self._c(
                    f"║  [{shortcut}] {action.label}", self.WHITE
                ))
                if action.description:
                    lines.append(self._c(f"║       {action.description}", self.DIM))
        else:
            lines.append(self._c(f"║   (нет доступных действий)", self.DIM))

        lines.append(self._c(f"╚{self.DOUBLE_SEP}╝", self.CYAN))

        # Футер: latency
        if card.latency_ms > 0:
            lines.append(self._c(
                f"  ⏱ Анализ выполнен за {card.latency_ms:.1f} мс", self.DIM
            ))

        return "\n".join(lines)


class AlertCardBuilder:
    """
    Строит AlertCard из результатов детектора и SHAP-объяснителя.

    Связывает воедино:
    - BeaconingDetector (score, threshold)
    - ShapExplainer (top-3 SHAP признаков)
    - TemplateRenderer (NL-шаблоны)
    - Threat intel (пока заглушка)
    """

    def __init__(
            self,
            template_renderer=None,
            threat_intel_providers: list = None,
    ):
        """
        Args:
            template_renderer: TemplateRenderer для NL-шаблонов
            threat_intel_providers: список провайдеров threat intel (пока пусто)
        """
        self.template_renderer = template_renderer
        self.threat_intel_providers = threat_intel_providers or []

    def build(
            self,
            alert_type: str,
            source_ip: str,
            target_ip: Optional[str],
            model_score: float,
            model_threshold: float,
            shap_explanation,  # AlertExplanation из ShapExplainer
            detector_name: str = "beaconing",
            tenant_id: str = "default",
    ) -> AlertCard:
        """
        Строит AlertCard из всех компонентов.
        """
        # Определяем severity по скору
        if model_score >= 0.9:
            severity = "critical"
        elif model_score >= 0.7:
            severity = "high"
        elif model_score >= 0.5:
            severity = "medium"
        else:
            severity = "low"

        # ============================================================
        # МАППИНГ: техническое имя фичи → feature_id в словаре
        # ============================================================
        FEATURE_NAME_TO_ID = {
            # Beaconing
            "mean_interarrival_time": "f_beacon_004",
            "std_interarrival_time": "f_beacon_006",
            "coefficient_of_variation": "f_beacon_001",
            "peak_autocorrelation_lag": "f_beacon_002",
            "autocorrelation_peak_value": "f_beacon_002",
            "entropy_interarrival": "f_beacon_003",
            "event_count": "f_beacon_005",
            # Brute-Force
            "auth_failure_rate": "f_brute_001",
            "unique_usernames": "f_brute_002",
            "target_hosts": "f_brute_003",
            "new_user_agents": "f_brute_004",
            "total_attempts": "f_brute_001",
            # DGA
            "mean_entropy": "f_dga_002",
            "max_entropy": "f_dga_002",
            "mean_domain_length": "f_dga_005",
            "mean_vowel_consonant_ratio": "f_dga_002",
            "mean_ngram_score": "f_dga_002",
            "unique_domains": "f_dga_001",
            "nxdomain_rate": "f_dga_003",
            "unique_tld_count": "f_dga_004",
            "total_queries": "f_dga_001",
        }

        # Собираем объяснения
        explanations = []
        if shap_explanation and self.template_renderer:
            for feat in shap_explanation.top_features:
                dict_id = FEATURE_NAME_TO_ID.get(feat.feature_id, feat.feature_id)

                ratio = None
                if feat.baseline_value and feat.baseline_value > 0:
                    ratio = feat.feature_value / feat.baseline_value

                context = {
                    **feat.context,
                    "value": feat.feature_value,
                    "baseline": feat.baseline_value,
                    "ratio": ratio,
                    "source_ip": source_ip,
                    "ip": source_ip,
                }

                explanation_full = self.template_renderer.render_feature(
                    feature_id=dict_id,
                    shap_value=feat.shap_value,
                    context=context,
                    use_short=False,
                )
                explanation_short = self.template_renderer.render_feature(
                    feature_id=dict_id,
                    shap_value=feat.shap_value,
                    context=context,
                    use_short=True,
                )

                explanations.append({
                    "feature_id": dict_id,
                    "feature_name": feat.feature_name,
                    "shap_value": feat.shap_value,
                    "shap_abs": feat.shap_abs,
                    "feature_value": feat.feature_value,
                    "explanation": explanation_full,
                    "explanation_short": explanation_short,
                })
        elif shap_explanation:
            for feat in shap_explanation.top_features:
                explanations.append({
                    "feature_id": feat.feature_id,
                    "feature_name": feat.feature_name,
                    "shap_value": feat.shap_value,
                    "shap_abs": feat.shap_abs,
                    "feature_value": feat.feature_value,
                    "explanation": f"{feat.feature_name} = {feat.feature_value:.3f} (SHAP={feat.shap_value:+.3f})",
                    "explanation_short": f"{feat.feature_name}={feat.feature_value:.3f}",
                })

        # Собираем threat intel
        threat_intel = []
        for provider in self.threat_intel_providers:
            if hasattr(provider, 'lookup_ip'):
                result = provider.lookup_ip(source_ip)
                if result:
                    threat_intel.append(result)

        # Собираем действия
        is_attack = model_score >= model_threshold
        actions = []

        if is_attack:
            actions.append(AlertAction(
                action_id="block_ip",
                label=f"Заблокировать IP {source_ip}",
                description="Отправить webhook или скопировать команду для блокировки",
                requires_confirmation=True,
                confirmation_text=f"Блокировка IP {source_ip} на 24 часа. Причина: {alert_type}",
            ))

        actions.append(AlertAction(
            action_id="ignore_alert",
            label="Игнорировать",
            description="Отметить как ложное срабатывание",
            requires_confirmation=False,
        ))

        actions.append(AlertAction(
            action_id="show_details",
            label="Подробный SHAP-анализ",
            description="Открыть полный график SHAP force plot",
            requires_confirmation=False,
        ))

        alert_id = f"alert-{detector_name}-{source_ip.replace('.', '-')}-{int(time.time())}"

        return AlertCard(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            source_ip=source_ip,
            target_ip=target_ip,
            timestamp=time.time(),
            model_score=model_score,
            model_threshold=model_threshold,
            explanations=explanations,
            threat_intel=threat_intel,
            actions=actions,
            latency_ms=shap_explanation.latency_ms if shap_explanation else 0.0,
            detector_name=detector_name,
            tenant_id=tenant_id,
        )


# ------------------------------------------------------------------
# Демо: полный цикл от детекции до карточки
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    import numpy as np
    import xgboost as xgb

    from src.data.synthetic_generator import SyntheticGenerator
    from src.features.window_stats import calculate_window_stats
    from src.explainers.shap_explainer import ShapExplainer
    from src.rendering.template_renderer import TemplateRenderer
    # BeaconingDetector не используется напрямую в демо — модель загружается из файла

    print("=" * 70)
    print("  ДЕМО: полный цикл Clarify — от события до карточки алерта")
    print("=" * 70)
    print()

    # 1. Загружаем обученную модель
    model_path = "models/beaconing_xgb.json"
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"✅ Модель загружена: {model_path}")
    except Exception as e:
        print(f"❌ Модель не найдена: {e}")
        print("   Сначала запустите: python -m src.models.train_beaconing")
        sys.exit(1)

    # 2. Инициализируем компоненты
    feature_names = [
        "mean_interarrival_time", "std_interarrival_time",
        "coefficient_of_variation", "peak_autocorrelation_lag",
        "autocorrelation_peak_value", "entropy_interarrival",
        "event_count",
    ]

    explainer = ShapExplainer(model, feature_names, top_n=3)
    renderer = TemplateRenderer()
    builder = AlertCardBuilder(template_renderer=renderer)
    cli_renderer = AlertCardRenderer(use_colors=True)

    # 3. Генерируем тестовый сценарий
    gen = SyntheticGenerator(seed=42)
    X_test, y_test = gen.generate_for_beaconing_training(
        mode="train", window_size_seconds=900, stride_seconds=300,
        min_events_per_window=8, num_hosts=30,
    )

    # Берём одну атаку и один benign
    attack_idx = [i for i, label in enumerate(y_test) if label == 1][0]
    benign_idx = [i for i, label in enumerate(y_test) if label == 0][0]

    # 4. Загружаем конфиг для порога
    import yaml

    with open("config/detectors.yaml", "r") as f:
        config = yaml.safe_load(f)
    threshold = config["detectors"]["beaconing"]["decision_threshold"]

    # 5. Демо: атака
    print()
    print("─" * 70)
    print("  Сценарий 1: ОБНАРУЖЕНА АТАКА (Beaconing)")
    print("─" * 70)

    sample_attack = X_test[attack_idx:attack_idx + 1]
    dmatrix = xgb.DMatrix(sample_attack, feature_names=feature_names)
    score_attack = float(model.predict(dmatrix)[0])

    shap_result = explainer.explain(
        feature_vector=sample_attack,
        alert_type="beaconing",
        context={"source_ip": "45.33.32.156"},
    )

    card = builder.build(
        alert_type="beaconing",
        source_ip="45.33.32.156",
        target_ip="10.0.5.17",
        model_score=score_attack,
        model_threshold=threshold,
        shap_explanation=shap_result,
        detector_name="beaconing",
    )

    print(cli_renderer.render(card))

    # 6. Демо: benign
    print()
    print("─" * 70)
    print("  Сценарий 2: НОРМАЛЬНЫЙ ТРАФИК (без алерта)")
    print("─" * 70)

    sample_benign = X_test[benign_idx:benign_idx + 1]
    dmatrix2 = xgb.DMatrix(sample_benign, feature_names=feature_names)
    score_benign = float(model.predict(dmatrix2)[0])

    shap_result2 = explainer.explain(
        feature_vector=sample_benign,
        alert_type="benign",
        context={"source_ip": "10.0.15.42"},
    )

    card2 = builder.build(
        alert_type="benign",
        source_ip="10.0.15.42",
        target_ip=None,
        model_score=score_benign,
        model_threshold=threshold,
        shap_explanation=shap_result2,
        detector_name="beaconing",
    )

    print(cli_renderer.render(card2))

    # 7. JSON-вывод
    print()
    print("─" * 70)
    print("  JSON-представление карточки (для API):")
    print("─" * 70)
    print(card.to_json())

    print()
    print("=" * 70)
    print("  Демо завершено.")
    print("=" * 70)