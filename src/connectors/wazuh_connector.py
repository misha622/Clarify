"""
Wazuh Connector для Clarify.

Читает алерты Wazuh из:
1. Локального файла alerts.json (стандартный вывод Wazuh)
2. Wazuh API (опционально, если указан URL)

Извлекает события аутентификации и DNS-запросы,
передаёт их в детекторы Clarify (Beaconing, Brute-Force).

Использование:
    python -m src.connectors.wazuh_connector --alerts-file /var/ossec/logs/alerts/alerts.json
    python -m src.connectors.wazuh_connector --api-url https://wazuh.example.com --api-user foo --api-pass bar
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Generator
from datetime import datetime

sys.path.insert(0, ".")

logger = logging.getLogger(__name__)


class WazuhAlertParser:
    """
    Парсит сырые алерты Wazuh в структурированные события.

    Wazuh хранит алерты в /var/ossec/logs/alerts/alerts.json
    Формат: одна JSON-строка на алерт.
    """

    # Правила Wazuh, которые нас интересуют
    AUTH_FAILURE_RULES = {
        "5710",  # sshd: Attempt to login using a non-existent user
        "5712",  # sshd: brute force attempt
        "5716",  # sshd: authentication failed
        "5718",  # sshd: multiple authentication failures
        "5720",  # sshd: invalid user
        "6010",  # Windows: logon failure
        "6020",  # Windows: multiple logon failures
    }

    DNS_QUERY_RULES = {
        "60001",  # DNS query (пользовательское правило)
        "60002",  # DNS high entropy (пользовательское правило)
    }

    def __init__(self, min_timestamp: float = None):
        """
        Args:
            min_timestamp: игнорировать алерты старше этого времени
        """
        self.min_timestamp = min_timestamp or (time.time() - 86400)  # последние 24ч
        self.events: list[dict] = []

    def parse_alert(self, alert: dict) -> Optional[dict]:
        """
        Парсит один алерт Wazuh.

        Returns:
            dict с полями:
            - type: "auth_failure" | "dns_query"
            - timestamp: float
            - source_ip: str
            - metadata: dict (зависит от типа)

            или None, если алерт не интересен.
        """
        rule = alert.get("rule", {})
        rule_id = str(rule.get("id", ""))
        data = alert.get("data", {})

        timestamp_str = alert.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            timestamp = time.time()

        # Пропускаем старые алерты
        if timestamp < self.min_timestamp:
            return None

        # ── Brute-Force / Auth Failure ──
        if rule_id in self.AUTH_FAILURE_RULES:
            source_ip = (
                    data.get("srcip")
                    or data.get("src_ip")
                    or data.get("source", {}).get("ip")
                    or "0.0.0.0"
            )

            return {
                "type": "auth_failure",
                "timestamp": timestamp,
                "source_ip": source_ip,
                "metadata": {
                    "rule_id": rule_id,
                    "rule_description": rule.get("description", ""),
                    "username": data.get("dstuser") or data.get("user") or "unknown",
                    "protocol": data.get("protocol") or "ssh",
                    "target_ip": data.get("dstip") or data.get("dst_ip") or "",
                    "agent": alert.get("agent", {}).get("name", "unknown"),
                },
            }

        # ── DNS Query ──
        if rule_id in self.DNS_QUERY_RULES:
            source_ip = (
                    data.get("srcip")
                    or data.get("src_ip")
                    or "0.0.0.0"
            )
            domain = data.get("domain") or data.get("query") or ""

            return {
                "type": "dns_query",
                "timestamp": timestamp,
                "source_ip": source_ip,
                "metadata": {
                    "rule_id": rule_id,
                    "domain": domain,
                    "nxdomain": data.get("nxdomain", False),
                    "agent": alert.get("agent", {}).get("name", "unknown"),
                },
            }

        return None

    def parse_file(self, filepath: str) -> list[dict]:
        """
        Читает alerts.json и парсит все алерты.

        Args:
            filepath: путь к alerts.json

        Returns:
            список структурированных событий
        """
        events = []
        path = Path(filepath)

        if not path.exists():
            logger.error(f"Файл не найден: {filepath}")
            return events

        logger.info(f"Чтение {filepath}...")

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    alert = json.loads(line)
                    parsed = self.parse_alert(alert)
                    if parsed:
                        events.append(parsed)
                except json.JSONDecodeError:
                    logger.debug(f"Пропущена строка {line_num}: не JSON")
                    continue

        logger.info(f"Извлечено {len(events)} событий из {filepath}")
        return events

    def group_by_source(self, events: list[dict]) -> dict[str, dict]:
        """
        Группирует события по source_ip и типу.

        Returns:
            {
                "203.0.113.45": {
                    "auth_failures": [...],
                    "dns_queries": [...],
                },
                ...
            }
        """
        groups: dict[str, dict] = {}

        for event in events:
            ip = event["source_ip"]
            if ip not in groups:
                groups[ip] = {"auth_failures": [], "dns_queries": []}

            if event["type"] == "auth_failure":
                groups[ip]["auth_failures"].append(event)
            elif event["type"] == "dns_query":
                groups[ip]["dns_queries"].append(event)

        return groups


class WazuhAPIClient:
    """Клиент для Wazuh REST API (опционально)."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token: Optional[str] = None

    def authenticate(self) -> bool:
        """Получает JWT токен."""
        import urllib.request
        import urllib.error

        try:
            url = f"{self.base_url}/security/user/authenticate"
            payload = json.dumps({
                "username": self.username,
                "password": self.password,
            }).encode("utf-8")

            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                self.token = data.get("data", {}).get("token", "")
                if self.token:
                    logger.info("Wazuh API: аутентификация успешна")
                    return True

        except Exception as e:
            logger.error(f"Wazuh API: ошибка аутентификации: {e}")

        return False

    def get_alerts(self, limit: int = 500) -> list[dict]:
        """Получает последние алерты через API."""
        import urllib.request

        if not self.token:
            if not self.authenticate():
                return []

        try:
            url = f"{self.base_url}/alerts?limit={limit}&sort=-timestamp"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self.token}"},
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                alerts = data.get("data", {}).get("alerts", [])
                logger.info(f"Wazuh API: получено {len(alerts)} алертов")
                return alerts

        except Exception as e:
            logger.error(f"Wazuh API: ошибка получения алертов: {e}")
            return []


class ClarifyWazuhRunner:
    """
    Связывает Wazuh → парсер → детекторы Clarify → карточки алертов.

    Полный пайплайн:
    1. Читает алерты Wazuh (файл или API)
    2. Парсит в структурированные события
    3. Группирует по source_ip
    4. Прогоняет через детекторы (Beaconing, Brute-Force)
    5. Генерирует SHAP-объяснения
    6. Выводит карточки алертов
    """

    def __init__(
            self,
            alerts_file: str = None,
            api_url: str = None,
            api_user: str = None,
            api_pass: str = None,
            lang: str = "ru",
    ):
        self.alerts_file = alerts_file
        self.api_url = api_url
        self.api_user = api_user
        self.api_pass = api_pass
        self.lang = lang

        # Инициализируем компоненты лениво
        self._beaconing_detector = None
        self._brute_force_detector = None
        self._shap_explainer_beaconing = None
        self._shap_explainer_brute_force = None
        self._template_renderer = None
        self._alert_builder = None
        self._cli_renderer = None

    def _init_components(self):
        """Ленивая инициализация компонентов Clarify."""
        import xgboost as xgb
        import yaml
        from src.explainers.shap_explainer import ShapExplainer
        from src.rendering.template_renderer import TemplateRenderer
        from src.ui.alert_card import AlertCardBuilder, AlertCardRenderer

        # Конфиг
        with open("config/detectors.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Beaconing
        bf_cfg = config["detectors"]["beaconing"]
        if Path(bf_cfg["model_path"]).exists():
            self._beaconing_model = xgb.Booster()
            self._beaconing_model.load_model(bf_cfg["model_path"])
            self._beaconing_threshold = bf_cfg["decision_threshold"]
            self._shap_explainer_beaconing = ShapExplainer(
                self._beaconing_model,
                bf_cfg["features"],
                top_n=3,
            )
            logger.info("Beaconing модель загружена")
        else:
            logger.warning("Beaconing модель не найдена, пропускаем")
            self._beaconing_model = None

        # Brute-Force
        bf_cfg = config["detectors"].get("brute_force", {})
        if bf_cfg.get("model_path") and Path(bf_cfg["model_path"]).exists():
            self._brute_force_model = xgb.Booster()
            self._brute_force_model.load_model(bf_cfg["model_path"])
            self._brute_force_threshold = bf_cfg["decision_threshold"]
            self._shap_explainer_brute_force = ShapExplainer(
                self._brute_force_model,
                bf_cfg["features"],
                top_n=3,
            )
            logger.info("Brute-Force модель загружена")
        else:
            logger.warning("Brute-Force модель не найдена, пропускаем")
            self._brute_force_model = None

        # Рендерер
        dict_path = f"config/feature_dictionary{'_en' if self.lang == 'en' else ''}.yaml"
        if not Path(dict_path).exists():
            dict_path = "config/feature_dictionary.yaml"
        self._template_renderer = TemplateRenderer(dictionary_path=dict_path)
        self._alert_builder = AlertCardBuilder(template_renderer=self._template_renderer)
        self._cli_renderer = AlertCardRenderer(use_colors=True)

    def run(self):
        """Основной цикл: читает алерты, детектит, объясняет."""
        self._init_components()

        # Шаг 1: Получаем алерты
        parser = WazuhAlertParser()

        if self.alerts_file:
            events = parser.parse_file(self.alerts_file)
        elif self.api_url:
            api = WazuhAPIClient(self.api_url, self.api_user, self.api_pass)
            raw_alerts = api.get_alerts()
            events = []
            for alert in raw_alerts:
                parsed = parser.parse_alert(alert)
                if parsed:
                    events.append(parsed)
        else:
            logger.error("Укажите --alerts-file или --api-url")
            return

        if not events:
            logger.warning("Нет событий для анализа")
            return

        # Шаг 2: Группируем по IP
        groups = parser.group_by_source(events)
        logger.info(f"Анализ {len(groups)} уникальных IP...")

        # Шаг 3: Прогоняем через детекторы
        alerts_found = 0

        for ip, data in groups.items():
            # ── Brute-Force ──
            auth_events = data["auth_failures"]
            if len(auth_events) >= 10 and self._brute_force_model:
                timestamps = [e["timestamp"] for e in auth_events]
                usernames = [e["metadata"]["username"] for e in auth_events]
                targets = [e["metadata"].get("target_ip", "") for e in auth_events]
                user_agents = [""] * len(auth_events)

                from src.detectors.brute_force import BruteForceDetector
                detector = BruteForceDetector()
                detector.model = self._brute_force_model
                detector.decision_threshold = self._brute_force_threshold

                result = detector.detect(
                    timestamps, usernames, targets, user_agents,
                    source_ip=ip,
                )

                if result["is_alert"] and result["feature_vector"] is not None:
                    # SHAP
                    shap_result = self._shap_explainer_brute_force.explain(
                        feature_vector=result["feature_vector"],
                        alert_type="brute_force",
                        context={"source_ip": ip},
                    )

                    card = self._alert_builder.build(
                        alert_type="brute_force",
                        source_ip=ip,
                        target_ip=targets[0] if targets else None,
                        model_score=result["score"],
                        model_threshold=self._brute_force_threshold,
                        shap_explanation=shap_result,
                        detector_name="brute_force",
                    )

                    print(self._cli_renderer.render(card))
                    alerts_found += 1

            # ── Beaconing ──
            dns_events = data["dns_queries"]
            if len(dns_events) >= 15 and self._beaconing_model:
                timestamps = sorted([e["timestamp"] for e in dns_events])

                from src.features.window_stats import calculate_window_stats
                from src.utils.thresholds import check_min_intervals

                intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
                gate = check_min_intervals(intervals, min_required=15)

                if gate.should_proceed:
                    stats = calculate_window_stats(timestamps)
                    feature_vector = np.array(stats.to_feature_vector()).reshape(1, -1)

                    import numpy as np
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(
                        feature_vector,
                        feature_names=self._shap_explainer_beaconing.feature_names,
                    )
                    score = float(self._beaconing_model.predict(dmatrix)[0])

                    if score >= self._beaconing_threshold:
                        shap_result = self._shap_explainer_beaconing.explain(
                            feature_vector=feature_vector,
                            alert_type="beaconing",
                            context={"source_ip": ip},
                        )

                        card = self._alert_builder.build(
                            alert_type="beaconing",
                            source_ip=ip,
                            target_ip=None,
                            model_score=score,
                            model_threshold=self._beaconing_threshold,
                            shap_explanation=shap_result,
                            detector_name="beaconing",
                        )

                        print(self._cli_renderer.render(card))
                        alerts_found += 1

        logger.info(f"Анализ завершён. Найдено алертов: {alerts_found}")


def main():
    parser = argparse.ArgumentParser(
        description="Clarify Wazuh Connector — анализ алертов Wazuh"
    )
    parser.add_argument(
        "--alerts-file",
        help="Путь к alerts.json Wazuh (обычно /var/ossec/logs/alerts/alerts.json)",
    )
    parser.add_argument(
        "--api-url",
        help="URL Wazuh API (например https://wazuh.example.com)",
    )
    parser.add_argument("--api-user", help="Пользователь Wazuh API")
    parser.add_argument("--api-pass", help="Пароль Wazuh API")
    parser.add_argument("--lang", default="ru", choices=["ru", "en"],
                        help="Язык объяснений")
    parser.add_argument("--log-level", default="INFO",
                        help="Уровень логирования")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )

    runner = ClarifyWazuhRunner(
        alerts_file=args.alerts_file,
        api_url=args.api_url,
        api_user=args.api_user,
        api_pass=args.api_pass,
        lang=args.lang,
    )

    runner.run()


if __name__ == "__main__":
    main()