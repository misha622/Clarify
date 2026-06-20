"""
Wazuh Connector РґР»СЏ Clarify.

Р§РёС‚Р°РµС‚ Р°Р»РµСЂС‚С‹ Wazuh РёР·:
1. Р›РѕРєР°Р»СЊРЅРѕРіРѕ С„Р°Р№Р»Р° alerts.json (СЃС‚Р°РЅРґР°СЂС‚РЅС‹Р№ РІС‹РІРѕРґ Wazuh)
2. Wazuh API (РѕРїС†РёРѕРЅР°Р»СЊРЅРѕ, РµСЃР»Рё СѓРєР°Р·Р°РЅ URL)

РР·РІР»РµРєР°РµС‚ СЃРѕР±С‹С‚РёСЏ Р°СѓС‚РµРЅС‚РёС„РёРєР°С†РёРё Рё DNS-Р·Р°РїСЂРѕСЃС‹,
РїРµСЂРµРґР°С‘С‚ РёС… РІ РґРµС‚РµРєС‚РѕСЂС‹ Clarify (Beaconing, Brute-Force).

РСЃРїРѕР»СЊР·РѕРІР°РЅРёРµ:
    python -m src.connectors.wazuh_connector --alerts-file /var/ossec/logs/alerts/alerts.json
    python -m src.connectors.wazuh_connector --api-url https://wazuh.example.com --api-user foo --api-pass bar
"""

import sys
import os
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
    РџР°СЂСЃРёС‚ СЃС‹СЂС‹Рµ Р°Р»РµСЂС‚С‹ Wazuh РІ СЃС‚СЂСѓРєС‚СѓСЂРёСЂРѕРІР°РЅРЅС‹Рµ СЃРѕР±С‹С‚РёСЏ.

    Wazuh С…СЂР°РЅРёС‚ Р°Р»РµСЂС‚С‹ РІ /var/ossec/logs/alerts/alerts.json
    Р¤РѕСЂРјР°С‚: РѕРґРЅР° JSON-СЃС‚СЂРѕРєР° РЅР° Р°Р»РµСЂС‚.
    """

    # РџСЂР°РІРёР»Р° Wazuh, РєРѕС‚РѕСЂС‹Рµ РЅР°СЃ РёРЅС‚РµСЂРµСЃСѓСЋС‚
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
        "60001",  # DNS query (РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРѕРµ РїСЂР°РІРёР»Рѕ)
        "60002",  # DNS high entropy (РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРѕРµ РїСЂР°РІРёР»Рѕ)
    }

    def __init__(self, min_timestamp: float = None):
        """
        Args:
            min_timestamp: РёРіРЅРѕСЂРёСЂРѕРІР°С‚СЊ Р°Р»РµСЂС‚С‹ СЃС‚Р°СЂС€Рµ СЌС‚РѕРіРѕ РІСЂРµРјРµРЅРё
        """
        self.min_timestamp = min_timestamp or (time.time() - 86400)  # РїРѕСЃР»РµРґРЅРёРµ 24С‡
        self.events: list[dict] = []

    def parse_alert(self, alert: dict) -> Optional[dict]:
        """
        РџР°СЂСЃРёС‚ РѕРґРёРЅ Р°Р»РµСЂС‚ Wazuh.

        Returns:
            dict СЃ РїРѕР»СЏРјРё:
            - type: "auth_failure" | "dns_query"
            - timestamp: float
            - source_ip: str
            - metadata: dict (Р·Р°РІРёСЃРёС‚ РѕС‚ С‚РёРїР°)

            РёР»Рё None, РµСЃР»Рё Р°Р»РµСЂС‚ РЅРµ РёРЅС‚РµСЂРµСЃРµРЅ.
        """
        rule = alert.get("rule", {})
        rule_id = str(rule.get("id", ""))
        data = alert.get("data", {})

        timestamp_str = alert.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            timestamp = time.time()

        # РџСЂРѕРїСѓСЃРєР°РµРј СЃС‚Р°СЂС‹Рµ Р°Р»РµСЂС‚С‹
        if timestamp < self.min_timestamp:
            return None

        # в”Ђв”Ђ Brute-Force / Auth Failure в”Ђв”Ђ
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

        # в”Ђв”Ђ DNS Query в”Ђв”Ђ
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
        Р§РёС‚Р°РµС‚ alerts.json Рё РїР°СЂСЃРёС‚ РІСЃРµ Р°Р»РµСЂС‚С‹.

        Args:
            filepath: РїСѓС‚СЊ Рє alerts.json

        Returns:
            СЃРїРёСЃРѕРє СЃС‚СЂСѓРєС‚СѓСЂРёСЂРѕРІР°РЅРЅС‹С… СЃРѕР±С‹С‚РёР№
        """
        events = []
        path = Path(filepath)

        if not path.exists():
            logger.error(f"Р¤Р°Р№Р» РЅРµ РЅР°Р№РґРµРЅ: {filepath}")
            return events

        logger.info(f"Р§С‚РµРЅРёРµ {filepath}...")

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
                    logger.debug(f"РџСЂРѕРїСѓС‰РµРЅР° СЃС‚СЂРѕРєР° {line_num}: РЅРµ JSON")
                    continue

        logger.info(f"РР·РІР»РµС‡РµРЅРѕ {len(events)} СЃРѕР±С‹С‚РёР№ РёР· {filepath}")
        return events

    def group_by_source(self, events: list[dict]) -> dict[str, dict]:
        """
        Р“СЂСѓРїРїРёСЂСѓРµС‚ СЃРѕР±С‹С‚РёСЏ РїРѕ source_ip Рё С‚РёРїСѓ.

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
    """РљР»РёРµРЅС‚ РґР»СЏ Wazuh REST API (РѕРїС†РёРѕРЅР°Р»СЊРЅРѕ)."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token: Optional[str] = None

    def authenticate(self) -> bool:
        """РџРѕР»СѓС‡Р°РµС‚ JWT С‚РѕРєРµРЅ."""
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
                    logger.info("Wazuh API: Р°СѓС‚РµРЅС‚РёС„РёРєР°С†РёСЏ СѓСЃРїРµС€РЅР°")
                    return True

        except Exception as e:
            logger.error(f"Wazuh API: РѕС€РёР±РєР° Р°СѓС‚РµРЅС‚РёС„РёРєР°С†РёРё: {e}")

        return False

    def get_alerts(self, limit: int = 500) -> list[dict]:
        """РџРѕР»СѓС‡Р°РµС‚ РїРѕСЃР»РµРґРЅРёРµ Р°Р»РµСЂС‚С‹ С‡РµСЂРµР· API."""
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
                logger.info(f"Wazuh API: РїРѕР»СѓС‡РµРЅРѕ {len(alerts)} Р°Р»РµСЂС‚РѕРІ")
                return alerts

        except Exception as e:
            logger.error(f"Wazuh API: РѕС€РёР±РєР° РїРѕР»СѓС‡РµРЅРёСЏ Р°Р»РµСЂС‚РѕРІ: {e}")
            return []


class ClarifyWazuhRunner:
    """
    РЎРІСЏР·С‹РІР°РµС‚ Wazuh в†’ РїР°СЂСЃРµСЂ в†’ РґРµС‚РµРєС‚РѕСЂС‹ Clarify в†’ РєР°СЂС‚РѕС‡РєРё Р°Р»РµСЂС‚РѕРІ.

    РџРѕР»РЅС‹Р№ РїР°Р№РїР»Р°Р№РЅ:
    1. Р§РёС‚Р°РµС‚ Р°Р»РµСЂС‚С‹ Wazuh (С„Р°Р№Р» РёР»Рё API)
    2. РџР°СЂСЃРёС‚ РІ СЃС‚СЂСѓРєС‚СѓСЂРёСЂРѕРІР°РЅРЅС‹Рµ СЃРѕР±С‹С‚РёСЏ
    3. Р“СЂСѓРїРїРёСЂСѓРµС‚ РїРѕ source_ip
    4. РџСЂРѕРіРѕРЅСЏРµС‚ С‡РµСЂРµР· РґРµС‚РµРєС‚РѕСЂС‹ (Beaconing, Brute-Force)
    5. Р“РµРЅРµСЂРёСЂСѓРµС‚ SHAP-РѕР±СЉСЏСЃРЅРµРЅРёСЏ
    6. Р’С‹РІРѕРґРёС‚ РєР°СЂС‚РѕС‡РєРё Р°Р»РµСЂС‚РѕРІ
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

        # РРЅРёС†РёР°Р»РёР·РёСЂСѓРµРј РєРѕРјРїРѕРЅРµРЅС‚С‹ Р»РµРЅРёРІРѕ
        self._beaconing_detector = None
        self._brute_force_detector = None
        self._shap_explainer_beaconing = None
        self._shap_explainer_brute_force = None
        self._template_renderer = None
        self._alert_builder = None
        self._cli_renderer = None

    def _init_components(self):
        """Р›РµРЅРёРІР°СЏ РёРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РєРѕРјРїРѕРЅРµРЅС‚РѕРІ Clarify."""
        import xgboost as xgb
        import yaml
        from src.explainers.shap_explainer import ShapExplainer
        from src.rendering.template_renderer import TemplateRenderer
        from src.ui.alert_card import AlertCardBuilder, AlertCardRenderer

        # РљРѕРЅС„РёРі
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
            logger.info("Beaconing РјРѕРґРµР»СЊ Р·Р°РіСЂСѓР¶РµРЅР°")
        else:
            logger.warning("Beaconing РјРѕРґРµР»СЊ РЅРµ РЅР°Р№РґРµРЅР°, РїСЂРѕРїСѓСЃРєР°РµРј")
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
            logger.info("Brute-Force РјРѕРґРµР»СЊ Р·Р°РіСЂСѓР¶РµРЅР°")
        else:
            logger.warning("Brute-Force РјРѕРґРµР»СЊ РЅРµ РЅР°Р№РґРµРЅР°, РїСЂРѕРїСѓСЃРєР°РµРј")
            self._brute_force_model = None

        # Р РµРЅРґРµСЂРµСЂ
        dict_path = f"config/feature_dictionary{'_en' if self.lang == 'en' else ''}.yaml"
        if not Path(dict_path).exists():
            dict_path = "config/feature_dictionary.yaml"
        self._template_renderer = TemplateRenderer(dictionary_path=dict_path)
        self._alert_builder = AlertCardBuilder(template_renderer=self._template_renderer)
        self._cli_renderer = AlertCardRenderer(use_colors=True)

    def run(self):
        """РћСЃРЅРѕРІРЅРѕР№ С†РёРєР»: С‡РёС‚Р°РµС‚ Р°Р»РµСЂС‚С‹, РґРµС‚РµРєС‚РёС‚, РѕР±СЉСЏСЃРЅСЏРµС‚."""
        self._init_components()

        # РЁР°Рі 1: РџРѕР»СѓС‡Р°РµРј Р°Р»РµСЂС‚С‹
        parser = WazuhAlertParser()

        if self.alerts_file:
            events = parser.parse_file(self.alerts_file)
        elif self.api_url:
            api = WazuhAPIClient(self.api_url, self.api_user, self.api_pass or os.environ.get("WAZUH_API_PASSWORD", ""))
            raw_alerts = api.get_alerts()
            events = []
            for alert in raw_alerts:
                parsed = parser.parse_alert(alert)
                if parsed:
                    events.append(parsed)
        else:
            logger.error("РЈРєР°Р¶РёС‚Рµ --alerts-file РёР»Рё --api-url")
            return

        if not events:
            logger.warning("РќРµС‚ СЃРѕР±С‹С‚РёР№ РґР»СЏ Р°РЅР°Р»РёР·Р°")
            return

        # РЁР°Рі 2: Р“СЂСѓРїРїРёСЂСѓРµРј РїРѕ IP
        groups = parser.group_by_source(events)
        logger.info(f"РђРЅР°Р»РёР· {len(groups)} СѓРЅРёРєР°Р»СЊРЅС‹С… IP...")

        # РЁР°Рі 3: РџСЂРѕРіРѕРЅСЏРµРј С‡РµСЂРµР· РґРµС‚РµРєС‚РѕСЂС‹
        alerts_found = 0

        for ip, data in groups.items():
            # в”Ђв”Ђ Brute-Force в”Ђв”Ђ
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

            # в”Ђв”Ђ Beaconing в”Ђв”Ђ
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

        logger.info(f"РђРЅР°Р»РёР· Р·Р°РІРµСЂС€С‘РЅ. РќР°Р№РґРµРЅРѕ Р°Р»РµСЂС‚РѕРІ: {alerts_found}")


def main():
    parser = argparse.ArgumentParser(
        description="Clarify Wazuh Connector вЂ” Р°РЅР°Р»РёР· Р°Р»РµСЂС‚РѕРІ Wazuh"
    )
    parser.add_argument(
        "--alerts-file",
        help="РџСѓС‚СЊ Рє alerts.json Wazuh (РѕР±С‹С‡РЅРѕ /var/ossec/logs/alerts/alerts.json)",
    )
    parser.add_argument(
        "--api-url",
        help="URL Wazuh API (РЅР°РїСЂРёРјРµСЂ https://wazuh.example.com)",
    )
    parser.add_argument("--api-user", help="РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ Wazuh API")
    parser.add_argument("--api-pass", help="РџР°СЂРѕР»СЊ Wazuh API")
    parser.add_argument("--lang", default="ru", choices=["ru", "en"],
                        help="РЇР·С‹Рє РѕР±СЉСЏСЃРЅРµРЅРёР№")
    parser.add_argument("--log-level", default="INFO",
                        help="РЈСЂРѕРІРµРЅСЊ Р»РѕРіРёСЂРѕРІР°РЅРёСЏ")

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
