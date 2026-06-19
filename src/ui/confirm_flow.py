"""
Confirm-flow РґР»СЏ РґРµР№СЃС‚РІРёР№ РЅР°Рґ Р°Р»РµСЂС‚Р°РјРё Clarify.

Р РµР°Р»РёР·СѓРµС‚ Human-in-the-loop:
- РљРЅРѕРїРєР° "Р—Р°Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ" РќР• РёСЃРїРѕР»РЅСЏРµС‚ РґРµР№СЃС‚РІРёРµ РЅР°РїСЂСЏРјСѓСЋ
- РћС‚РєСЂС‹РІР°РµС‚ confirm-flow: РјРѕРґР°Р»СЊРЅРѕРµ РѕРєРЅРѕ в†’ webhook РёР»Рё РєРѕРїРёСЂРѕРІР°РЅРёРµ РєРѕРјР°РЅРґС‹
- Webhook РїСЂРѕРІРµСЂСЏРµС‚СЃСЏ РїСЂРё СЃРѕС…СЂР°РЅРµРЅРёРё РєРѕРЅС„РёРіР°
- Р•СЃР»Рё webhook РЅРµ РЅР°СЃС‚СЂРѕРµРЅ вЂ” Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРё РїРѕРєР°Р·С‹РІР°РµС‚ РєРѕРјР°РЅРґСѓ РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ
- Р§РµРєР±РѕРєСЃ "РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ webhook" РґРѕСЃС‚СѓРїРµРЅ С‚РѕР»СЊРєРѕ РїСЂРё РІР°Р»РёРґРЅРѕРј webhook URL
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ActionMethod(Enum):
    """РЎРїРѕСЃРѕР± РёСЃРїРѕР»РЅРµРЅРёСЏ РґРµР№СЃС‚РІРёСЏ."""
    WEBHOOK = "webhook"
    COPY_COMMAND = "copy_command"
    NONE = "none"


@dataclass
class FirewallCommand:
    """Команда для блокировки IP на разных фаерволах."""
    
    @staticmethod
    def _escape_comment(text: str) -> str:
        """Экранирует кавычки в комментарии для shell."""
        return text.replace('"', '\\"').replace("'", "'\\''")
    
    """РљРѕРјР°РЅРґР° РґР»СЏ Р±Р»РѕРєРёСЂРѕРІРєРё IP РЅР° СЂР°Р·РЅС‹С… С„Р°РµСЂРІРѕР»Р°С…."""
    ip: str
    reason: str
    duration_hours: int = 24

    @property
    def iptables(self) -> str:
        return (
            f"iptables -A INPUT -s {self.ip} -j DROP "
            f"-m comment --comment \"Clarify: {self.reason} (alert)\""
        )

    @property
    def firewall_cmd(self) -> str:
        return (
            f"firewall-cmd --add-rich-rule='rule family=\"ipv4\" "
            f"source address=\"{self.ip}\" drop'"
        )

    @property
    def ufw(self) -> str:
        return f"ufw deny from {self.ip} comment 'Clarify: {self.reason}'"

    @property
    def windows_firewall(self) -> str:
        return (
            f"New-NetFirewallRule -DisplayName \"Clarify Block {self.ip}\" "
            f"-Direction Inbound -RemoteAddress {self.ip} -Action Block"
        )

    def get_command(self, firewall_type: str = "iptables") -> str:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ РєРѕРјР°РЅРґСѓ РґР»СЏ СѓРєР°Р·Р°РЅРЅРѕРіРѕ С‚РёРїР° С„Р°РµСЂРІРѕР»Р°."""
        commands = {
            "iptables": self.iptables,
            "firewalld": self.firewall_cmd,
            "ufw": self.ufw,
            "windows": self.windows_firewall,
        }
        return commands.get(firewall_type, self.iptables)


@dataclass
class WebhookConfig:
    """РљРѕРЅС„РёРіСѓСЂР°С†РёСЏ webhook РґР»СЏ Р°РІС‚Рѕ-Р±Р»РѕРєРёСЂРѕРІРєРё."""
    url: str
    method: str = "POST"
    headers: dict = None
    timeout_seconds: float = 5.0
    validated: bool = False

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}


@dataclass
class ConfirmFlowResult:
    """Р РµР·СѓР»СЊС‚Р°С‚ confirm-flow."""
    action_id: str
    method: ActionMethod
    success: bool
    message: str
    command: Optional[str] = None
    webhook_response: Optional[dict] = None


class ConfirmFlow:
    """
    РћР±СЂР°Р±РѕС‚С‡РёРє confirm-flow РґР»СЏ РґРµР№СЃС‚РІРёР№.

    РџСЂР°РІРёР»Р°:
    1. Р•СЃР»Рё webhook РЅР°СЃС‚СЂРѕРµРЅ Рё РІР°Р»РёРґРµРЅ в†’ РѕС‚РїСЂР°РІР»СЏРµС‚ POST
    2. Р•СЃР»Рё webhook РќР• РЅР°СЃС‚СЂРѕРµРЅ в†’ РїРѕРєР°Р·С‹РІР°РµС‚ РєРѕРјР°РЅРґСѓ РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ
    3. Р§РµРєР±РѕРєСЃ "РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ webhook" РґРѕСЃС‚СѓРїРµРЅ С‚РѕР»СЊРєРѕ РїСЂРё РІР°Р»РёРґРЅРѕРј webhook
    4. Р’СЃРµ СЂРµС€РµРЅРёСЏ Р»РѕРіРёСЂСѓСЋС‚СЃСЏ
    """

    def __init__(
            self,
            webhook_config: Optional[WebhookConfig] = None,
            default_firewall: str = "iptables",
    ):
        """
        Args:
            webhook_config: РєРѕРЅС„РёРіСѓСЂР°С†РёСЏ webhook (None = РЅРµ РЅР°СЃС‚СЂРѕРµРЅ)
            default_firewall: С‚РёРї С„Р°РµСЂРІРѕР»Р° РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ РґР»СЏ РєРѕРјР°РЅРґ
        """
        self.webhook_config = webhook_config
        self.default_firewall = default_firewall
        self.decisions_log: list[dict] = []

    def validate_webhook(self) -> tuple[bool, str]:
        """
        РџСЂРѕРІРµСЂСЏРµС‚ РґРѕСЃС‚СѓРїРЅРѕСЃС‚СЊ webhook URL.
        РћС‚РїСЂР°РІР»СЏРµС‚ С‚РµСЃС‚РѕРІС‹Р№ POST Рё Р¶РґС‘С‚ 2xx.

        Returns:
            (СѓСЃРїРµС…, СЃРѕРѕР±С‰РµРЅРёРµ)
        """
        if not self.webhook_config or not self.webhook_config.url:
            return False, "Webhook URL РЅРµ РЅР°СЃС‚СЂРѕРµРЅ"

        try:
            import urllib.request
            import urllib.error

            test_payload = json.dumps({
                "type": "test",
                "message": "Clarify webhook validation",
                "timestamp": __import__("time").time(),
            }).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_config.url,
                data=test_payload,
                headers=self.webhook_config.headers or {},
                method=self.webhook_config.method,
            )

            with urllib.request.urlopen(req, timeout=self.webhook_config.timeout_seconds) as resp:
                if 200 <= resp.status < 300:
                    self.webhook_config.validated = True
                    return True, f"Webhook РґРѕСЃС‚СѓРїРµРЅ (status={resp.status})"
                else:
                    return False, f"Webhook РІРµСЂРЅСѓР» СЃС‚Р°С‚СѓСЃ {resp.status}"

        except urllib.error.URLError as e:
            return False, f"Webhook РЅРµРґРѕСЃС‚СѓРїРµРЅ: {e.reason}"
        except Exception as e:
            return False, f"РћС€РёР±РєР° РїСЂРѕРІРµСЂРєРё webhook: {e}"

    def execute_block(
            self,
            ip: str,
            reason: str,
            alert_id: str,
            duration_hours: int = 24,
            operator: str = "unknown",
    ) -> ConfirmFlowResult:
        """
        Р’С‹РїРѕР»РЅСЏРµС‚ Р±Р»РѕРєРёСЂРѕРІРєСѓ IP С‡РµСЂРµР· confirm-flow.

        Args:
            ip: IP РґР»СЏ Р±Р»РѕРєРёСЂРѕРІРєРё
            reason: РїСЂРёС‡РёРЅР° (РёР· SHAP-РѕР±СЉСЏСЃРЅРµРЅРёСЏ)
            alert_id: ID Р°Р»РµСЂС‚Р°
            duration_hours: РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ Р±Р»РѕРєРёСЂРѕРІРєРё
            operator: РєС‚Рѕ РІС‹РїРѕР»РЅРёР» РґРµР№СЃС‚РІРёРµ

        Returns:
            ConfirmFlowResult
        """
        # Р›РѕРіРёСЂСѓРµРј СЂРµС€РµРЅРёРµ
        decision = {
            "action": "block_ip",
            "ip": ip,
            "reason": reason,
            "alert_id": alert_id,
            "duration_hours": duration_hours,
            "operator": operator,
            "timestamp": __import__("time").time(),
        }

        # Р•СЃР»Рё webhook РЅР°СЃС‚СЂРѕРµРЅ Рё РІР°Р»РёРґРµРЅ вЂ” РѕС‚РїСЂР°РІР»СЏРµРј
        if (
                self.webhook_config
                and self.webhook_config.url
                and self.webhook_config.validated
        ):
            return self._send_webhook(ip, reason, alert_id, duration_hours, decision)

        # РРЅР°С‡Рµ вЂ” РІРѕР·РІСЂР°С‰Р°РµРј РєРѕРјР°РЅРґСѓ РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ
        fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=duration_hours)
        command = fw_cmd.get_command(self.default_firewall)

        decision["method"] = "copy_command"
        decision["firewall_type"] = self.default_firewall
        decision["command"] = command
        self.decisions_log.append(decision)

        logger.info(
            f"Р‘Р»РѕРєРёСЂРѕРІРєР° {ip} (alert={alert_id}): "
            f"webhook РЅРµ РЅР°СЃС‚СЂРѕРµРЅ, РєРѕРјР°РЅРґР° РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ"
        )

        return ConfirmFlowResult(
            action_id="block_ip",
            method=ActionMethod.COPY_COMMAND,
            success=True,
            message="РљРѕРјР°РЅРґР° РіРѕС‚РѕРІР° РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ",
            command=command,
        )

    def _send_webhook(
            self,
            ip: str,
            reason: str,
            alert_id: str,
            duration_hours: int,
            decision: dict,
    ) -> ConfirmFlowResult:
        """РћС‚РїСЂР°РІР»СЏРµС‚ webhook РЅР° Р±Р»РѕРєРёСЂРѕРІРєСѓ."""
        try:
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "type": "block_ip",
                "ip": ip,
                "reason": reason,
                "alert_id": alert_id,
                "duration_hours": duration_hours,
                "source": "Clarify Autonomous Security Layer",
                "timestamp": __import__("time").time(),
            }).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_config.url,
                data=payload,
                headers=self.webhook_config.headers or {},
                method=self.webhook_config.method,
            )

            with urllib.request.urlopen(req, timeout=self.webhook_config.timeout_seconds) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")

                decision["method"] = "webhook"
                decision["webhook_status"] = resp.status
                decision["webhook_response"] = response_body[:500]
                self.decisions_log.append(decision)

                logger.info(f"Webhook РѕС‚РїСЂР°РІР»РµРЅ: {ip} Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ (alert={alert_id})")

                return ConfirmFlowResult(
                    action_id="block_ip",
                    method=ActionMethod.WEBHOOK,
                    success=True,
                    message=f"IP {ip} Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ С‡РµСЂРµР· webhook",
                    webhook_response={
                        "status": resp.status,
                        "body": response_body[:200],
                    },
                )

        except urllib.error.URLError as e:
            decision["method"] = "webhook"
            decision["error"] = str(e.reason)
            self.decisions_log.append(decision)

            logger.error(f"РћС€РёР±РєР° webhook РґР»СЏ {ip}: {e.reason}")

            # Fallback: РІРѕР·РІСЂР°С‰Р°РµРј РєРѕРјР°РЅРґСѓ
            fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=duration_hours)

            return ConfirmFlowResult(
                action_id="block_ip",
                method=ActionMethod.COPY_COMMAND,
                success=False,
                message=f"Webhook РЅРµРґРѕСЃС‚СѓРїРµРЅ ({e.reason}). РСЃРїРѕР»СЊР·СѓР№С‚Рµ РєРѕРјР°РЅРґСѓ РІСЂСѓС‡РЅСѓСЋ.",
                command=fw_cmd.get_command(self.default_firewall),
            )

    def get_decisions(self, limit: int = 50) -> list[dict]:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ РїРѕСЃР»РµРґРЅРёРµ N СЂРµС€РµРЅРёР№ РёР· Р»РѕРіР°."""
        return self.decisions_log[-limit:]

    def is_webhook_available(self) -> bool:
        """Р”РѕСЃС‚СѓРїРµРЅ Р»Рё webhook (РЅР°СЃС‚СЂРѕРµРЅ + РїСЂРѕРІР°Р»РёРґРёСЂРѕРІР°РЅ)."""
        return (
                self.webhook_config is not None
                and bool(self.webhook_config.url)
                and self.webhook_config.validated
        )


# ------------------------------------------------------------------
# РўРµСЃС‚
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("РўРµСЃС‚ Confirm Flow")
    print()

    # РўРµСЃС‚ 1: Р‘РµР· webhook вЂ” РєРѕРјР°РЅРґР° РґР»СЏ РєРѕРїРёСЂРѕРІР°РЅРёСЏ
    print("1. Р‘РµР· webhook:")
    flow = ConfirmFlow(webhook_config=None)
    result = flow.execute_block(
        ip="203.0.113.45",
        reason="Brute-force RDP, С‡Р°СЃС‚РѕС‚Р° 404 РІ 11Г— РІС‹С€Рµ РЅРѕСЂРјС‹ (Clarify SHAP)",
        alert_id="alert-beaconing-203-0-113-45-12345",
        operator="admin",
    )
    print(f"   РњРµС‚РѕРґ: {result.method.value}")
    print(f"   РЈСЃРїРµС…: {result.success}")
    print(f"   РЎРѕРѕР±С‰РµРЅРёРµ: {result.message}")
    print(f"   РљРѕРјР°РЅРґР°:")
    print(f"   $ {result.command}")

    print()

    # РўРµСЃС‚ 2: Р’Р°Р»РёРґР°С†РёСЏ webhook (Р·Р°РІРµРґРѕРјРѕ РЅРµРґРѕСЃС‚СѓРїРЅС‹Р№)
    print("2. Р’Р°Р»РёРґР°С†РёСЏ РЅРµРґРѕСЃС‚СѓРїРЅРѕРіРѕ webhook:")
    flow2 = ConfirmFlow(
        webhook_config=WebhookConfig(url="http://localhost:99999/webhook")
    )
    valid, msg = flow2.validate_webhook()
    print(f"   Р”РѕСЃС‚СѓРїРµРЅ: {valid}")
    print(f"   РЎРѕРѕР±С‰РµРЅРёРµ: {msg}")
    print(f"   Р§РµРєР±РѕРєСЃ РґРѕСЃС‚СѓРїРµРЅ: {flow2.is_webhook_available()}")

    print()

    # РўРµСЃС‚ 3: Р›РѕРі СЂРµС€РµРЅРёР№
    print("3. Р›РѕРі СЂРµС€РµРЅРёР№:")
    for d in flow.get_decisions():
        print(f"   {d['ip']}: {d['reason'][:60]}... (РјРµС‚РѕРґ: {d['method']})")

    print("=" * 60)
