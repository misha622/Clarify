"""
Confirm-flow Р Т‘Р В»РЎРЏ Р Т‘Р ВµР в„–РЎРѓРЎвЂљР Р†Р С‘Р в„– Р Р…Р В°Р Т‘ Р В°Р В»Р ВµРЎР‚РЎвЂљР В°Р СР С‘ Clarify.

Р В Р ВµР В°Р В»Р С‘Р В·РЎС“Р ВµРЎвЂљ Human-in-the-loop:
- Р С™Р Р…Р С•Р С—Р С”Р В° "Р вЂ”Р В°Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р В°РЎвЂљРЎРЉ" Р СњР вЂў Р С‘РЎРѓР С—Р С•Р В»Р Р…РЎРЏР ВµРЎвЂљ Р Т‘Р ВµР в„–РЎРѓРЎвЂљР Р†Р С‘Р Вµ Р Р…Р В°Р С—РЎР‚РЎРЏР СРЎС“РЎР‹
- Р С›РЎвЂљР С”РЎР‚РЎвЂ№Р Р†Р В°Р ВµРЎвЂљ confirm-flow: Р СР С•Р Т‘Р В°Р В»РЎРЉР Р…Р С•Р Вµ Р С•Р С”Р Р…Р С• РІвЂ вЂ™ webhook Р С‘Р В»Р С‘ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘Р Вµ Р С”Р С•Р СР В°Р Р…Р Т‘РЎвЂ№
- Webhook Р С—РЎР‚Р С•Р Р†Р ВµРЎР‚РЎРЏР ВµРЎвЂљРЎРѓРЎРЏ Р С—РЎР‚Р С‘ РЎРѓР С•РЎвЂ¦РЎР‚Р В°Р Р…Р ВµР Р…Р С‘Р С‘ Р С”Р С•Р Р…РЎвЂћР С‘Р С–Р В°
- Р вЂўРЎРѓР В»Р С‘ webhook Р Р…Р Вµ Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р… РІР‚вЂќ Р В°Р Р†РЎвЂљР С•Р СР В°РЎвЂљР С‘РЎвЂЎР ВµРЎРѓР С”Р С‘ Р С—Р С•Р С”Р В°Р В·РЎвЂ№Р Р†Р В°Р ВµРЎвЂљ Р С”Р С•Р СР В°Р Р…Р Т‘РЎС“ Р Т‘Р В»РЎРЏ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ
- Р В§Р ВµР С”Р В±Р С•Р С”РЎРѓ "Р С‘РЎРѓР С—Р С•Р В»РЎРЉР В·Р С•Р Р†Р В°РЎвЂљРЎРЉ webhook" Р Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р… РЎвЂљР С•Р В»РЎРЉР С”Р С• Р С—РЎР‚Р С‘ Р Р†Р В°Р В»Р С‘Р Т‘Р Р…Р С•Р С webhook URL
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ActionMethod(Enum):
    """Р РЋР С—Р С•РЎРѓР С•Р В± Р С‘РЎРѓР С—Р С•Р В»Р Р…Р ВµР Р…Р С‘РЎРЏ Р Т‘Р ВµР в„–РЎРѓРЎвЂљР Р†Р С‘РЎРЏ."""
    WEBHOOK = "webhook"
    COPY_COMMAND = "copy_command"
    NONE = "none"


@dataclass
class FirewallCommand:
    """РљРѕРјР°РЅРґР° РґР»СЏ Р±Р»РѕРєРёСЂРѕРІРєРё IP РЅР° СЂР°Р·РЅС‹С… С„Р°РµСЂРІРѕР»Р°С…."""
    
    @staticmethod
    def _escape_comment(text: str) -> str:
        """Р­РєСЂР°РЅРёСЂСѓРµС‚ РєР°РІС‹С‡РєРё РІ РєРѕРјРјРµРЅС‚Р°СЂРёРё РґР»СЏ shell."""
        return text.replace('"', '\\"').replace("'", "'\\''")
    
    """Р С™Р С•Р СР В°Р Р…Р Т‘Р В° Р Т‘Р В»РЎРЏ Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”Р С‘ IP Р Р…Р В° РЎР‚Р В°Р В·Р Р…РЎвЂ№РЎвЂ¦ РЎвЂћР В°Р ВµРЎР‚Р Р†Р С•Р В»Р В°РЎвЂ¦."""
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
        return f"ufw deny from {self.ip} comment 'Clarify: {self._escape_comment(self.reason)}'"

    @property
    def windows_firewall(self) -> str:
        return (
            f"New-NetFirewallRule -DisplayName \"Clarify Block {self.ip}\" "
            f"-Direction Inbound -RemoteAddress {self.ip} -Action Block"
        )

    def get_command(self, firewall_type: str = "iptables") -> str:
        """Р вЂ™Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµРЎвЂљ Р С”Р С•Р СР В°Р Р…Р Т‘РЎС“ Р Т‘Р В»РЎРЏ РЎС“Р С”Р В°Р В·Р В°Р Р…Р Р…Р С•Р С–Р С• РЎвЂљР С‘Р С—Р В° РЎвЂћР В°Р ВµРЎР‚Р Р†Р С•Р В»Р В°."""
        commands = {
            "iptables": self.iptables,
            "firewalld": self.firewall_cmd,
            "ufw": self.ufw,
            "windows": self.windows_firewall,
        }
        return commands.get(firewall_type, self.iptables)


@dataclass
class WebhookConfig:
    """Р С™Р С•Р Р…РЎвЂћР С‘Р С–РЎС“РЎР‚Р В°РЎвЂ Р С‘РЎРЏ webhook Р Т‘Р В»РЎРЏ Р В°Р Р†РЎвЂљР С•-Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”Р С‘."""
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
    """Р В Р ВµР В·РЎС“Р В»РЎРЉРЎвЂљР В°РЎвЂљ confirm-flow."""
    action_id: str
    method: ActionMethod
    success: bool
    message: str
    command: Optional[str] = None
    webhook_response: Optional[dict] = None


class ConfirmFlow:
    """
    Р С›Р В±РЎР‚Р В°Р В±Р С•РЎвЂљРЎвЂЎР С‘Р С” confirm-flow Р Т‘Р В»РЎРЏ Р Т‘Р ВµР в„–РЎРѓРЎвЂљР Р†Р С‘Р в„–.

    Р СџРЎР‚Р В°Р Р†Р С‘Р В»Р В°:
    1. Р вЂўРЎРѓР В»Р С‘ webhook Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р… Р С‘ Р Р†Р В°Р В»Р С‘Р Т‘Р ВµР Р… РІвЂ вЂ™ Р С•РЎвЂљР С—РЎР‚Р В°Р Р†Р В»РЎРЏР ВµРЎвЂљ POST
    2. Р вЂўРЎРѓР В»Р С‘ webhook Р СњР вЂў Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р… РІвЂ вЂ™ Р С—Р С•Р С”Р В°Р В·РЎвЂ№Р Р†Р В°Р ВµРЎвЂљ Р С”Р С•Р СР В°Р Р…Р Т‘РЎС“ Р Т‘Р В»РЎРЏ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ
    3. Р В§Р ВµР С”Р В±Р С•Р С”РЎРѓ "Р С‘РЎРѓР С—Р С•Р В»РЎРЉР В·Р С•Р Р†Р В°РЎвЂљРЎРЉ webhook" Р Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р… РЎвЂљР С•Р В»РЎРЉР С”Р С• Р С—РЎР‚Р С‘ Р Р†Р В°Р В»Р С‘Р Т‘Р Р…Р С•Р С webhook
    4. Р вЂ™РЎРѓР Вµ РЎР‚Р ВµРЎв‚¬Р ВµР Р…Р С‘РЎРЏ Р В»Р С•Р С–Р С‘РЎР‚РЎС“РЎР‹РЎвЂљРЎРѓРЎРЏ
    """

    def __init__(
            self,
            webhook_config: Optional[WebhookConfig] = None,
            default_firewall: str = "iptables",
    ):
        """
        Args:
            webhook_config: Р С”Р С•Р Р…РЎвЂћР С‘Р С–РЎС“РЎР‚Р В°РЎвЂ Р С‘РЎРЏ webhook (None = Р Р…Р Вµ Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р…)
            default_firewall: РЎвЂљР С‘Р С— РЎвЂћР В°Р ВµРЎР‚Р Р†Р С•Р В»Р В° Р С—Р С• РЎС“Р СР С•Р В»РЎвЂЎР В°Р Р…Р С‘РЎР‹ Р Т‘Р В»РЎРЏ Р С”Р С•Р СР В°Р Р…Р Т‘
        """
        self.webhook_config = webhook_config
        self.default_firewall = default_firewall
        self.decisions_log: list[dict] = []

    def validate_webhook(self) -> tuple[bool, str]:
        """
        Р СџРЎР‚Р С•Р Р†Р ВµРЎР‚РЎРЏР ВµРЎвЂљ Р Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р Р…Р С•РЎРѓРЎвЂљРЎРЉ webhook URL.
        Р С›РЎвЂљР С—РЎР‚Р В°Р Р†Р В»РЎРЏР ВµРЎвЂљ РЎвЂљР ВµРЎРѓРЎвЂљР С•Р Р†РЎвЂ№Р в„– POST Р С‘ Р В¶Р Т‘РЎвЂРЎвЂљ 2xx.

        Returns:
            (РЎС“РЎРѓР С—Р ВµРЎвЂ¦, РЎРѓР С•Р С•Р В±РЎвЂ°Р ВµР Р…Р С‘Р Вµ)
        """
        if not self.webhook_config or not self.webhook_config.url:
            return False, "Webhook URL Р Р…Р Вµ Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р…"

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
                    return True, f"Webhook Р Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р… (status={resp.status})"
                else:
                    return False, f"Webhook Р Р†Р ВµРЎР‚Р Р…РЎС“Р В» РЎРѓРЎвЂљР В°РЎвЂљРЎС“РЎРѓ {resp.status}"

        except urllib.error.URLError as e:
            return False, f"Webhook Р Р…Р ВµР Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р…: {e.reason}"
        except Exception as e:
            return False, f"Р С›РЎв‚¬Р С‘Р В±Р С”Р В° Р С—РЎР‚Р С•Р Р†Р ВµРЎР‚Р С”Р С‘ webhook: {e}"

    def execute_block(
            self,
            ip: str,
            reason: str,
            alert_id: str,
            duration_hours: int = 24,
            operator: str = "unknown",
    ) -> ConfirmFlowResult:
        """
        Р вЂ™РЎвЂ№Р С—Р С•Р В»Р Р…РЎРЏР ВµРЎвЂљ Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”РЎС“ IP РЎвЂЎР ВµРЎР‚Р ВµР В· confirm-flow.

        Args:
            ip: IP Р Т‘Р В»РЎРЏ Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”Р С‘
            reason: Р С—РЎР‚Р С‘РЎвЂЎР С‘Р Р…Р В° (Р С‘Р В· SHAP-Р С•Р В±РЎР‰РЎРЏРЎРѓР Р…Р ВµР Р…Р С‘РЎРЏ)
            alert_id: ID Р В°Р В»Р ВµРЎР‚РЎвЂљР В°
            duration_hours: Р Т‘Р В»Р С‘РЎвЂљР ВµР В»РЎРЉР Р…Р С•РЎРѓРЎвЂљРЎРЉ Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”Р С‘
            operator: Р С”РЎвЂљР С• Р Р†РЎвЂ№Р С—Р С•Р В»Р Р…Р С‘Р В» Р Т‘Р ВµР в„–РЎРѓРЎвЂљР Р†Р С‘Р Вµ

        Returns:
            ConfirmFlowResult
        """
        # Р вЂєР С•Р С–Р С‘РЎР‚РЎС“Р ВµР С РЎР‚Р ВµРЎв‚¬Р ВµР Р…Р С‘Р Вµ
        decision = {
            "action": "block_ip",
            "ip": ip,
            "reason": reason,
            "alert_id": alert_id,
            "duration_hours": duration_hours,
            "operator": operator,
            "timestamp": __import__("time").time(),
        }

        # Р вЂўРЎРѓР В»Р С‘ webhook Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р… Р С‘ Р Р†Р В°Р В»Р С‘Р Т‘Р ВµР Р… РІР‚вЂќ Р С•РЎвЂљР С—РЎР‚Р В°Р Р†Р В»РЎРЏР ВµР С
        if (
                self.webhook_config
                and self.webhook_config.url
                and self.webhook_config.validated
        ):
            return self._send_webhook(ip, reason, alert_id, duration_hours, decision)

        # Р ВР Р…Р В°РЎвЂЎР Вµ РІР‚вЂќ Р Р†Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµР С Р С”Р С•Р СР В°Р Р…Р Т‘РЎС“ Р Т‘Р В»РЎРЏ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ
        fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=duration_hours)
        command = fw_cmd.get_command(self.default_firewall)

        decision["method"] = "copy_command"
        decision["firewall_type"] = self.default_firewall
        decision["command"] = command
        self.decisions_log.append(decision)

        logger.info(
            f"Р вЂР В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”Р В° {ip} (alert={alert_id}): "
            f"webhook Р Р…Р Вµ Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р…, Р С”Р С•Р СР В°Р Р…Р Т‘Р В° Р Т‘Р В»РЎРЏ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ"
        )

        return ConfirmFlowResult(
            action_id="block_ip",
            method=ActionMethod.COPY_COMMAND,
            success=True,
            message="Р С™Р С•Р СР В°Р Р…Р Т‘Р В° Р С–Р С•РЎвЂљР С•Р Р†Р В° Р Т‘Р В»РЎРЏ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ",
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
        """Р С›РЎвЂљР С—РЎР‚Р В°Р Р†Р В»РЎРЏР ВµРЎвЂљ webhook Р Р…Р В° Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р С”РЎС“."""
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

                logger.info(f"Webhook Р С•РЎвЂљР С—РЎР‚Р В°Р Р†Р В»Р ВµР Р…: {ip} Р В·Р В°Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р В°Р Р… (alert={alert_id})")

                return ConfirmFlowResult(
                    action_id="block_ip",
                    method=ActionMethod.WEBHOOK,
                    success=True,
                    message=f"IP {ip} Р В·Р В°Р В±Р В»Р С•Р С”Р С‘РЎР‚Р С•Р Р†Р В°Р Р… РЎвЂЎР ВµРЎР‚Р ВµР В· webhook",
                    webhook_response={
                        "status": resp.status,
                        "body": response_body[:200],
                    },
                )

        except urllib.error.URLError as e:
            decision["method"] = "webhook"
            decision["error"] = str(e.reason)
            self.decisions_log.append(decision)

            logger.error(f"Р С›РЎв‚¬Р С‘Р В±Р С”Р В° webhook Р Т‘Р В»РЎРЏ {ip}: {e.reason}")

            # Fallback: Р Р†Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµР С Р С”Р С•Р СР В°Р Р…Р Т‘РЎС“
            fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=duration_hours)

            return ConfirmFlowResult(
                action_id="block_ip",
                method=ActionMethod.COPY_COMMAND,
                success=False,
                message=f"Webhook Р Р…Р ВµР Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р… ({e.reason}). Р ВРЎРѓР С—Р С•Р В»РЎРЉР В·РЎС“Р в„–РЎвЂљР Вµ Р С”Р С•Р СР В°Р Р…Р Т‘РЎС“ Р Р†РЎР‚РЎС“РЎвЂЎР Р…РЎС“РЎР‹.",
                command=fw_cmd.get_command(self.default_firewall),
            )

    def get_decisions(self, limit: int = 50) -> list[dict]:
        """Р вЂ™Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµРЎвЂљ Р С—Р С•РЎРѓР В»Р ВµР Т‘Р Р…Р С‘Р Вµ N РЎР‚Р ВµРЎв‚¬Р ВµР Р…Р С‘Р в„– Р С‘Р В· Р В»Р С•Р С–Р В°."""
        return self.decisions_log[-limit:]

    def is_webhook_available(self) -> bool:
        """Р вЂќР С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р… Р В»Р С‘ webhook (Р Р…Р В°РЎРѓРЎвЂљРЎР‚Р С•Р ВµР Р… + Р С—РЎР‚Р С•Р Р†Р В°Р В»Р С‘Р Т‘Р С‘РЎР‚Р С•Р Р†Р В°Р Р…)."""
        return (
                self.webhook_config is not None
                and bool(self.webhook_config.url)
                and self.webhook_config.validated
        )


# ------------------------------------------------------------------
# Р СћР ВµРЎРѓРЎвЂљ
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Р СћР ВµРЎРѓРЎвЂљ Confirm Flow")
    print()

    # Р СћР ВµРЎРѓРЎвЂљ 1: Р вЂР ВµР В· webhook РІР‚вЂќ Р С”Р С•Р СР В°Р Р…Р Т‘Р В° Р Т‘Р В»РЎРЏ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ
    print("1. Р вЂР ВµР В· webhook:")
    flow = ConfirmFlow(webhook_config=None)
    result = flow.execute_block(
        ip="203.0.113.45",
        reason="Brute-force RDP, РЎвЂЎР В°РЎРѓРЎвЂљР С•РЎвЂљР В° 404 Р Р† 11Р“вЂ” Р Р†РЎвЂ№РЎв‚¬Р Вµ Р Р…Р С•РЎР‚Р СРЎвЂ№ (Clarify SHAP)",
        alert_id="alert-beaconing-203-0-113-45-12345",
        operator="admin",
    )
    print(f"   Р СљР ВµРЎвЂљР С•Р Т‘: {result.method.value}")
    print(f"   Р Р€РЎРѓР С—Р ВµРЎвЂ¦: {result.success}")
    print(f"   Р РЋР С•Р С•Р В±РЎвЂ°Р ВµР Р…Р С‘Р Вµ: {result.message}")
    print(f"   Р С™Р С•Р СР В°Р Р…Р Т‘Р В°:")
    print(f"   $ {result.command}")

    print()

    # Р СћР ВµРЎРѓРЎвЂљ 2: Р вЂ™Р В°Р В»Р С‘Р Т‘Р В°РЎвЂ Р С‘РЎРЏ webhook (Р В·Р В°Р Р†Р ВµР Т‘Р С•Р СР С• Р Р…Р ВµР Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р Р…РЎвЂ№Р в„–)
    print("2. Р вЂ™Р В°Р В»Р С‘Р Т‘Р В°РЎвЂ Р С‘РЎРЏ Р Р…Р ВµР Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р Р…Р С•Р С–Р С• webhook:")
    flow2 = ConfirmFlow(
        webhook_config=WebhookConfig(url="http://localhost:99999/webhook")
    )
    valid, msg = flow2.validate_webhook()
    print(f"   Р вЂќР С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р…: {valid}")
    print(f"   Р РЋР С•Р С•Р В±РЎвЂ°Р ВµР Р…Р С‘Р Вµ: {msg}")
    print(f"   Р В§Р ВµР С”Р В±Р С•Р С”РЎРѓ Р Т‘Р С•РЎРѓРЎвЂљРЎС“Р С—Р ВµР Р…: {flow2.is_webhook_available()}")

    print()

    # Р СћР ВµРЎРѓРЎвЂљ 3: Р вЂєР С•Р С– РЎР‚Р ВµРЎв‚¬Р ВµР Р…Р С‘Р в„–
    print("3. Р вЂєР С•Р С– РЎР‚Р ВµРЎв‚¬Р ВµР Р…Р С‘Р в„–:")
    for d in flow.get_decisions():
        print(f"   {d['ip']}: {d['reason'][:60]}... (Р СР ВµРЎвЂљР С•Р Т‘: {d['method']})")

    print("=" * 60)

