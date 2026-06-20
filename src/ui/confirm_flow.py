"""
Confirm-flow для действий над алертами Clarify.

Реализует Human-in-the-loop:
- Кнопка "Заблокировать" НЕ исполняет действие напрямую
- Открывает confirm-flow: модальное окно → webhook или копирование команды
- Webhook проверяется при сохранении конфига
- Если webhook не настроен — автоматически показывает команду для копирования
- Чекбокс "использовать webhook" доступен только при валидном webhook URL
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ActionMethod(Enum):
    """Способ исполнения действия."""
    WEBHOOK = "webhook"
    COPY_COMMAND = "copy_command"
    NONE = "none"


@dataclass
class FirewallCommand:
    """Команда для блокировки IP на разных фаерволах."""
    ip: str
    reason: str
    duration_hours: int = 24

    @staticmethod
    def _escape_comment(text: str) -> str:
        """Экранирует кавычки в комментарии для shell."""
        return text.replace('"', '\\"').replace("'", "'\\''")

    @property
    def iptables(self) -> str:
        return (
            f"iptables -A INPUT -s {self.ip} -j DROP "
            f"-m comment --comment \"Clarify: {self._escape_comment(self.reason)} (alert)\""
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
        """Возвращает команду для указанного типа фаервола."""
        commands = {
            "iptables": self.iptables,
            "firewalld": self.firewall_cmd,
            "ufw": self.ufw,
            "windows": self.windows_firewall,
        }
        return commands.get(firewall_type, self.iptables)


@dataclass
class WebhookConfig:
    """Конфигурация webhook для авто-блокировки."""
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
    """Результат confirm-flow."""
    action_id: str
    method: ActionMethod
    success: bool
    message: str
    command: Optional[str] = None
    webhook_response: Optional[dict] = None


class ConfirmFlow:
    """
    Обработчик confirm-flow для действий.

    Правила:
    1. Если webhook настроен и валиден → отправляет POST
    2. Если webhook НЕ настроен → показывает команду для копирования
    3. Чекбокс "использовать webhook" доступен только при валидном webhook
    4. Все решения логируются
    """

    def __init__(
            self,
            webhook_config: Optional[WebhookConfig] = None,
            default_firewall: str = "iptables",
    ):
        """
        Args:
            webhook_config: конфигурация webhook (None = не настроен)
            default_firewall: тип фаервола по умолчанию для команд
        """
        self.webhook_config = webhook_config
        self.default_firewall = default_firewall
        self.decisions_log: list[dict] = []

    def validate_webhook(self) -> tuple[bool, str]:
        """
        Проверяет доступность webhook URL.
        Отправляет тестовый POST и ждёт 2xx.

        Returns:
            (успех, сообщение)
        """
        if not self.webhook_config or not self.webhook_config.url:
            return False, "Webhook URL не настроен"

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
                    return True, f"Webhook доступен (status={resp.status})"
                else:
                    return False, f"Webhook вернул статус {resp.status}"

        except urllib.error.URLError as e:
            return False, f"Webhook недоступен: {e.reason}"
        except Exception as e:
            return False, f"Ошибка проверки webhook: {e}"

    def execute_block(
            self,
            ip: str,
            reason: str,
            alert_id: str,
            duration_hours: int = 24,
            operator: str = "unknown",
    ) -> ConfirmFlowResult:
        """
        Выполняет блокировку IP через confirm-flow.

        Args:
            ip: IP для блокировки
            reason: причина (из SHAP-объяснения)
            alert_id: ID алерта
            duration_hours: длительность блокировки
            operator: кто выполнил действие

        Returns:
            ConfirmFlowResult
        """
        # Логируем решение
        decision = {
            "action": "block_ip",
            "ip": ip,
            "reason": reason,
            "alert_id": alert_id,
            "duration_hours": duration_hours,
            "operator": operator,
            "timestamp": __import__("time").time(),
        }

        # Если webhook настроен и валиден — отправляем
        if (
                self.webhook_config
                and self.webhook_config.url
                and self.webhook_config.validated
        ):
            return self._send_webhook(ip, reason, alert_id, duration_hours, decision)

        # Иначе — возвращаем команду для копирования
        fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=duration_hours)
        command = fw_cmd.get_command(self.default_firewall)

        decision["method"] = "copy_command"
        decision["firewall_type"] = self.default_firewall
        decision["command"] = command
        self.decisions_log.append(decision)

        logger.info(
            f"Блокировка {ip} (alert={alert_id}): "
            f"webhook не настроен, команда для копирования"
        )

        return ConfirmFlowResult(
            action_id="block_ip",
            method=ActionMethod.COPY_COMMAND,
            success=True,
            message="Команда готова для копирования",
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
        """Отправляет webhook на блокировку."""
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

                logger.info(f"Webhook отправлен: {ip} заблокирован (alert={alert_id})")

                return ConfirmFlowResult(
                    action_id="block_ip",
                    method=ActionMethod.WEBHOOK,
                    success=True,
                    message=f"IP {ip} заблокирован через webhook",
                    webhook_response={
                        "status": resp.status,
                        "body": response_body[:200],
                    },
                )

        except urllib.error.URLError as e:
            decision["method"] = "webhook"
            decision["error"] = str(e.reason)
            self.decisions_log.append(decision)

            logger.error(f"Ошибка webhook для {ip}: {e.reason}")

            # Fallback: возвращаем команду
            fw_cmd = FirewallCommand(ip=ip, reason=reason, duration_hours=duration_hours)

            return ConfirmFlowResult(
                action_id="block_ip",
                method=ActionMethod.COPY_COMMAND,
                success=False,
                message=f"Webhook недоступен ({e.reason}). Используйте команду вручную.",
                command=fw_cmd.get_command(self.default_firewall),
            )

    def get_decisions(self, limit: int = 50) -> list[dict]:
        """Возвращает последние N решений из лога."""
        return self.decisions_log[-limit:]

    def is_webhook_available(self) -> bool:
        """Доступен ли webhook (настроен + провалидирован)."""
        return (
                self.webhook_config is not None
                and bool(self.webhook_config.url)
                and self.webhook_config.validated
        )


# ------------------------------------------------------------------
# Тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Тест Confirm Flow")
    print()

    # Тест 1: Без webhook — команда для копирования
    print("1. Без webhook:")
    flow = ConfirmFlow(webhook_config=None)
    result = flow.execute_block(
        ip="203.0.113.45",
        reason='Brute-force RDP, частота 404 в 11× выше нормы (Clarify SHAP)',
        alert_id="alert-beaconing-203-0-113-45-12345",
        operator="admin",
    )
    print(f"   Метод: {result.method.value}")
    print(f"   Успех: {result.success}")
    print(f"   Сообщение: {result.message}")
    print(f"   Команда:")
    print(f"   $ {result.command}")

    print()

    # Тест 2: Валидация webhook (заведомо недоступный)
    print("2. Валидация недоступного webhook:")
    flow2 = ConfirmFlow(
        webhook_config=WebhookConfig(url="http://localhost:99999/webhook")
    )
    valid, msg = flow2.validate_webhook()
    print(f"   Доступен: {valid}")
    print(f"   Сообщение: {msg}")
    print(f"   Чекбокс доступен: {flow2.is_webhook_available()}")

    print()

    # Тест 3: Лог решений
    print("3. Лог решений:")
    for d in flow.get_decisions():
        print(f"   {d['ip']}: {d['reason'][:60]}... (метод: {d['method']})")

    print()

    # Тест 4: Проверка экранирования кавычек
    print("4. Проверка экранирования кавычек:")
    cmd = FirewallCommand(ip="203.0.113.45", reason='Test "quoted" reason')
    print(f"   iptables: {cmd.iptables}")
    print(f"   ufw: {cmd.ufw}")

    print("=" * 60)
