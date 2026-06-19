"""
Confidence gating для всех детекторов SHARD.
Обеспечивает, что детектор не выдаёт score при недостаточности данных.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class GatingDecision(Enum):
    """Решение confidence gate."""
    PROCEED = "proceed"  # Данных достаточно, считаем score
    INSUFFICIENT_DATA = "insufficient_data"  # Слишком мало данных
    DETECTOR_DISABLED = "detector_disabled"  # Детектор выключен


@dataclass
class GatingResult:
    """Результат проверки confidence gate."""
    decision: GatingDecision
    score: Optional[float] = None
    reason: Optional[str] = None

    @property
    def should_proceed(self) -> bool:
        return self.decision == GatingDecision.PROCEED


def check_min_intervals(
        interarrival_times: list[float],
        min_required: int = 15
) -> GatingResult:
    """
    Проверяет, достаточно ли интервалов для расчёта оконных статистик.

    Args:
        interarrival_times: список интервалов между событиями в секундах
        min_required: минимальное требуемое количество

    Returns:
        GatingResult с решением
    """
    n_intervals = len(interarrival_times)

    if n_intervals < min_required:
        return GatingResult(
            decision=GatingDecision.INSUFFICIENT_DATA,
            score=None,
            reason=(
                f"Недостаточно интервалов: {n_intervals} < {min_required}. "
                f"Score не рассчитан."
            )
        )

    return GatingResult(decision=GatingDecision.PROCEED)


def check_min_events(
        event_count: int,
        min_required: int = 10
) -> GatingResult:
    """
    Проверяет минимальное количество событий (для DGA, BruteForce).

    Args:
        event_count: количество событий
        min_required: минимальное требуемое количество

    Returns:
        GatingResult с решением
    """
    if event_count < min_required:
        return GatingResult(
            decision=GatingDecision.INSUFFICIENT_DATA,
            score=None,
            reason=(
                f"Недостаточно событий: {event_count} < {min_required}. "
                f"Score не рассчитан."
            )
        )

    return GatingResult(decision=GatingDecision.PROCEED)