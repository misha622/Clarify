"""
Базовый класс для детекторов Clarify.
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseDetector(ABC):
    """Абстрактный базовый класс для всех детекторов."""

    @abstractmethod
    def detect(self, *args, **kwargs) -> Any:
        """Обнаруживает аномалии."""
        pass
