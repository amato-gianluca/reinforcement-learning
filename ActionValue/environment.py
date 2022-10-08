from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

A = TypeVar('A')

class Environment(Generic[A], ABC):
    """An abstract class representing an environment."""

    @abstractmethod
    def actions(self) -> Iterable[A]:
        """Returns an iterable of the actions which are supported by the environment."""

    @abstractmethod
    def reset(self) -> None:
        """Put the environment at the state immediately after creation."""

    @abstractmethod
    def copy(self) -> Environment[A]:
        """Create a copy of the environment, including activity logs"""

    @abstractmethod
    def interaction(self, action: A) -> float:
        """Interact with the environment executing the given action. It returns the reward."""

__all__ = ['Environment']
