from __future__ import annotations
from typing import Any, Iterable, Generic, TypeVar
from abc import abstractmethod, ABC

T = TypeVar('T')

class Environment(Generic[T], ABC):
    """An abstract class representing an environment."""

    @abstractmethod
    def actions(self) -> Iterable[T]:
        """Returns an iterable of the actions which are supported by the environment."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Put the environment at the state immediately after creation."""
        pass

    @abstractmethod
    def copy(self) -> Environment[T]:
        """Create a copy of the environment, including activity logs"""
        pass

    @abstractmethod
    def interaction(self, action: T) -> float:
        """Interact with the environment executing the given action. It returns the reward."""
        pass
