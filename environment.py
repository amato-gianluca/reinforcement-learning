from abc import abstractmethod

class Environment:
    """An abstract class representing an environment."""

    @abstractmethod
    def actions(self):
        """Returns an iterable of the actions which are supported by the environment."""
        pass

    @abstractmethod
    def reset(self):
        """Put the environment at the state immediately after creation."""
        pass

    @abstractmethod
    def copy(self):
        """Create a copy of the environment, including activity logs"""
        pass

    @abstractmethod
    def interaction(self, action: any) -> float:
        """Interact with the environment executing the given action. It returns the reward."""
        pass
