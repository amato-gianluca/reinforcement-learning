class Environment:
    """An abstract class representing an environment."""

    def actions(self):
        """Returns an iterable of the actions which are supported by the environment."""
        pass

    def reset(self):
        """Put the environment at the state immediately after creation."""
        pass

    def copy(self):
        """Create a copy of the environment, including activity logs"""
        pass

    def interaction(self, action: any) -> float:
        """Interact with the environment executing the given action. It returns the reward."""
        pass
