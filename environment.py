from __future__ import annotations
from typing import Generic, TypeVar, NewType, Dict, List, Tuple, Iterable, Iterator, Optional
from abc import abstractmethod, ABC

import random

S = TypeVar('S')
A = TypeVar('A')

Reward = NewType('Reward', float)
Probability = NewType('Probability', float)
Policy = Dict[S, List[Tuple[Probability,A]]]
Episode = List[Tuple[S, A, Reward]]

class Environment(Generic[S, A], ABC):
    """An abstract class representing an environment."""

    @abstractmethod
    def initial_state(self) -> S:
        """Returns one of the possible initial states"""

    @abstractmethod
    def states(self) -> Iterable[S]:
        """Returns an iterable of the states in which it is possible to find the environment."""
        pass

    @abstractmethod
    def actions(self, state: S) -> Iterable[A]:
        """Returns an iterable of the actions which are possible in the given state."""
        pass

    @abstractmethod
    def interaction(self, state: S, action: A) -> Tuple[Reward, S]:
        """Interact with the environment executing the given action on the given state. It returns the
        reward and new state."""
        pass

    def generate_episode(self, initial_state: Optional[S]) -> Episode[S,A]:
        """Generate an episode for the environment"""
        episode: Episode[S,A] = []
        state = self.initial_state() if initial_state is None else initial_state
        while True:
            actions = list(self.actions(state))
            if len(actions) == 0:
                return episode
            action = random.choice(actions)
            reward, newstate = self.interaction(state, action)
            episode.append((state, action, reward))
            state = newstate
