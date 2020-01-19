from __future__ import annotations
from typing import Generic, TypeVar, NewType, Dict, List, Tuple, Iterable, Iterator, Optional
from abc import abstractmethod, ABC

import random

S = TypeVar('S')
A = TypeVar('A')

Reward = NewType('Reward', float)
Probability = NewType('Probability', float)
Policy = Dict[S, List[Tuple[Probability, A]]]
Episode = List[Tuple[S, A, Reward]]

def policy_action_selection(pi: Policy[S,A], s: S) -> A:
        """Returns a random action following a policy"""
        prob = random.random()
        prob_accum = 0.0
        for p, a in pi[s]:
            prob_accum += p
            if prob < prob_accum:
                return a
        return a     # should never happen
    
class Environment(Generic[S, A], ABC):
    """An abstract class representing an environment."""

    @abstractmethod
    def initial_state(self) -> S:
        """Returns one of the possible initial states"""

    @abstractmethod
    def is_final(self, state: S) -> bool:
        """Returns whether a state is final for an episode"""

    @abstractmethod
    def states(self) -> Iterable[S]:
        """Returns an iterable of the states in which it is possible to find the environment."""

    @abstractmethod
    def actions(self, state: S) -> Iterable[A]:
        """Returns an iterable of the actions which are possible in the given state."""

    @abstractmethod
    def interaction(self, state: S, action: A) -> Tuple[Reward, S]:
        """Interact with the environment executing the given action on the given state. It returns the
        reward and new state."""

    @abstractmethod
    def random_policy(self) -> Policy[S, A]:
        """Returns the random policy for the given environment"""

    def generate_episode(self, pi: Policy[S, A], initial_state: Optional[S] = None) -> Episode[S,A]:
        """Generate an episode for the environment"""
        episode: Episode[S,A] = []
        state = self.initial_state() if initial_state is None else initial_state
        while not self.is_final(state):
            action = policy_action_selection(pi, state)
            reward, newstate = self.interaction(state, action)
            episode.append((state, action, reward))
            state = newstate
        return episode
