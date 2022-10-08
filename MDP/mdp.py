from abc import abstractmethod
from random import choice, random
from typing import Iterable, TypeVar

from environment import Environment, Probability, Reward

S = TypeVar('S')
A = TypeVar('A')

class MDP(Environment[S, A]):
    """A environment derived from a Markov Decision Process"""

    @abstractmethod
    def p(self, s: S, a: A) -> Iterable[tuple[Probability, Reward, S]]:
        """Returns a list of triples (p, r, s') shich is the set of possible pairs of reward and
        newstate with related probability"""

    def initial_state(self) -> S:
        return choice(list(self.states()))

    def interaction(self, state: S, action: A) -> tuple[Reward, S]:
        results = self.p(state, action)
        prob = random()
        prob_accum = 0.0
        for p, reward, newstate in results:
            prob_accum += p
            if prob < prob_accum:
                return reward, newstate
        raise Exception("Error no interaction selected. Thid should never happen.")

__all__ = ['MDP']