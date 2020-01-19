from typing import Tuple, Iterable
from abc import abstractmethod
from random import random, choice

from environment import S, A, Environment, Probability, Reward

class MDP(Environment[S, A]):
    """A environment derived from a Markov Decision Process"""

    @abstractmethod
    def p(self, s: S, a: A) -> Iterable[Tuple[Probability, Reward, S]]:
        """Returns a list of triples (p, r, s') shich is the set of possible pairs of reward and
        newstate with related probability"""

    def initial_state(self) -> S:
        return choice(list(self.states()))

    def interaction(self, state: S, action: A) -> Tuple[Reward, S]:
        results = self.p(state, action)
        prob = random()
        prob_accum = 0.0
        for p, reward, newstate in results:
            prob_accum += p
            if prob < prob_accum:
                return reward, newstate
        return reward, newstate   # should never happen
