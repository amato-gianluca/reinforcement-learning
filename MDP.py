from typing import Generic, TypeVar, Tuple, Iterable, NewType, Dict, List
from abc import abstractmethod, ABC

S = TypeVar('S')
A = TypeVar('A')
Reward = NewType('Reward', float)
Probability = NewType('Probability', float)
Policy = Dict[S, List[Tuple[Probability,A]]]

class MDP(Generic[S,A], ABC):
    """A class representing a Markov Decision Process"""

    @abstractmethod
    def states(self) -> Iterable[S]:
        """Returns the set of states of this MDP"""
        pass

    @abstractmethod
    def actions(self, s: S) -> Iterable[A]:
        """Returns the set of actions available in states s"""
        pass

    @abstractmethod
    def p(self, s: S, a: A) -> Iterable[Tuple[Probability, Reward, S]]:
        """Returns a list of triples (p, r, s') shich is the set of possible
        pairs of reward and newstate with related probability"""
        pass

    def random_policy(self) -> Policy[S, A]:
        """Return a random policy where each action has the same probability"""
        policy = { }
        for s in self.states():
            actions = list(self.actions(s))
            n = len(actions)
            policy[s] = [ (Probability(1/n), a) for a in actions ]
        return policy

__all__ = [ 'MDP', 'Probability', 'Reward', 'Policy']