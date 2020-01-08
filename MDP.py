from typing import Generic, TypeVar, Tuple, Iterable, NewType
from abc import abstractmethod, ABC

S = TypeVar('S')
A = TypeVar('A')
Reward = NewType('Reward', float)
Probability = NewType('Probability', float)

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

__all__ = [ 'MDP', 'Probability', 'Reward' ]