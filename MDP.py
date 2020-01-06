from typing import Generic, TypeVar, Tuple, Iterator
from abc import abstractmethod, ABC

S = TypeVar('S')
A = TypeVar('A')

class MDP(Generic[S,A], ABC):
    """A class representing a Markov Decision Process"""

    @abstractmethod
    def states(self) -> Iterator[S]:
        """Returns the set of states of this MDP"""
        pass

    @abstractmethod
    def actions(self,s: S) -> Iterator[A]:
        """Returns the set of actions available in states s"""
        pass

    @abstractmethod
    def p(self, s: S, a: A) -> Iterator[Tuple[float, float, S]]:
        """Returns a list of triples (p, r, s') shich is the set of possible
        pairs of reward and newstate with related probability"""
        pass
