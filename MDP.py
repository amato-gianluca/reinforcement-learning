from abc import ABC, abstractmethod

class MDP(ABC):
    """A class representing a Markov Decision Process"""

    @abstractmethod
    def states(self):
        """Returns the set of states of this MDP"""
        pass

    @abstractmethod
    def actions(self,s):
        """Returns the set of actions available in states s"""
        pass

    @abstractmethod
    def p(self, s, a):
        """Returns a list of triples (p, r, s') shich is the set of possible
        pairs of reward and newstate with related probability"""
        pass
