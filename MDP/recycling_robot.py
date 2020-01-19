from typing import Tuple, Iterable
from enum import Enum, auto

from environment import Reward, Probability
from MDP.mdp import MDP

class State(Enum):
    high = auto()
    low = auto()

class Action(Enum):
    search = auto()
    wait = auto()
    recharge = auto()

class RecyclingRobot(MDP[State, Action]):

    def __init__(self, alpha: float, beta: float, rsearch: float, rwait: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.rsearch = rsearch
        self.rwait = rwait

    def states(self) -> Iterable[State]:
        return State

    def is_final(self, _state: State) -> bool:
        return False

    def actions(self, state: State)-> Iterable[Action]:
        if state == State.high:
            return [Action.wait, Action.search]
        else:
            return Action

    def p(self, s: State, a: Action) -> Iterable[Tuple[Probability, Reward, State]]:
        if s == State.high:
            if a == Action.search:
                return [
                    (Probability(self.alpha), Reward(self.rsearch), State.high),
                    (Probability(1.0-self.alpha), Reward(self.rsearch), State.low)
                ]
            else:
                return [
                    (Probability(1.0), Reward(self.rwait), State.high)
                ]
        else:
            if a == Action.search:
                return [
                    (Probability(self.beta), Reward(self.rsearch), State.low),
                    (Probability(1.0-self.beta), Reward(-3.0), State.high)
                ]
            elif a == Action.wait:
                return [
                    (Probability(1.0), Reward(self.rwait), State.low)
                ]
            else:
                return [
                    (Probability(1.0), Reward(0.0), State.high)
                ]
