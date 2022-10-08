from typing import Iterable

from environment import Probability, Reward

from MDP.mdp import MDP

class Gambler(MDP[int, int]):
    def __init__(self, goal: int, ph: float) -> None:
        self.goal = goal
        self.ph = ph

    def states(self) -> Iterable[int]:
        return range(self.goal+1)

    def is_final(self, state: int) -> bool:
        return state == 0 or state == self.goal

    def actions(self, state: int) -> Iterable[int]:
        return range(min(state + 1, self.goal - state + 1))

    def p(self, s: int, a: int) -> Iterable[tuple[Probability, Reward, int]]:
        if a == 0:
            return [(Probability(1), Reward(0.0), 0)]
        else:
            return [
                (Probability(self.ph), Reward(1.0 if s+a == self.goal else 0.0), s+a),
                (Probability(1 - self.ph), Reward(0.0), s - a)
            ]

__all__ = ['Gambler']