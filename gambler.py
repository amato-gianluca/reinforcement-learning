from MDP import Reward, Probability, MDP
from typing import Tuple, Iterable

class Gambler(MDP[int, int]):
    def __init__(self, goal: int, ph: float) -> None:
        self.goal = goal
        self.ph = ph

    def states(self) -> Iterable[int]:
        return range(self.goal+1)
    
    def actions(self, s: int) -> Iterable[int]:
        return range(min(s + 1, self.goal - s + 1))

    def p(self, s: int, a: int) -> Iterable[Tuple[Probability, Reward, int]]:
        if a == 0:
            return [ (Probability(1), Reward(0.0), 0) ]
        else: 
            return [ 
                (Probability(self.ph), Reward(1.0 if s+a == self.goal else 0.0), s+a),
                (Probability(1 - self.ph), Reward(0.0), s - a)
            ]
        