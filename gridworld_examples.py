from typing import Tuple, Iterable

from MDP import Probability, Reward
from gridworld import Cell, Action, GridWorld

class GridWorld1(GridWorld):
    def __init__(self) -> None:
        super().__init__(rows = 5, cols = 5, wrap = False)

    def p(self, s: Cell, a: Action) -> Iterable[Tuple[Probability, Reward, Cell]]:
        if s == (0, 1):
            return [ (Probability(1.0), Reward(10.0), (4,1)) ]
        elif s == (0, 3):
            return [ (Probability(1.0), Reward(10.0), (2,3)) ]
        else:
            s1, border = self.normal_move(s, a)
            return [ (Probability(1.0), Reward(-1.0 * border), s1) ]
