from typing import Iterable

from environment import Probability, Reward

from MDP.gridworld import Action, Cell, GridWorld

class GridWorld1(GridWorld):
    def __init__(self) -> None:
        super().__init__(rows=5, cols=5, wrap=False)

    def is_final(self, _state: Cell) -> bool:
        return False

    def p(self, s: Cell, a: Action) -> Iterable[tuple[Probability, Reward, Cell]]:
        if s == (0, 1):
            return [(Probability(1.0), Reward(10.0), (4, 1))]
        elif s == (0, 3):
            return [(Probability(1.0), Reward(5.0), (2, 3))]
        else:
            s1, border = self.normal_move(s, a)
            return [(Probability(1.0), Reward(-1.0 * border), s1)]

class GridWorld2(GridWorld):
    def __init__(self) -> None:
        super().__init__(rows=4, cols=4, wrap=False)

    def is_final(self, state: Cell) -> bool:
        return state == (0, 0) or state == (3, 3)

    def p(self, s: Cell, a: Action) -> Iterable[tuple[Probability, Reward, Cell]]:
        if s == (0, 0) or s == (3, 3):
            return [(Probability(1.0), Reward(0.0), s)]
        else:
            return [(Probability(1.0), Reward(-1.0), self.normal_move(s, a)[0])]

__all__ = ['GridWorld1', 'GridWorld2']
