from typing import Tuple, Iterable
from enum import Enum, auto
from itertools import product

from MDP.mdp import MDP

Cell = Tuple[int, int]

class Action(Enum):
    west = auto()
    east = auto()
    north = auto()
    south = auto()

class GridWorld(MDP[Cell, Action]):
    def __init__(self, rows: int, cols: int, wrap: bool):
        self.rows = rows
        self.cols = cols
        self.wrap = wrap

    def states(self) -> Iterable[Cell]:
        """A state for a grid world is the set of all possible pairs r, c where r is
        the row number and c is the column number."""
        return product(range(self.rows), range(self.cols))

    def actions(self, state: Cell) -> Iterable[Action]:
        return Action

    def normal_move(self, s: Cell, a: Action) -> Tuple[Cell, bool]:
        r, c = s
        border = False
        if a == Action.west:
            if c == 0:
                border = True
                c = self.cols - 1 if self.wrap else r
            else:
                c -= 1
        elif a == Action.east:
            if c == self.cols - 1:
                border = True
                c = 0 if self.wrap else c
            else:
                c += 1
        elif a == Action.north:
            if r == 0:
                border = True
                r = self.rows - 1 if self.wrap else r
            else:
                r -= 1
        elif a == Action.south:
            if r == self.rows - 1:
                border = True
                r = 0 if self.wrap else r
            else:
                r += 1
        return (r, c), border
