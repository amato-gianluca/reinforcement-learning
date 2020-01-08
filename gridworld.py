from typing import Tuple, NewType, List, Iterable
from enum import Enum, auto
from itertools import product

from MDP import MDP

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

    def actions(self, s: Cell) -> Iterable[Action]:
        return Action

    def normal_move(self, s: Cell, a: Action) -> Tuple[Cell, bool]:
        r, c = s
        border = False
        if a == Action.west:
            c -= 1
            if c == -1:
                border = True
                c = self.cols - 1 if self.wrap else 1
        elif a == Action.east:
            c += 1
            if c == self.cols:
                border = True
                c = 0 if self.wrap else self.cols - 1
        elif a == Action.north:
            r -= 1
            if r == -1:
                border = True
                r = self.rows-1 if self.wrap else 1
        elif a == Action.south:
            r += 1
            if r == self.rows:
                border = True
                r = 0 if self.wrap else self.rows - 1
        return (r, c), border
