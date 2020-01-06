from typing import Tuple, NewType, List, Iterator
from enum import Enum, auto

from MDP import MDP

Cell = Tuple[int, int]

class GridWorldAction(Enum):
    L = auto()
    R = auto()
    U = auto()
    D = auto()

class GridWorld(MDP[Cell, GridWorldAction]):
    rows: int
    cols: int
    wrap: bool

    def states(self) -> Iterator[Cell]:
        """A state for a grid world is the set of all possible pairs r, c where r is
        the row number and c is the column number."""
        return zip(range(self.rows), range(self.cols))

    def actions(self, s: Cell) -> Iterator[GridWorldAction]:
        return iter(GridWorldAction)

    def normal_move(self, s: Cell, a: GridWorldAction) -> Cell:
        r, c = s
        if a == GridWorldAction.L:
            c -= 1
            if c == -1:    
                c = self.cols - 1 if self.wrap else 1
        if a == GridWorldAction.R:
            c += 1
            if c == self.cols:
                c = 0 if self.wrap else self.cols - 1
        if a == GridWorldAction.U:
            r -= 1
            if r == -1:
                r = self.rows-1 if self.wrap else 1
        if a == GridWorldAction.D:
            r += 1
            if r == self.rows:
                r = 0 if self.wrap else self.rows - 1
        return r, c
