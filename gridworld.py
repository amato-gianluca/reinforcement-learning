from MDP import MDP

class GridWorld(MDP):
    rows: int
    cols: int
    wrap: bool

    def states(self):
        """A state for a grid world is the set of all possible pairs r, c where r is
        the row number and c is the column number."""
        return zip(range(self.rows), range(self.cols))

    def actions(self, s):
        return [ "L", "R", "U", "D" ]

    def normal_move(self, s, a):
        r, c = s
        if a == "L":
            c -= 1
            if c == -1:    
                c = self.cols - 1 if self.wrap else 1
        if a == "R":
            c += 1
            if c == self.cols:
                c = 0 if self.wrap else self.cols - 1
        if a == "U":
            r -= 1
            if r == -1:
                r = self.rows-1 if self.wrap else 1
        if a == "D":
            r += 1
            if r == self.rows:
                r = 0 if self.wrap else self.rows - 1
        return r, c         

    
