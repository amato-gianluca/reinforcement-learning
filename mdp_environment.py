import random
from typing import Iterable, Generic, TypeVar, Tuple, List

from environment import Environment
from MDP.mdp import MDP, Reward

S = TypeVar('S')
A = TypeVar('A')

class MDPEnviroment(Environment):
    
    def __init__(self, mdp: MDP[S,A], initial_states: List[S]):
        self.mdp = mdp
        self.initial_states = initial_states

    def initial_state(self) -> S:
        return random.choice(self.initial_states)

    def states(self) -> Iterable[S]:
        return self.mdp.states()

    def actions(self, state: S) -> Iterable[A]:
        return self.mdp.actions(state)

    def interaction(self, state: S, action: A) -> Tuple[Reward,S]:
        results = self.mdp.p(state, action)
        prob = random.random()
        prob_accum = 0.0
        for p, reward, newstate in results:
            prob_accum += p
            if prob < prob_accum:
                return reward, newstate
        return reward, newstate   # should never happen
