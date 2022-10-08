from random import choice
from typing import Optional, TypeVar

from environment import (Environment, Policy, Probability, StateActionValue,
                         StateValue)

S = TypeVar('S')
A = TypeVar('A')

def policy_evaluation(env: Environment[S, A], pi: Policy[S, A], gamma: float, iterates: int) -> StateValue[S]:
    """Evaluate a policy using Monte Carlo methods"""
    v = {s : 0.0 for s in env.states()}
    numvisits = {s : 0 for s in env.states()}
    for _i in range(iterates):
        G = 0.0
        episode = env.generate_episode(pi)
        for i in range(len(episode)-1, -1, -1):
            state, _, reward = episode[i]
            G = G*gamma + reward
            if any(s == state for s, _, r in episode[0:i]):
                continue
            numvisits[state] += 1
            v[state] += (G - v[state])/numvisits[state]
    return v

def policy_evaluation_es(env: Environment[S, A], pi: Policy[S, A], gamma: float,
                         iterates: int) -> StateActionValue[S, A]:
    """Evaluate a policy using Monte Carlo methods and Exploring Starts method"""
    stateactions = [(s, a) for s in env.states() for a in env.actions(s)]
    v = {(s, a) : 0.0 for s, a in stateactions}
    numvisits = {(s, a) : 0 for s, a in stateactions}
    for _i in range(iterates):
        G = 0.0
        s0, a0 = choice(stateactions)
        episode = env.generate_episode(pi, initial_state=s0, initial_action=a0)
        for i in range(len(episode)-1, -1, -1):
            state, action, reward = episode[i]
            G = G*gamma + reward
            if any(s == state and a == action for s, a, r in episode[0:i]):
                continue
            numvisits[state, action] += 1
            v[state, action] += (G - v[state, action])/numvisits[state, action]
    return v

def montecarlo_es(env: Environment[S, A], gamma: float, iterates: int,
                                  pi: Optional[Policy[S, A]] = None) -> Policy[S, A]:
    """Determine optimal policy using the Montecarlo Exploring Starts method. It only works if
       every policy is guaranteed to reach a terminal state."""
    stateactions = [(s, a) for s in env.states() for a in env.actions(s)]
    v = {(s, a) : 0.0 for s, a in stateactions}
    numvisits = {(s, a) : 0 for s, a in stateactions}
    if pi is None:
        pi = env.random_policy()
    for _i in range(iterates):
        G = 0.0
        s0, a0 = choice(stateactions)
        episode = env.generate_episode(pi, initial_state=s0, initial_action=a0)
        for i in range(len(episode)-1, -1, -1):
            state, action, reward = episode[i]
            G = G*gamma + reward
            if any(s == state and a == action for s, a, r in episode[0:i]):
                continue
            numvisits[state, action] += 1
            v[state, action] += (G - v[state, action])/numvisits[state, action]
            pi[state] = [(Probability(1.0), max(env.actions(state), key=lambda a: v[state, a]))]
        print(pi)
    return pi

__all__ = ['policy_evaluation', 'policy_evaluation_es', 'montecarlo_es']
