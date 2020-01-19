from typing import Generic, TypeVar, MutableMapping, Optional

from environment import Environment, Reward, Policy

S = TypeVar('S')
A = TypeVar('A')

StateValue = MutableMapping[S, float]

def policy_evaluation(env: Environment[S,A], pi: Policy[S,A], gamma: float, Delta: float) -> StateValue[S]:
    """Evaluate a policy using Monte Carlo methods"""
    v = { s : 0.0 for s in env.states() }
    numvisits = { s : 0 for s in env.states() }
    while True:
        Delta = 0
        G = 0.0
        episode = env.generate_episode()
        for i in range( len(episode) - 1, -1, -1):
            state, _action, reward = episode[i]
            if any( s == state for s, a, r in episode[0:i]):
                continue
            G += gamma * reward
            v[state] += (reward - v[state])/numvisits[state]
            numvisits[state] += 1

