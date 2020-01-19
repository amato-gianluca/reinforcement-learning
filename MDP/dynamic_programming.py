from typing import Optional, Tuple
import math

from environment import S, A, Probability, Policy, StateValue
from MDP.mdp import MDP

def policy_evaluation(mdp: MDP[S, A], pi: Policy[S, A], gamma: float, theta: float,
                      v: Optional[StateValue[S]] = None) -> StateValue[S]:
    """Returns the state-value function for a given MDP and policy. gamma is the discount rate while
    theta is used to detect convergenze of successive approximations. The smaller is theta, the
    more precise is the result. If v is given, computation starts from the provided state-value
    function, otherwise computation starts from an always null state-value function."""
    if v is None:
        v = {s : 0.0 for s in mdp.states()}
    while True:
        delta = 0.0
        for s in mdp.states():
            newvalue = 0.0
            for pi_a, a in pi[s]:
                newvalue += pi_a * sum(p * (r + gamma*v[snew]) for p, r, snew in mdp.p(s, a))
            delta = max(delta, abs(newvalue-v[s]))
            v[s] = newvalue
        if delta < theta:
            break
    return v

def policy_improvement(mdp: MDP[S, A], v: StateValue[S], gamma: float) -> Policy[S, A]:
    """Returns the best policy for a given state-value function v and discount factor gamma"""
    pi = {}
    for s in mdp.states():
        best_action = None
        best_action_value = -math.inf
        for a in mdp.actions(s):
            newvalue = sum(p * (r + gamma*v[snew]) for p, r, snew in mdp.p(s, a))
            if newvalue > best_action_value:
                best_action_value = newvalue
                best_action = a
        pi[s] = [(Probability(1.0), best_action)] if best_action is not None else []
    return pi

def policy_iteration(mdp: MDP[S, A], gamma: float, theta: float,
                     pi: Optional[Policy[S, A]] = None) -> Tuple[Policy[S, A], StateValue[S]]:
    """Return best policy and state-value function for an MDP. gamma and theta have the same
    meaning as in policy_evaluation, and pi is an optional initial policy"""
    if pi is None:
        pi = mdp.random_policy()
    v = {s : 0.0 for s in mdp.states()}
    while True:
        policy_evaluation(mdp, pi, gamma, theta, v)
        pinew = policy_improvement(mdp, v, gamma)
        if pi == pinew:
            break
        pi = pinew
    return pi, v

def value_iteration(mdp: MDP[S, A], gamma: float, theta: float) -> Tuple[Policy[S, A], StateValue[S]]:
    """Similar to  policy_iteration but use the value-iteration method instead"""
    v = {s : 0.0 for s in mdp.states()}
    while True:
        delta = 0.0
        for s in mdp.states():
            best = -math.inf
            for a in mdp.actions(s):
                newvalue = sum(p * (r + gamma*v[snew]) for p, r, snew in mdp.p(s, a))
                best = max(best, newvalue)
            delta = max(delta, abs(best-v[s]))
            v[s] = best
        if delta < theta:
            break
    return policy_improvement(mdp, v, gamma), v
