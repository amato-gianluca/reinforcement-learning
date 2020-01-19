from environment import S, A, Environment, Policy, StateValue

def policy_evaluation(env: Environment[S, A], pi: Policy[S, A], gamma: float, iterates: int) -> StateValue[S]:
    """Evaluate a policy using Monte Carlo methods"""
    v = {s : 0.0 for s in env.states()}
    numvisits = {s : 0 for s in env.states()}
    for _i in range(iterates):
        G = 0.0
        episode = env.generate_episode(pi)
        for i in range(len(episode)-1, -1, -1):
            state, _action, reward = episode[i]
            G = G*gamma + reward
            if any(s == state for s, _, r in episode[0:i]):
                continue
            numvisits[state] += 1
            v[state] += (G - v[state])/numvisits[state]
    return v
