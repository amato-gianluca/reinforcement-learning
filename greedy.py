from random import random, randrange
from dataclasses import dataclass

@dataclass
class ActionInQueue:
    action: any
    estimated_reward: float
    num_taken: int

class EpsilonGreedy:
    '''An ε-greedy learner.'''

    def __init__(self, env, epsilon: float, initial_estimate: float = 0.0):
        '''Initialize the ε-greedy learner by specifying the environment e the value 
        for ε'''
        self.env = env
        self.epsilon = epsilon
        self.initial_estimate = initial_estimate
        self.actions = [ ActionInQueue(a, initial_estimate, 0) for a in env.actions() ]
    
    def reset(self):
        '''Reinitialize learner.'''
        self.env.reset()
        for a in self.actions:
            a.estimated_reward = self.initial_estimate
            a.num_taken = 0

    def alpha(self, n: int) -> float:
        '''Learning step.'''
        pass

    def step(self):
        '''Perform a single interaction with the environment and updates the internal state.
        It returns the action performed and the reward obtained'''
        if random() < self.epsilon:
            a = self.actions[randrange(0, len(self.actions))]
        else:
            a = max(self.actions, key = lambda x: x.estimated_reward)
        a.num_taken += 1
        reward = self.env.interaction(a.action)
        a.estimated_reward = a.estimated_reward + (reward - a.estimated_reward) * self.alpha(a.num_taken)
        return a.action, reward

class MeanValueEpsilonGreedy(EpsilonGreedy):
    '''An ε-greedy learner which alpha = 1/n.'''

    def alpha(self, n):
        return 1/n

class ConstantStepEpsilonGreedy(EpsilonGreedy):
    '''An ε-greedy learner with constant learning factor.'''

    def __init__(self, env, epsilon: float, initial_estimate: float = 0.0, rate: float = 0.1):
        super().__init__(env, epsilon = epsilon, initial_estimate = initial_estimate)
        self.rate = rate

    def alpha(self, n):
        return self.rate
   
__all__ = [ 'EpsilonGreedy' , 'MeanValueEpsilonGreedy', 'ConstantStepEpsilonGreedy' ]
