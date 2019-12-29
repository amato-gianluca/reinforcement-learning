from random import random, randrange
from dataclasses import dataclass
from abc import abstractmethod

@dataclass
class ActionInQueue:
    action: any
    preference: float
    num_taken: int

class ActionValue:
    '''An action-value agent for reinforcement learning.'''

    def __init__(self, env, initial_preference: float = 0.0):
        '''Initialize the agent'''
        self.env = env
        self.initial_preference = initial_preference
        self.actions = [ ActionInQueue(a, initial_preference, 0) for a in env.actions() ]
    
    def reset(self):
        '''Reinitialize agent.'''
        self.env.reset()
        self.actions = [ ActionInQueue(a, self.initial_preference, 0) for a in self.env.actions() ]

       # for a in self.actions:
       #     a.estimated_reward = self.initial_preference
       #     a.num_taken = 0

    @abstractmethod
    def update_preferences(self, a, reward: float) -> float:
        '''Updates the preferences of the agent given the executed action and corresponding reward'''
        pass

    @abstractmethod
    def select_action(self):
        '''Returns the action the agent would perform in the current step'''
        pass

    def step(self):
        '''Perform a single interaction with the environment and updates the internal state.
        It returns the action performed and the reward obtained'''
        a = self.select_action()
        a.num_taken += 1
        reward = self.env.interaction(a.action)
        self.update_preferences(a, reward)
        return a.action, reward

class GreedyActionSelection(ActionValue):
    def select_action(self):
        return max(self.actions, key = lambda x: x.preference)

class EpsilonGreedyActionSelection(ActionValue):
    epsilon: float

    def select_action(self):
        if random() < self.epsilon:
            return self.actions[randrange(0, len(self.actions))]
        else:
            return max(self.actions, key = lambda x: x.preference)

class MeanValueUpdate(ActionValue):
    def update_preferences(self, a, reward):
        a.preference = a.preference + (reward - a.preference) / a.num_taken

class ConstantStepUpdate(ActionValue):
    alpha: float
    def update_preferences(self, a, reward):
        a.preference = a.preference + self.alpha * (reward - a.preference)

class MeanValueGreedy(MeanValueUpdate, GreedyActionSelection):
    pass

class MeanValueEpsilonGreedy(MeanValueUpdate, EpsilonGreedyActionSelection):
    def __init__(self, env, epsilon: float, initial_preference: float = 0.0):
        super().__init__(env, initial_preference)
        self.epsilon = epsilon

class ConstantStepEpsilonGreedy(ConstantStepUpdate, EpsilonGreedyActionSelection):
    def __init__(self, env, epsilon: float, alpha: float, initial_preference: float = 0.0):
        super().__init__(env, initial_preference)
        self.epsilon = epsilon
        self.alpha = alpha

__all__ = [ 'ActionValue' , 'GreedyActionSelection', 'EpsilonGreedyActionSelection',
            'MeanValueUpdate', 'ConstantStepUpdate', 'MeanValueGreedy', 'MeanValueEpsilonGreedy',
            'ConstantStepEpsilonGreedy' ]
