import math
import random

from dataclasses import dataclass
from abc import abstractmethod, ABCMeta

def maximal_action(seq, key):
    prefs = [key(x) for x in seq]
    m = max(prefs)
    matches = [i for i, x in enumerate(prefs) if x == m]
    return seq[random.choice(matches)]

@dataclass
class ActionInQueue:
    action: any
    preference: float
    num_taken: int

class ActionValue(metaclass = ABCMeta):
    '''An action-value agent for reinforcement learning.'''

    def __init__(self, env, initial_preference: float = 0.0):
        '''Initialize the agent with environment and initial preference for all actions.'''
        self.env = env
        self.initial_preference = initial_preference
        self.actions = [ ActionInQueue(a, initial_preference, 0) for a in env.actions() ]
    
    def reset(self):
        '''Reinitialize agent and related environment.'''
        self.env.reset()
        for a in self.actions:
            a.preference = self.initial_preference
            a.num_taken = 0


    @abstractmethod
    def update_preferences(self, a, reward: float) -> float:
        '''Updates the preferences of the agent given the executed action and corresponding reward.'''
        pass

    @abstractmethod
    def select_action(self):
        '''Returns the action the agent would perform in the current step'''
        pass

    def step(self):
        '''Perform a single interaction with the environment and updates the internal state.
        It returns the action performed and the reward obtained.'''
        a = self.select_action()
        a.num_taken += 1
        reward = self.env.interaction(a.action)
        self.update_preferences(a, reward)
        return a.action, reward

class GreedyActionSelection(ActionValue):    
    def select_action(self):
        return maximal_action(self.actions, lambda x: x.preference)

class EpsilonGreedyActionSelection(ActionValue):
    epsilon: float

    def __init__(self, *args, epsilon: float, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def select_action(self):
        if random.random() < self.epsilon:
            return self.actions[random.randrange(0, len(self.actions))]
        else:
            return maximal_action(self.actions, lambda x: x.preference)

class UCBActionSelection(ActionValue):
    c: float
    numsteps: int

    def __init__(self, *args, c: float, **kwargs):
        self.c = c
        self.numsteps = 0
        super().__init__(*args, **kwargs)

    def reset(self):
        self.numsteps = 0
        super().reset()

    def __preference(self, a) -> float:
        if a.num_taken > 0:
            return a.preference + self.c * math.sqrt(math.log(self.numsteps)/a.num_taken)
        else:
            return math.inf

    def select_action(self):
        self.numsteps += 1
        return maximal_action(self.actions, key = self.__preference)

class MeanValueUpdate(ActionValue):
    def update_preferences(self, a, reward):
        a.preference += (reward - a.preference) / a.num_taken

class ConstantStepUpdate(ActionValue):
    alpha: float

    def __init__(self, *args, alpha: float, **kwargs):
        self.alpha = alpha
        super().__init__(*args,**kwargs)

    def update_preferences(self, a, reward):
        a.preference += self.alpha * (reward - a.preference)

class MeanValueGreedy(MeanValueUpdate, GreedyActionSelection):
    pass

class MeanValueEpsilonGreedy(MeanValueUpdate, EpsilonGreedyActionSelection):
    pass

class ConstantStepEpsilonGreedy(ConstantStepUpdate, EpsilonGreedyActionSelection):
    pass

class MeanValueUCB(MeanValueUpdate, UCBActionSelection):
    pass

class ConstantStepUCB(ConstantStepUpdate, UCBActionSelection):
    pass

__all__ = [ 'ActionValue' , 'GreedyActionSelection', 'EpsilonGreedyActionSelection', 'UCBActionSelection',
            'MeanValueUpdate', 'ConstantStepUpdate', 'MeanValueGreedy', 'MeanValueEpsilonGreedy',
            'ConstantStepEpsilonGreedy', 'MeanValueUCB', 'ConstantStepUCB']
