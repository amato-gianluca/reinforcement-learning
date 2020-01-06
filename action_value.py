import math
import random

from typing import Any, List, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass
from abc import abstractmethod, ABC

from environment import Environment

T = TypeVar('T')

def maximal_action(seq: List[T], key: Callable[[T], float]) -> T:
    prefs = [key(x) for x in seq]
    m = max(prefs)
    matches = [i for i, x in enumerate(prefs) if x == m]
    return seq[random.choice(matches)]

@dataclass
class ActionInQueue(Generic[T]):
    action: T
    preference: float
    num_taken: int

class ActionValue(Generic[T], ABC):
    '''An action-value agent for reinforcement learning.'''

    def __init__(self, env: Environment[T], initial_preference: float = 0.0) -> None:
        '''Initialize the agent with environment and initial preference for all actions.'''
        self.env = env
        self.initial_preference = initial_preference
        self.actions = [ ActionInQueue(a, initial_preference, 0) for a in env.actions() ]
    
    def reset(self) -> None:
        '''Reinitialize agent and related environment.'''
        self.env.reset()
        for a in self.actions:
            a.preference = self.initial_preference
            a.num_taken = 0

    @abstractmethod
    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        '''Updates the preferences of the agent given the executed action and corresponding reward.'''
        pass

    @abstractmethod
    def select_action(self) -> ActionInQueue[T]:
        '''Returns the action the agent would perform in the current step'''
        pass

    def step(self) -> Tuple[T, float]:
        '''Perform a single interaction with the environment and updates the internal state.
        It returns the action performed and the reward obtained.'''
        a = self.select_action()
        a.num_taken += 1
        reward = self.env.interaction(a.action)
        self.update_preferences(a, reward)
        return a.action, reward

class GreedyActionSelection(ActionValue[T]):
    def select_action(self) -> ActionInQueue[T]:
        return maximal_action(self.actions, lambda x: x.preference)

class EpsilonGreedyActionSelection(ActionValue[T]):
    def __init__(self, *args: Any, epsilon: float, **kwargs: Any) -> None:
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def select_action(self) -> ActionInQueue[T]:
        if random.random() < self.epsilon:
            return self.actions[random.randrange(0, len(self.actions))]
        else:
            return maximal_action(self.actions, lambda x: x.preference)

class UCBActionSelection(ActionValue[T]):
    def __init__(self, *args: Any, c: float, **kwargs: Any) -> None:
        self.c = c
        self.numsteps = 0
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.numsteps = 0
        super().reset()

    def __preference(self, a: ActionInQueue[T]) -> float:
        if a.num_taken > 0:
            return a.preference + self.c * math.sqrt(math.log(self.numsteps)/a.num_taken)
        else:
            return math.inf

    def select_action(self) -> ActionInQueue[T]:
        self.numsteps += 1
        return maximal_action(self.actions, key = self.__preference)

class MeanValueUpdate(ActionValue[T]):
    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        a.preference += (reward - a.preference) / a.num_taken

class ConstantStepUpdate(ActionValue[T]):
    def __init__(self, *args: Any, alpha: float, **kwargs: Any) -> None:
        self.alpha = alpha
        super().__init__(*args,**kwargs)

    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        a.preference += self.alpha * (reward - a.preference)

class MeanValueGreedy(MeanValueUpdate[T], GreedyActionSelection[T]):
    pass

class MeanValueEpsilonGreedy(MeanValueUpdate[T], EpsilonGreedyActionSelection[T]):
    pass

class ConstantStepEpsilonGreedy(ConstantStepUpdate[T], EpsilonGreedyActionSelection[T]):
    pass

class MeanValueUCB(MeanValueUpdate[T], UCBActionSelection[T]):
    pass

class ConstantStepUCB(ConstantStepUpdate[T], UCBActionSelection[T]):
    pass

__all__ = [ 'ActionValue' , 'GreedyActionSelection', 'EpsilonGreedyActionSelection', 'UCBActionSelection',
            'MeanValueUpdate', 'ConstantStepUpdate', 'MeanValueGreedy', 'MeanValueEpsilonGreedy',
            'ConstantStepEpsilonGreedy', 'MeanValueUCB', 'ConstantStepUCB']
