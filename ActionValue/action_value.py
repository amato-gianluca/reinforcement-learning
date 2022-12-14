import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from random import choice, random, randrange
from typing import Any, Callable, Generic, TypeVar

from ActionValue.environment import Environment

T = TypeVar('T')

def maximal_action(seq: list[T], key: Callable[[T], float]) -> T:
    '''Returns a randomly picked element of seq whose key value is maximal'''
    prefs = [key(x) for x in seq]
    m = max(prefs)
    matches = [i for i, x in enumerate(prefs) if x == m]
    return seq[choice(matches)]

@dataclass(eq=False)
class ActionInQueue(Generic[T]):
    action: T
    preference: float
    num_taken: int
    pi: float = field(init=False, default=0.0)   # Needed for some applications

class ActionValue(Generic[T], metaclass = ABCMeta):
    '''An action-value agent for reinforcement learning.'''

    def __init__(self, env: Environment[T], initial_preference: float = 0.0) -> None:
        '''Initialize the agent with environment and initial preference for all actions.'''
        self.env = env
        self.initial_preference = initial_preference
        self.actions = [ActionInQueue(a, initial_preference, 0) for a in env.actions()]

    def reset(self) -> None:
        '''Reinitialize agent and related environment.'''
        self.env.reset()
        for a in self.actions:
            a.preference = self.initial_preference
            a.num_taken = 0
            a.pi = 0.0

    @abstractmethod
    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        '''Updates the preferences of the agent given the executed action and corresponding
        reward.'''

    @abstractmethod
    def select_action(self) -> ActionInQueue[T]:
        '''Returns the action the agent would perform in the current step'''

    def step(self) -> tuple[T, float]:
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
        if random() < self.epsilon:
            return self.actions[randrange(0, len(self.actions))]
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
        return maximal_action(self.actions, key=self.__preference)

class MeanValueUpdate(ActionValue[T]):
    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        a.preference += (reward - a.preference) / a.num_taken

class ConstantStepUpdate(ActionValue[T]):
    def __init__(self, *args: Any, alpha: float, **kwargs: Any) -> None:
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        a.preference += self.alpha * (reward - a.preference)

class ExpWeightNoBias(ActionValue[T]):
    def __init__(self, *args: Any, alpha: float, **kwargs: Any) -> None:
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        a.pi = a.pi + self.alpha*(1-a.pi)
        a.preference += self.alpha / a.pi * (reward - a.preference)

class MeanValueGreedy(MeanValueUpdate[T], GreedyActionSelection[T]):
    pass

class MeanValueEpsilonGreedy(MeanValueUpdate[T], EpsilonGreedyActionSelection[T]):
    pass

class ConstantStepEpsilonGreedy(ConstantStepUpdate[T], EpsilonGreedyActionSelection[T]):
    pass

class ExpWeightNoBiasEpsilonGreedy(ExpWeightNoBias[T], EpsilonGreedyActionSelection[T]):
    pass

class MeanValueUCB(MeanValueUpdate[T], UCBActionSelection[T]):
    pass

class ConstantStepUCB(ConstantStepUpdate[T], UCBActionSelection[T]):
    pass

class GradientAlgorithm(ActionValue[T]):
    def __init__(self, *args: Any, alpha: float, baseline: bool = True, **kwargs: Any) -> None:
        self.alpha = alpha
        self.numsteps = 0
        self.baseline = baseline
        self.meanreward = 0.0
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.numsteps = 0
        self.meanreward = 0.0
        super().reset()

    def select_action(self) -> ActionInQueue[T]:
        self.numsteps += 1
        r = random()
        sum_exp = 0.0
        for a in self.actions:
            a.pi = math.exp(a.preference)
            sum_exp += a.pi
        for a in self.actions:
            a.pi /= sum_exp
        sum_cum = 0.0
        for a in self.actions:
            sum_cum += a.pi
            if sum_cum > r:
                return a
        return self.actions[-1]         # this should never happen, just to make mypy happy

    def update_preferences(self, a: ActionInQueue[T], reward: float) -> None:
        if self.baseline:
            self.meanreward += (reward - self.meanreward) / self.numsteps
        for act in self.actions:
            if act is a:
                act.preference += self.alpha*(reward - self.meanreward)*(1 - act.pi)
            else:
                act.preference -= self.alpha*(reward - self.meanreward)*act.pi

__all__ = ['ActionValue', 'GreedyActionSelection', 'EpsilonGreedyActionSelection',
           'UCBActionSelection', 'MeanValueUpdate', 'ConstantStepUpdate', 'MeanValueGreedy',
           'MeanValueEpsilonGreedy', 'ConstantStepEpsilonGreedy', 'MeanValueUCB', 'ConstantStepUCB',
           'GradientAlgorithm', 'ExpWeightNoBiasEpsilonGreedy']
