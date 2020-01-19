from __future__ import annotations

import random
from typing import List, TypeVar, Iterable
from abc import abstractmethod

from ActionValue.environment import Environment

class MultiArmedBandit(Environment[int]):
    def __init__(self, means: List[float], scale: float = 1.0, drift: float = 0.0):
        self.original_means = means.copy()
        self.num_actions = len(means)
        self.scale = scale
        self.drift = drift
        self.best_actions: List[int] = []
        self.rewards: List[float] = []
        self.reset()

    def reset(self) -> None:
        self.means = self.original_means.copy()
        self.best_action = max(range(self.num_actions), key = self.means.__getitem__)
        self.best_actions.clear()
        self.rewards.clear()

    def copy(self) -> MultiArmedBandit:
        mab = MultiArmedBandit(self.original_means, self.scale, self.drift)
        mab.means = self.means.copy()
        mab.best_action = self.best_action
        mab.best_actions = self.best_actions.copy()
        mab.rewards = self.rewards.copy()
        return mab

    def actions(self) -> Iterable[int]:
        return range(self.num_actions)

    def interaction(self, action: int) -> float:
        reward = random.normalvariate(self.means[action], self.scale)
        self.best_actions.append(action == self.best_action)
        self.rewards.append(reward)
        if self.drift != 0.0:
            for a in range(self.num_actions):
                self.means[a] += random.normalvariate(0.0, self.drift)
            self.best_action = max(range(self.num_actions), key = self.means.__getitem__)
        return reward

    @staticmethod
    def random_gen(n: int, mean_loc: float = 0.0, mean_scale: float = 1.0, scale: float = 1.0) -> MultiArmedBandit:
        means = [ random.normalvariate(mean_loc, mean_scale) for x in range(n) ]
        return MultiArmedBandit(means, scale)
