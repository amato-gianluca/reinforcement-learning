import random

from environment import Environment

class MultiArmedBandit(Environment):
    def __init__(self, means, scale: float = 1.0):
        self.means = means
        self.scale = scale
        self.best_action = max(range(len(means)), key = means.__getitem__)
        self.best_actions= []
        self.rewards = []

    def reset(self):
        self.best_actions.clear()
        self.rewards.clear()

    def copy(self):
        mab = MultiArmedBandit(self.means, self.scale)
        mab.best_actions = self.best_actions.copy()
        mab.rewards = self.rewards.copy()
        return mab

    def actions(self) -> range:
        return range(0, len(self.means))

    def interaction(self, action: int) -> float:
        self.best_actions.append(True if action == self.best_action else False)
        reward = random.normalvariate(self.means[action], self.scale)
        self.rewards.append(reward)
        return reward

    @staticmethod
    def random_gen(n: int, mean_loc: float = 0.0, mean_scale: float = 1.0, scale: float = 1.0):
        means = [ random.normalvariate(mean_loc, mean_scale) for x in range(n) ]
        return MultiArmedBandit(means, scale)
