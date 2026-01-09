from agents.base_agent import BaseAgent
import numpy as np


class QLearningAgent(BaseAgent):
    """
        Q-learning method
    """
    def update(self, s, a, r, s_next):
        i, j = self.discretize(s)
        ni, nj = self.discretize(s_next)

        best_next = np.max(self.Q[ni, nj])
        td_target = r + self.gamma * best_next

        self.Q[i, j, a] += self.alpha * (td_target - self.Q[i, j, a])
