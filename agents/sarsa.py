from agents.base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    """
        SARSA method
    """
    def update(self, s, a, r, s_next, a_next):
        i, j = self.discretize(s)
        ni, nj = self.discretize(s_next)

        td_target = r + self.gamma * self.Q[ni, nj, a_next]

        self.Q[i, j, a] += self.alpha * (td_target - self.Q[i, j, a])
