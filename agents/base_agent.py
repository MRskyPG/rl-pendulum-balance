import numpy as np


class BaseAgent:
    def __init__(self, bins=(10, 10), alpha=0.1, gamma=0.9, epsilon=1.0):
        """
        Initialize Base RL agent
        Args:
            bins (tuple): number of discretization bins for (theta, theta_dot)
            alpha (float): learning rate (0 < alpha <= 1)
            gamma (float): discount factor for future rewards (0 < gamma < 1)
            epsilon (float): exploration rate for epsilon-greedy policy
        """
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.theta_bins = np.linspace(-0.785, 0.785, bins[0])
        self.theta_dot_bins = np.linspace(-1.0, 1.0, bins[1])

        self.Q = np.zeros((*bins, 3))

    def discretize(self, state):
        theta, theta_dot = state

        # Indexes of the interval (bin)
        i = np.digitize(theta, self.theta_bins) - 1
        j = np.digitize(theta_dot, self.theta_dot_bins) - 1
        return np.clip(i, 0, self.bins[0]-1), np.clip(j, 0, self.bins[1]-1)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        i, j = self.discretize(state)
        return np.argmax(self.Q[i, j])
