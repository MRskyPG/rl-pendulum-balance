import numpy as np

class BalanceEnv:
    # Simple pendulum
    def __init__(self):
        """
        Environment of simple number pendulum for RL
        """
        self.theta_dot = None
        self.theta = None
        self.dt = 0.05
        self.max_angle = np.pi / 4
        self.actions = [-1.0, 0.0, 1.0] #left, straight, right

    def reset(self):
        # Small initial angle and velocity for learning in new episode
        self.theta = np.random.uniform(-0.262, 0.262)
        self.theta_dot = np.random.uniform(-0.2, 0.2)
        return self.state()

    def state(self):
        return np.array([self.theta, self.theta_dot])

    def step(self, action):
        u = self.actions[action]

        self.theta_dot += self.dt * u
        self.theta += self.dt * self.theta_dot

        # Reward function
        reward = -(self.theta**2 + 0.1 * self.theta_dot**2)

        # Main condition
        done = abs(self.theta) > self.max_angle

        return self.state(), reward, done
