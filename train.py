import pickle
import numpy as np
from env import BalanceEnv
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent

EPISODES = 500
MAX_STEPS = 500

env = BalanceEnv()

agents = {
    "qlearning": QLearningAgent(),
    "sarsa": SARSAAgent()
}

stats = {}

for name, agent in agents.items():
    rewards = []
    thetas = []
    theta_dots = []

    for ep in range(EPISODES):
        state = env.reset()
        # initial action
        action = agent.act(state)

        ep_rewards = []
        ep_theta = []
        ep_theta_dot = []

        for _ in range(MAX_STEPS):
            next_state, reward, done = env.step(action)

            if name == "sarsa":
                next_action = agent.act(next_state)
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state)
                action = agent.act(next_state)

            state = next_state

            ep_rewards.append(reward)
            ep_theta.append(np.rad2deg(state[0]))
            ep_theta_dot.append(np.rad2deg(state[1]))

            if done:
                break

        # exponential decrease
        agent.epsilon *= 0.985

        mean_reward = np.mean(ep_rewards)
        rewards.append(mean_reward)
        thetas.append(np.mean(ep_theta))
        theta_dots.append(np.mean(ep_theta_dot))

        if ep % 50 == 0:
            print(f"{name} | episode {ep} | mean reward {mean_reward:.4f}")

    stats[name] = {
        "reward": rewards,
        "theta": thetas,
        "theta_dot": theta_dots
    }

with open("rewards.pkl", "wb") as f:
    pickle.dump(stats, f)

print("Training finished. Data saved.")
