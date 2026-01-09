import pickle
import matplotlib.pyplot as plt

with open("rewards.pkl", "rb") as f:
    data = pickle.load(f)

plt.figure(figsize=(12, 9))

# Reward
plt.subplot(3, 1, 1)
plt.plot(data["qlearning"]["reward"], label="Q-learning")
plt.plot(data["sarsa"]["reward"], label="SARSA")
plt.ylabel("Mean reward")
plt.legend()
plt.grid()

# Angle
plt.subplot(3, 1, 2)
plt.plot(data["qlearning"]["theta"], label="Q-learning")
plt.plot(data["sarsa"]["theta"], label="SARSA")
plt.ylabel("Angle (deg)")
plt.legend()
plt.grid()

# Angular velocity
plt.subplot(3, 1, 3)
plt.plot(data["qlearning"]["theta_dot"], label="Q-learning")
plt.plot(data["sarsa"]["theta_dot"], label="SARSA")
plt.ylabel("Angular velocity (deg/s)")
plt.xlabel("Episode")
plt.legend()
plt.grid()

plt.suptitle("Q-learning vs SARSA: Pendulum stabilization")
plt.show()
