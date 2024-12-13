import gymnasium as gym

from stable_baselines3 import SAC

env = gym.make("InvertedPendulum-v4")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, log_interval=4)
model.save("sac_pendulum")