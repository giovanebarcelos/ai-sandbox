# GO1820-Exercício3CompararAlgoritmosNoLunarlander
from stable_baselines3 import PPO


if __name__ == "__main__":
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
