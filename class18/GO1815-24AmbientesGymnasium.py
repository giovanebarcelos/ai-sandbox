# GO1815-24AmbientesGymnasium
import gymnasium as gym

# CartPole: equilibrar vara em carrinho

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')

    # MountainCar: carro sobe montanha com momentum
    env = gym.make('MountainCar-v0')

    # LunarLander: pousar nave espacial
    env = gym.make('LunarLander-v2')

    # Atari Breakout
    env = gym.make('ALE/Breakout-v5')
