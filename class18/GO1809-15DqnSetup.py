# GO1809-15DqnSetup
import numpy as np
import gym
from tensorflow import keras
from keras import layers
from collections import deque
import random


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n  # 2 (esquerda/direita)

    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_size, activation='linear')  # Q-values
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse')
        return model

    Q_network = build_model()
    Q_target = build_model()
    Q_target.set_weights(Q_network.get_weights())

    replay_buffer = deque(maxlen=10000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
