# GO1817-26ProjetoDqnCartpoleSetup
import numpy as np
import gymnasium as gym
from tensorflow import keras
from keras import layers
from collections import deque
import random

# Hiperparâmetros
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 10
REPLAY_BUFFER_SIZE = 10000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START

        # Q-network e Target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='mse')
        return model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    def replay(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)

        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Calcular Q-targets
        targets = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(q_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# Criar ambiente e agente
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Treinar
reward_history = []

for episode in range(EPISODES):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)
    agent.decay_epsilon()

    # Atualizar target network
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()

    # Média últimos 100 episódios
    avg_reward = np.mean(reward_history[-100:])

    if episode % 10 == 0:
        print(f"Episode {episode}/{EPISODES}, Reward: {total_reward}, "
              f"Avg(100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Resolvido?
    if avg_reward >= 195 and len(reward_history) >= 100:
        print(f"\n🎉 Resolvido no episódio {episode}!")
        agent.model.save('dqn_cartpole.h5')
        break

env.close()

# Testar modelo treinado
env = gym.make('CartPole-v1', render_mode='human')
model = keras.models.load_model('dqn_cartpole.h5')

for test_episode in range(5):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(500):
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        env.render()

        if terminated or truncated:
            break

    print(f"Test Episode {test_episode}: {total_reward} steps")

env.close()
