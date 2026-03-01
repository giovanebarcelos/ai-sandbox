# GO1823-Codigo
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import deque
import random
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 1. DUELING DQN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

def build_dueling_dqn(state_size, action_size, learning_rate=0.001):
    """
    Dueling DQN separa Value Stream e Advantage Stream
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    inputs = layers.Input(shape=(state_size,))

    # Shared feature extractor
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)

    # Value stream
    value_stream = layers.Dense(64, activation='relu')(x)
    value = layers.Dense(1, activation='linear', name='value')(value_stream)

    # Advantage stream
    advantage_stream = layers.Dense(64, activation='relu')(x)
    advantage = layers.Dense(action_size, activation='linear', name='advantage')(advantage_stream)

    # Aggregate: Q(s,a) = V(s) + (A(s,a) - mean(A))
    # Subtração da média garante identificabilidade única
    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    model = keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    return model

# ═══════════════════════════════════════════════════════════════════
# 2. DOUBLE DQN AGENT
# ═══════════════════════════════════════════════════════════════════

class DoubleDuelingDQNAgent:
    """
    Double DQN: Usa Q-network para selecionar ação, Q-target para avaliar
    Dueling: Arquitetura separada para Value e Advantage
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10

        # Memory
        self.memory = deque(maxlen=10000)

        # Networks
        self.model = build_dueling_dqn(state_size, action_size, self.learning_rate)
        self.target_model = build_dueling_dqn(state_size, action_size, self.learning_rate)
        self.update_target_model()

        print("="*70)
        print("DOUBLE DUELING DQN AGENT INICIALIZADO")
        print("="*70)
        self.model.summary()

    def update_target_model(self):
        """Copiar pesos do Q-network para target network"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Armazenar transição na memória"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Treinar com mini-batch usando Double DQN"""
        if len(self.memory) < self.batch_size:
            return

        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)

        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Double DQN: usar model para SELECIONAR ação, target_model para AVALIAR
        targets = self.model.predict(states, verbose=0)

        # Ações selecionadas pelo Q-network
        next_q_values_model = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(next_q_values_model, axis=1)

        # Q-values avaliados pelo target network
        next_q_values_target = self.target_model.predict(next_states, verbose=0)

        # Calcular targets com Double DQN
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # Double DQN: Q_target(s', argmax_a Q(s',a))
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_values_target[i][best_actions[i]]

        # Treinar
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

    def decay_epsilon(self):
        """Decair epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ═══════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("INICIANDO TREINAMENTO - CARTPOLE COM DOUBLE DUELING DQN")
print("="*70)

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n  # 2

agent = DoubleDuelingDQNAgent(state_size, action_size)

num_episodes = 300
reward_history = []
avg_reward_history = []
epsilon_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    steps = 0

    for step in range(500):
        # Select action
        action = agent.act(state, training=True)

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        # Store transition
        agent.remember(state, action, reward, next_state, done)

        # Train agent
        agent.replay()

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    # Update target network
    if episode % agent.target_update_freq == 0:
        agent.update_target_model()

    # Decay epsilon
    agent.decay_epsilon()

    # Metrics
    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-100:])
    avg_reward_history.append(avg_reward)
    epsilon_history.append(agent.epsilon)

    # Print progress
    if (episode + 1) % 20 == 0:
        print(f"Episódio {episode+1:3d} | Reward: {total_reward:3.0f} | "
              f"Avg (100): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.4f} | "
              f"Steps: {steps:3d}")

    # Check if solved
    if avg_reward >= 475:
        print(f"\n🎉 RESOLVIDO no episódio {episode+1}! Avg Reward = {avg_reward:.2f}")
        break

print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO!")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 4. ANÁLISE E COMPARAÇÃO
# ═══════════════════════════════════════════════════════════════════

print(f"\n📊 Estatísticas Finais:")
print(f"   Total de Episódios: {len(reward_history)}")
print(f"   Melhor Reward: {max(reward_history):.0f}")
print(f"   Reward Médio (últimos 100): {np.mean(reward_history[-100:]):.2f}")
print(f"   Epsilon Final: {agent.epsilon:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Reward por episódio
ax1 = axes[0, 0]
ax1.plot(reward_history, alpha=0.3, label='Reward', color='blue')
ax1.plot(avg_reward_history, linewidth=2, label='Avg (100 ep)', color='red')
ax1.axhline(y=475, color='green', linestyle='--', label='Target (475)')
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Total Reward')
ax1.set_title('Aprendizado - Reward por Episódio')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Epsilon decay
ax2 = axes[0, 1]
ax2.plot(epsilon_history, linewidth=2, color='orange')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Epsilon')
ax2.set_title('Epsilon Decay (Exploration)')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 3. Distribuição de rewards
ax3 = axes[1, 0]
ax3.hist(reward_history, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax3.axvline(x=np.mean(reward_history), color='red', linestyle='--', 
            label=f'Média: {np.mean(reward_history):.1f}')
ax3.axvline(x=np.median(reward_history), color='green', linestyle='--', 
            label=f'Mediana: {np.median(reward_history):.1f}')
ax3.set_xlabel('Total Reward')
ax3.set_ylabel('Frequência')
ax3.set_title('Distribuição de Rewards')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Rolling statistics
ax4 = axes[1, 1]
rolling_mean = np.array([np.mean(reward_history[max(0, i-20):i+1]) 
                         for i in range(len(reward_history))])
rolling_std = np.array([np.std(reward_history[max(0, i-20):i+1]) 
                        for i in range(len(reward_history))])
ax4.plot(rolling_mean, linewidth=2, label='Rolling Mean (20)', color='blue')
ax4.fill_between(range(len(rolling_mean)), 
                 rolling_mean - rolling_std, 
                 rolling_mean + rolling_std, 
                 alpha=0.3, color='blue', label='± 1 Std Dev')
ax4.set_xlabel('Episódio')
ax4.set_ylabel('Reward')
ax4.set_title('Rolling Mean e Desvio Padrão (Janela 20)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. TESTAR AGENTE TREINADO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTANDO AGENTE TREINADO (50 episódios, sem exploração)")
print("="*70)

test_rewards = []
for test_ep in range(50):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(500):
        action = agent.act(state, training=False)  # Greedy (ε=0)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        total_reward += reward

        if done:
            break

    test_rewards.append(total_reward)

test_avg = np.mean(test_rewards)
test_std = np.std(test_rewards)
test_min = min(test_rewards)
test_max = max(test_rewards)

print(f"\n📊 Resultados do Teste:")
print(f"   Média: {test_avg:.2f}")
print(f"   Desvio Padrão: {test_std:.2f}")
print(f"   Mínimo: {test_min:.0f}")
print(f"   Máximo: {test_max:.0f}")
print(f"   Taxa de Sucesso (≥475): {sum(r >= 475 for r in test_rewards)/50*100:.1f}%")

print("\n💡 Vantagens Double Dueling DQN:")
print("   ✅ Double DQN: Reduz overestimation de Q-values (mais estável)")
print("   ✅ Dueling: Separa V(s) e A(s,a) - aprende valor de estados independentemente")
print("   ✅ Convergência mais rápida que DQN vanilla")
print("   ✅ Menos oscilações no treinamento")
print("   ✅ Resolve CartPole em ~150-250 episódios (DQN vanilla ~300-400)")

env.close()
