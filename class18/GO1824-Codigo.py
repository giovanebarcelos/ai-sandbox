# GO1824-Codigo
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 1. POLICY NETWORK (ACTOR) E VALUE NETWORK (BASELINE)
# ═══════════════════════════════════════════════════════════════════

def build_policy_network(state_size, action_size, learning_rate=0.001):
    """
    Policy network: π(a|s) - Retorna probabilidades sobre ações
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_size,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_size, activation='softmax')  # Probabilidades
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    return model, optimizer

def build_value_network(state_size, learning_rate=0.001):
    """
    Value network: V(s) - Baseline para reduzir variância
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_size,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  # Valor escalar
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

# ═══════════════════════════════════════════════════════════════════
# 2. REINFORCE AGENT COM BASELINE
# ═══════════════════════════════════════════════════════════════════

class REINFORCEAgent:
    """
    REINFORCE com baseline (Monte Carlo Policy Gradient)

    Update rule:
    ∇J(θ) = E[∇log π(a|s) * (G_t - V(s))]
    onde G_t = return e V(s) = baseline
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99

        # Networks
        self.policy_net, self.policy_optimizer = build_policy_network(
            state_size, action_size, learning_rate=0.001
        )
        self.value_net = build_value_network(state_size, learning_rate=0.001)

        print("="*70)
        print("REINFORCE AGENT COM BASELINE INICIALIZADO")
        print("="*70)
        print("\n📋 Policy Network:")
        self.policy_net.summary()
        print("\n📋 Value Network (Baseline):")
        self.value_net.summary()

    def act(self, state):
        """
        Sample ação da distribuição π(a|s)
        """
        probs = self.policy_net(state, training=False).numpy()[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs

    def train_on_episode(self, states, actions, rewards):
        """
        Treinar após episódio completo (Monte Carlo)

        1. Calcular returns G_t para cada timestep
        2. Treinar value network V(s) para predizer G_t
        3. Calcular advantages A_t = G_t - V(s)
        4. Atualizar política com gradiente ponderado por advantage
        """
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        # 1. Calcular returns (discounted cumulative rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        # Normalizar returns (estabilidade)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # 2. Treinar value network (baseline)
        self.value_net.fit(states, returns, epochs=1, verbose=0, batch_size=len(states))

        # 3. Calcular advantages
        values = self.value_net(states, training=False).numpy().flatten()
        advantages = returns.numpy() - values
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # 4. Atualizar policy network
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs = self.policy_net(states, training=True)

            # Log probabilities das ações tomadas
            indices = tf.range(len(actions)) * self.action_size + actions
            log_probs = tf.math.log(tf.gather(tf.reshape(action_probs, [-1]), indices) + 1e-8)

            # Policy gradient loss: -E[log π(a|s) * A(s,a)]
            # Negativo porque queremos gradient ASCENT (maximizar J)
            policy_loss = -tf.reduce_mean(log_probs * advantages)

        # Calcular gradientes e atualizar
        grads = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        return policy_loss.numpy()

# ═══════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("INICIANDO TREINAMENTO - LUNARLANDER COM REINFORCE")
print("="*70)

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]  # 8
action_size = env.action_space.n  # 4

agent = REINFORCEAgent(state_size, action_size)

num_episodes = 1000
reward_history = []
avg_reward_history = []
loss_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    # Coletar episódio completo
    states_episode = []
    actions_episode = []
    rewards_episode = []

    done = False
    truncated = False
    total_reward = 0
    steps = 0

    while not (done or truncated) and steps < 1000:
        # Sample ação
        action, probs = agent.act(state)

        # Executar
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Armazenar
        states_episode.append(state[0])
        actions_episode.append(action)
        rewards_episode.append(reward)

        state = next_state
        total_reward += reward
        steps += 1

    # Treinar após episódio completo (REINFORCE = Monte Carlo)
    loss = agent.train_on_episode(states_episode, actions_episode, rewards_episode)

    # Metrics
    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-100:])
    avg_reward_history.append(avg_reward)
    loss_history.append(loss)

    # Print progress
    if (episode + 1) % 50 == 0:
        print(f"Episódio {episode+1:4d} | Reward: {total_reward:7.2f} | "
              f"Avg (100): {avg_reward:7.2f} | Loss: {loss:.4f} | Steps: {steps:3d}")

    # Check if solved
    if avg_reward >= 200:
        print(f"\n🎉 RESOLVIDO no episódio {episode+1}! Avg Reward = {avg_reward:.2f}")
        break

print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO!")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 4. ANÁLISE DOS RESULTADOS
# ═══════════════════════════════════════════════════════════════════

print(f"\n📊 Estatísticas Finais:")
print(f"   Total de Episódios: {len(reward_history)}")
print(f"   Melhor Reward: {max(reward_history):.2f}")
print(f"   Pior Reward: {min(reward_history):.2f}")
print(f"   Reward Médio (últimos 100): {np.mean(reward_history[-100:]):.2f}")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Reward progression
ax1 = axes[0, 0]
ax1.plot(reward_history, alpha=0.3, label='Reward', color='blue')
ax1.plot(avg_reward_history, linewidth=2, label='Avg (100 ep)', color='red')
ax1.axhline(y=200, color='green', linestyle='--', label='Target (200)')
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Total Reward')
ax1.set_title('Aprendizado - Reward por Episódio')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Policy loss
ax2 = axes[0, 1]
ax2.plot(loss_history, linewidth=1, color='purple', alpha=0.7)
window = 50
if len(loss_history) > window:
    loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    ax2.plot(range(window-1, len(loss_history)), loss_smooth, 
             linewidth=2, color='red', label=f'Smooth ({window})')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Policy Loss')
ax2.set_title('Policy Gradient Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Distribuição de rewards
ax3 = axes[1, 0]
ax3.hist(reward_history, bins=40, edgecolor='black', alpha=0.7, color='skyblue')
ax3.axvline(x=np.mean(reward_history), color='red', linestyle='--', 
            label=f'Média: {np.mean(reward_history):.1f}')
ax3.axvline(x=200, color='green', linestyle='--', label='Target: 200')
ax3.set_xlabel('Total Reward')
ax3.set_ylabel('Frequência')
ax3.set_title('Distribuição de Rewards')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Learning curve with confidence interval
ax4 = axes[1, 1]
rolling_mean = np.array([np.mean(reward_history[max(0, i-50):i+1]) 
                         for i in range(len(reward_history))])
rolling_std = np.array([np.std(reward_history[max(0, i-50):i+1]) 
                        for i in range(len(reward_history))])
ax4.plot(rolling_mean, linewidth=2, label='Rolling Mean (50)', color='blue')
ax4.fill_between(range(len(rolling_mean)), 
                 rolling_mean - rolling_std, 
                 rolling_mean + rolling_std, 
                 alpha=0.3, color='blue', label='± 1 Std Dev')
ax4.axhline(y=200, color='green', linestyle='--', label='Target (200)')
ax4.set_xlabel('Episódio')
ax4.set_ylabel('Reward')
ax4.set_title('Learning Curve com Confidence Interval')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. TESTAR AGENTE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTANDO AGENTE TREINADO (30 episódios)")
print("="*70)

test_rewards = []
for test_ep in range(30):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    steps = 0

    done = False
    truncated = False

    while not (done or truncated) and steps < 1000:
        action, _ = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        total_reward += reward
        steps += 1

    test_rewards.append(total_reward)

test_avg = np.mean(test_rewards)
test_std = np.std(test_rewards)

print(f"\n📊 Resultados do Teste:")
print(f"   Média: {test_avg:.2f}")
print(f"   Desvio Padrão: {test_std:.2f}")
print(f"   Mínimo: {min(test_rewards):.2f}")
print(f"   Máximo: {max(test_rewards):.2f}")
print(f"   Taxa de Sucesso (≥200): {sum(r >= 200 for r in test_rewards)/30*100:.1f}%")

print("\n💡 Vantagens REINFORCE com Baseline:")
print("   ✅ Aprende política π(a|s) diretamente (não precisa Q-values)")
print("   ✅ Funciona com espaços de ação discretos E contínuos")
print("   ✅ Baseline V(s) reduz variância significativamente")
print("   ✅ Mais estável que REINFORCE vanilla")
print("   ✅ On-policy: aprende da política atual")
print("\n💡 Desvantagens:")
print("   ❌ Monte Carlo: precisa episódio completo (não funciona online)")
print("   ❌ Alta variância (mesmo com baseline)")
print("   ❌ Sample inefficient (precisa muitos episódios)")
print("   ❌ Sensível a hiperparâmetros (learning rate)")

env.close()
