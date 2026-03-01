# GO1825-Codigo
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 1. SHARED NETWORK ARCHITECTURE (ACTOR-CRITIC)
# ═══════════════════════════════════════════════════════════════════

def build_actor_critic(state_size, action_size):
    """
    Shared base network com duas cabeças:
    - Actor head: π(a|s)
    - Critic head: V(s)
    """
    inputs = layers.Input(shape=(state_size,))

    # Shared feature extractor
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)

    # Actor head (policy)
    actor = layers.Dense(128, activation='relu')(x)
    action_probs = layers.Dense(action_size, activation='softmax', name='actor')(actor)

    # Critic head (value function)
    critic = layers.Dense(128, activation='relu')(x)
    state_value = layers.Dense(1, activation='linear', name='critic')(critic)

    model = keras.Model(inputs=inputs, outputs=[action_probs, state_value])

    return model

# ═══════════════════════════════════════════════════════════════════
# 2. A2C AGENT
# ═══════════════════════════════════════════════════════════════════

class A2CAgent:
    """
    Advantage Actor-Critic (A2C)

    Atualização a cada step com TD learning:
    - Advantage: A_t = r_t + γ*V(s_{t+1}) - V(s_t)
    - Actor loss: -log π(a|s) * A_t
    - Critic loss: MSE(V(s), r + γ*V(s'))
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # Shared Actor-Critic network
        self.model = build_actor_critic(state_size, action_size)

        # Separate optimizers para actor e critic
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_lr)

        print("="*70)
        print("A2C AGENT INICIALIZADO")
        print("="*70)
        self.model.summary()

    def act(self, state):
        """Sample ação da distribuição π(a|s)"""
        probs, value = self.model(state, training=False)
        probs = probs.numpy()[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, value.numpy()[0, 0]

    def train_step(self, state, action, reward, next_state, done):
        """
        Treinar com um único step (TD learning)

        1. Calcular TD target: r + γ*V(s')
        2. Calcular TD error (advantage): δ = TD_target - V(s)
        3. Atualizar Critic para minimizar TD error
        4. Atualizar Actor com gradiente ponderado por advantage
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        # Forward pass para next_state (sem gradientes)
        _, next_value = self.model(next_state, training=False)

        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * next_value.numpy()[0, 0]

        td_target = tf.convert_to_tensor([[td_target]], dtype=tf.float32)

        # Treinar Critic
        with tf.GradientTape() as critic_tape:
            _, value = self.model(state, training=True)
            critic_loss = tf.reduce_mean(tf.square(td_target - value))

        critic_grads = critic_tape.gradient(critic_loss, self.model.trainable_variables)
        # Aplicar gradientes apenas nas variáveis do critic (últimas camadas)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.model.trainable_variables))

        # Calcular advantage
        advantage = td_target.numpy()[0, 0] - value.numpy()[0, 0]

        # Treinar Actor
        with tf.GradientTape() as actor_tape:
            probs, _ = self.model(state, training=True)
            # Log probability da ação tomada
            action_onehot = tf.one_hot(action, self.action_size)
            log_prob = tf.math.log(tf.reduce_sum(probs * action_onehot, axis=1) + 1e-8)
            # Actor loss (gradient ASCENT, daí o negativo)
            actor_loss = -log_prob * advantage

        actor_grads = actor_tape.gradient(actor_loss, self.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy(), advantage

# ═══════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("INICIANDO TREINAMENTO - ACROBOT COM A2C")
print("="*70)

env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]  # 6
action_size = env.action_space.n  # 3

agent = A2CAgent(state_size, action_size)

num_episodes = 500
reward_history = []
avg_reward_history = []
actor_loss_history = []
critic_loss_history = []
advantage_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    steps = 0

    episode_advantages = []

    done = False
    truncated = False

    while not (done or truncated) and steps < 500:
        # Select action
        action, value = agent.act(state)

        # Execute action
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Train online (TD learning)
        actor_loss, critic_loss, advantage = agent.train_step(
            state, action, reward, next_state, done or truncated
        )

        episode_advantages.append(advantage)

        state = next_state
        total_reward += reward
        steps += 1

    # Metrics
    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-100:])
    avg_reward_history.append(avg_reward)
    actor_loss_history.append(actor_loss)
    critic_loss_history.append(critic_loss)
    advantage_history.append(np.mean(episode_advantages))

    # Print progress
    if (episode + 1) % 25 == 0:
        print(f"Episódio {episode+1:3d} | Reward: {total_reward:6.1f} | "
              f"Avg (100): {avg_reward:6.1f} | Steps: {steps:3d} | "
              f"Advantage: {np.mean(episode_advantages):6.2f}")

    # Check if solved
    if avg_reward <= -100:
        print(f"\n🎉 RESOLVIDO no episódio {episode+1}! Avg Reward = {avg_reward:.1f}")
        break

print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO!")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 4. ANÁLISE DOS RESULTADOS
# ═══════════════════════════════════════════════════════════════════

print(f"\n📊 Estatísticas Finais:")
print(f"   Total de Episódios: {len(reward_history)}")
print(f"   Melhor Reward: {max(reward_history):.1f}")
print(f"   Reward Médio (últimos 100): {np.mean(reward_history[-100:]):.1f}")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Reward progression
ax1 = axes[0, 0]
ax1.plot(reward_history, alpha=0.3, label='Reward', color='blue')
ax1.plot(avg_reward_history, linewidth=2, label='Avg (100 ep)', color='red')
ax1.axhline(y=-100, color='green', linestyle='--', label='Target (-100)')
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Total Reward')
ax1.set_title('Aprendizado - Reward por Episódio')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Actor loss
ax2 = axes[0, 1]
ax2.plot(actor_loss_history, linewidth=1, alpha=0.7, color='purple')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Actor Loss')
ax2.set_title('Actor Loss (Policy Gradient)')
ax2.grid(True, alpha=0.3)

# 3. Critic loss
ax3 = axes[0, 2]
ax3.plot(critic_loss_history, linewidth=1, alpha=0.7, color='orange')
ax3.set_xlabel('Episódio')
ax3.set_ylabel('Critic Loss')
ax3.set_title('Critic Loss (Value Function TD Error)')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 4. Advantage evolution
ax4 = axes[1, 0]
ax4.plot(advantage_history, linewidth=1, alpha=0.7, color='green')
window = 50
if len(advantage_history) > window:
    adv_smooth = np.convolve(advantage_history, np.ones(window)/window, mode='valid')
    ax4.plot(range(window-1, len(advantage_history)), adv_smooth, 
             linewidth=2, color='red', label=f'Smooth ({window})')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.set_xlabel('Episódio')
ax4.set_ylabel('Mean Advantage')
ax4.set_title('Advantage (TD Error) por Episódio')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Distribuição de rewards
ax5 = axes[1, 1]
ax5.hist(reward_history, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax5.axvline(x=np.mean(reward_history), color='red', linestyle='--', 
            label=f'Média: {np.mean(reward_history):.1f}')
ax5.axvline(x=-100, color='green', linestyle='--', label='Target: -100')
ax5.set_xlabel('Total Reward')
ax5.set_ylabel('Frequência')
ax5.set_title('Distribuição de Rewards')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Learning curve com intervalo de confiança
ax6 = axes[1, 2]
rolling_mean = np.array([np.mean(reward_history[max(0, i-30):i+1]) 
                         for i in range(len(reward_history))])
rolling_std = np.array([np.std(reward_history[max(0, i-30):i+1]) 
                        for i in range(len(reward_history))])
ax6.plot(rolling_mean, linewidth=2, label='Rolling Mean (30)', color='blue')
ax6.fill_between(range(len(rolling_mean)), 
                 rolling_mean - rolling_std, 
                 rolling_mean + rolling_std, 
                 alpha=0.3, color='blue')
ax6.axhline(y=-100, color='green', linestyle='--', label='Target (-100)')
ax6.set_xlabel('Episódio')
ax6.set_ylabel('Reward')
ax6.set_title('Learning Curve com Std Dev')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. TESTAR AGENTE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTANDO AGENTE TREINADO (50 episódios)")
print("="*70)

test_rewards = []
test_steps = []

for test_ep in range(50):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    steps = 0

    done = False
    truncated = False

    while not (done or truncated) and steps < 500:
        action, _ = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        total_reward += reward
        steps += 1

    test_rewards.append(total_reward)
    test_steps.append(steps)

test_avg_reward = np.mean(test_rewards)
test_avg_steps = np.mean(test_steps)

print(f"\n📊 Resultados do Teste:")
print(f"   Reward Médio: {test_avg_reward:.1f}")
print(f"   Steps Médios: {test_avg_steps:.1f}")
print(f"   Melhor Reward: {max(test_rewards):.1f}")
print(f"   Taxa de Sucesso (≤-100): {sum(r <= -100 for r in test_rewards)/50*100:.1f}%")

print("\n💡 Vantagens A2C:")
print("   ✅ Aprende ONLINE (TD learning, não precisa episódio completo)")
print("   ✅ Menor variância que REINFORCE (bootstrap com V(s'))")
print("   ✅ Mais sample-efficient que Monte Carlo")
print("   ✅ Actor e Critic compartilham feature extractor (eficiente)")
print("   ✅ Estável e rápido de convergir")
print("\n💡 Comparação:")
print("   • REINFORCE: Monte Carlo, alta variância, episódios completos")
print("   • A2C: TD learning, menor variância, aprende a cada step")
print("   • A3C: Versão assíncrona com múltiplos workers paralelos")

env.close()
