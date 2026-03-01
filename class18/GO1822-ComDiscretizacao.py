# GO1822-ComDiscretização
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# 1. AMBIENTE E DISCRETIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

env = gym.make('MountainCar-v0')

print("="*70)
print("MOUNTAIN CAR - Q-LEARNING COM DISCRETIZAÇÃO")
print("="*70)
print(f"Espaço de estados original: {env.observation_space}")
print(f"Espaço de ações: {env.action_space}")

# Discretizar espaço contínuo
n_bins_position = 20
n_bins_velocity = 20
n_actions = env.action_space.n  # 3

# Criar bins
position_bins = np.linspace(-1.2, 0.6, n_bins_position)
velocity_bins = np.linspace(-0.07, 0.07, n_bins_velocity)

def discretize_state(state):
    """Converter estado contínuo [posição, velocidade] em índice discreto"""
    position, velocity = state
    pos_idx = np.digitize(position, position_bins) - 1
    vel_idx = np.digitize(velocity, velocity_bins) - 1

    # Garantir limites
    pos_idx = np.clip(pos_idx, 0, n_bins_position - 1)
    vel_idx = np.clip(vel_idx, 0, n_bins_velocity - 1)

    # Estado discreto único
    discrete_state = pos_idx * n_bins_velocity + vel_idx
    return discrete_state

n_discrete_states = n_bins_position * n_bins_velocity
print(f"\nEstados discretizados: {n_discrete_states}")
print(f"Bins posição: {n_bins_position}, Bins velocidade: {n_bins_velocity}")

# ═══════════════════════════════════════════════════════════════════
# 2. Q-LEARNING PARA MOUNTAIN CAR
# ═══════════════════════════════════════════════════════════════════

# Hiperparâmetros
alpha = 0.1
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
num_episodes = 5000

def q_learning_mountain_car(env, num_episodes, alpha, gamma, epsilon_start, 
                            epsilon_min, epsilon_decay):
    """Q-Learning com discretização de estados"""

    # Inicializar Q-table
    Q = np.zeros((n_discrete_states, n_actions))

    # Métricas
    rewards_history = []
    steps_history = []
    success_history = []
    epsilon_history = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state)
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[discrete_state])

            # Executar ação
            next_state, reward, done, truncated, info = env.step(action)
            next_discrete_state = discretize_state(next_state)

            # Recompensa modificada (acelerar aprendizado)
            # Recompensar progresso na posição
            position_reward = next_state[0] - state[0]
            modified_reward = reward + position_reward

            # Q-Learning update
            best_next_action = np.argmax(Q[next_discrete_state])
            td_target = modified_reward + gamma * Q[next_discrete_state, best_next_action]
            td_error = td_target - Q[discrete_state, action]
            Q[discrete_state, action] += alpha * td_error

            discrete_state = next_discrete_state
            state = next_state
            total_reward += reward
            steps += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Métricas
        rewards_history.append(total_reward)
        steps_history.append(steps)
        success = 1 if state[0] >= 0.5 else 0
        success_history.append(success)
        epsilon_history.append(epsilon)

        # Print progresso
        if (episode + 1) % 500 == 0:
            recent_steps = np.mean(steps_history[-100:])
            recent_success = np.mean(success_history[-100:]) * 100
            print(f"Episódio {episode+1:4d} | Epsilon: {epsilon:.4f} | "
                  f"Steps Médios (últimos 100): {recent_steps:6.1f} | "
                  f"Taxa Sucesso: {recent_success:5.1f}%")

    return Q, rewards_history, steps_history, success_history, epsilon_history

# ═══════════════════════════════════════════════════════════════════
# 3. TREINAR AGENTE
# ═══════════════════════════════════════════════════════════════════

Q, rewards_history, steps_history, success_history, epsilon_history = \
    q_learning_mountain_car(env, num_episodes, alpha, gamma, 
                            epsilon_start, epsilon_min, epsilon_decay)

print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO!")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 4. ANÁLISE DOS RESULTADOS
# ═══════════════════════════════════════════════════════════════════

final_steps = np.mean(steps_history[-100:])
final_success = np.mean(success_history[-100:]) * 100

print(f"\n📊 Desempenho Final (últimos 100 episódios):")
print(f"   Steps Médios: {final_steps:.1f}")
print(f"   Taxa de Sucesso: {final_success:.1f}%")
print(f"   Melhor episódio: {min(steps_history)} steps")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Steps ao longo do treinamento
ax1 = axes[0, 0]
window = 100
steps_smooth = np.convolve(steps_history, np.ones(window)/window, mode='valid')
ax1.plot(steps_smooth, linewidth=2, color='blue')
ax1.axhline(y=200, color='green', linestyle='--', label='Target <200 steps')
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Steps para Completar')
ax1.set_title(f'Aprendizado - Eficiência (Janela {window})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Epsilon decay
ax2 = axes[0, 1]
ax2.plot(epsilon_history, linewidth=1, color='orange')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Epsilon')
ax2.set_title('Epsilon Decay')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 3. Taxa de sucesso
ax3 = axes[0, 2]
success_smooth = np.convolve(success_history, np.ones(window)/window, mode='valid')
ax3.plot(success_smooth, linewidth=2, color='green')
ax3.set_xlabel('Episódio')
ax3.set_ylabel('Taxa de Sucesso')
ax3.set_title(f'Taxa de Sucesso (Janela {window})')
ax3.set_ylim([0, 1.05])
ax3.grid(True, alpha=0.3)

# 4. Heatmap Q-values (posição vs velocidade) - Ação Push Right
ax4 = axes[1, 0]
Q_grid_right = Q[:, 2].reshape(n_bins_position, n_bins_velocity)
sns.heatmap(Q_grid_right, cmap='RdYlGn', cbar_kws={'label': 'Q-value'}, ax=ax4)
ax4.set_title('Q-values: Ação PUSH RIGHT')
ax4.set_xlabel('Velocidade (binned)')
ax4.set_ylabel('Posição (binned)')

# 5. Heatmap Q-values - Ação Push Left
ax5 = axes[1, 1]
Q_grid_left = Q[:, 0].reshape(n_bins_position, n_bins_velocity)
sns.heatmap(Q_grid_left, cmap='RdYlGn', cbar_kws={'label': 'Q-value'}, ax=ax5)
ax5.set_title('Q-values: Ação PUSH LEFT')
ax5.set_xlabel('Velocidade (binned)')
ax5.set_ylabel('Posição (binned)')

# 6. Política aprendida (melhor ação por estado)
ax6 = axes[1, 2]
policy = np.argmax(Q, axis=1).reshape(n_bins_position, n_bins_velocity)
im = ax6.imshow(policy, cmap='viridis', aspect='auto')
ax6.set_title('Política Aprendida (0=Left, 1=None, 2=Right)')
ax6.set_xlabel('Velocidade (binned)')
ax6.set_ylabel('Posição (binned)')
cbar = plt.colorbar(im, ax=ax6, ticks=[0, 1, 2])
cbar.set_label('Ação')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. TESTAR POLÍTICA APRENDIDA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTANDO POLÍTICA APRENDIDA (50 episódios)")
print("="*70)

test_steps_list = []
test_successes = 0

for test_ep in range(50):
    state, _ = env.reset()
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < 200:
        discrete_state = discretize_state(state)
        action = np.argmax(Q[discrete_state])  # Greedy
        state, reward, done, truncated, info = env.step(action)
        steps += 1

    test_steps_list.append(steps)
    if state[0] >= 0.5:
        test_successes += 1

test_avg_steps = np.mean(test_steps_list)
test_success_rate = test_successes / 50 * 100

print(f"✅ Taxa de Sucesso: {test_success_rate:.1f}% ({test_successes}/50)")
print(f"📊 Steps Médios: {test_avg_steps:.1f}")
print(f"🏆 Melhor teste: {min(test_steps_list)} steps")

print("\n💡 Insights sobre Mountain Car:")
print("   • Problema DIFÍCIL: motor fraco, precisa momentum")
print("   • Estratégia: recuar (esquerda) para ganhar impulso, depois direita")
print("   • Discretização permite Q-Learning em espaço contínuo")
print("   • Recompensa modificada acelera aprendizado (reward shaping)")
print("   • Resolvido quando steps < 110 consistentemente")

env.close()
