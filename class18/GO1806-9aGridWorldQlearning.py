# GO1806-9aGridWorldQlearning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# 1. DEFINIR AMBIENTE GRID WORLD
# ═══════════════════════════════════════════════════════════════════

class GridWorld:
    """
    Grid World 4x4 para Q-Learning

    Layout:
    ┌───┬───┬───┬───┐
    │ S │ 1 │ 2 │ G │  S(0) = Start, G(3) = Goal (+10)
    ├───┼───┼───┼───┤
    │ 4 │ X │ 6 │ 7 │  X(5) = Obstáculo (-5)
    ├───┼───┼───┼───┤
    │ 8 │ 9 │ X │11 │  X(10) = Obstáculo (-5)
    ├───┼───┼───┼───┤
    │12 │13 │14 │15 │  Cada passo: -0.1
    └───┴───┴───┴───┘

    Ações: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """

    def __init__(self):
        self.grid_size = 4
        self.n_states = self.grid_size * self.grid_size  # 16 estados
        self.n_actions = 4  # UP, DOWN, LEFT, RIGHT

        # Posições especiais
        self.start_state = 0
        self.goal_state = 3
        self.obstacles = [5, 10]

        # Recompensas
        self.reward_goal = +10.0
        self.reward_obstacle = -5.0
        self.reward_step = -0.1

        self.current_state = self.start_state

    def reset(self):
        """Reiniciar episódio"""
        self.current_state = self.start_state
        return self.current_state

    def _state_to_position(self, state):
        """Converter estado (0-15) para posição (row, col)"""
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col

    def _position_to_state(self, row, col):
        """Converter posição para estado"""
        return row * self.grid_size + col

    def step(self, action):
        """
        Executar ação e retornar (next_state, reward, done)

        Ações: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        row, col = self._state_to_position(self.current_state)

        # Aplicar ação
        if action == 0:  # UP
            row = max(0, row - 1)
        elif action == 1:  # DOWN
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # LEFT
            col = max(0, col - 1)
        elif action == 3:  # RIGHT
            col = min(self.grid_size - 1, col + 1)

        next_state = self._position_to_state(row, col)

        # Calcular recompensa e status
        if next_state == self.goal_state:
            reward = self.reward_goal
            done = True
        elif next_state in self.obstacles:
            reward = self.reward_obstacle
            done = True
        else:
            reward = self.reward_step
            done = False

        self.current_state = next_state
        return next_state, reward, done

    def render(self, Q=None, policy=None):
        """Visualizar grid world com Q-values ou política"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Desenhar grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = i * self.grid_size + j

                # Cor de fundo
                if state == self.start_state:
                    color = 'lightgreen'
                    label = 'S'
                elif state == self.goal_state:
                    color = 'gold'
                    label = 'G'
                elif state in self.obstacles:
                    color = 'lightcoral'
                    label = 'X'
                else:
                    color = 'white'
                    label = ''

                # Desenhar célula
                rect = Rectangle((j, self.grid_size - 1 - i), 1, 1, 
                                linewidth=2, edgecolor='black', 
                                facecolor=color)
                ax.add_patch(rect)

                # Adicionar label
                ax.text(j + 0.5, self.grid_size - 0.5 - i, label,
                       ha='center', va='center', fontsize=20, fontweight='bold')

                # Mostrar política aprendida (seta)
                if policy is not None and state not in [self.goal_state] + self.obstacles:
                    action = policy[state]
                    arrows = ['↑', '↓', '←', '→']
                    ax.text(j + 0.5, self.grid_size - 0.8 - i, arrows[action],
                           ha='center', va='center', fontsize=16, color='blue')

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Grid World 4x4', fontsize=16, fontweight='bold')

        return fig

# ═══════════════════════════════════════════════════════════════════
# 2. ALGORITMO Q-LEARNING
# ═══════════════════════════════════════════════════════════════════

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, 
               epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    """
    Algoritmo Q-Learning tabular

    Parâmetros:
        env: ambiente GridWorld
        num_episodes: número de episódios de treinamento
        alpha: learning rate
        gamma: discount factor
        epsilon_start: epsilon inicial para ε-greedy
        epsilon_min: epsilon mínimo
        epsilon_decay: taxa de decaimento do epsilon
    """

    # Inicializar Q-table
    Q = np.zeros((env.n_states, env.n_actions))

    epsilon = epsilon_start
    rewards_history = []
    steps_history = []

    print("="*60)
    print("TREINANDO AGENTE COM Q-LEARNING")
    print("="*60)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:  # Limite de 100 steps por episódio
            # Escolher ação (ε-greedy)
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)  # Exploração
            else:
                action = np.argmax(Q[state])  # Exploitação

            # Executar ação
            next_state, reward, done = env.step(action)

            # Atualizar Q-value (TD Update)
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        # Decair epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(total_reward)
        steps_history.append(steps)

        # Log progresso
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            print(f"Episódio {episode+1:4d}/{num_episodes} | "
                  f"Recompensa Média: {avg_reward:6.2f} | "
                  f"Steps Médios: {avg_steps:5.1f} | "
                  f"Epsilon: {epsilon:.3f}")

    return Q, rewards_history, steps_history

# ═══════════════════════════════════════════════════════════════════
# 3. EXTRAIR POLÍTICA ÓTIMA
# ═══════════════════════════════════════════════════════════════════

def extract_policy(Q):
    """Extrair política gulosa da Q-table"""
    policy = np.argmax(Q, axis=1)
    return policy

# ═══════════════════════════════════════════════════════════════════
# 4. EXECUTAR TREINAMENTO
# ═══════════════════════════════════════════════════════════════════

env = GridWorld()

# Treinar agente
Q, rewards_history, steps_history = q_learning(
    env, 
    num_episodes=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
)

# Extrair política
policy = extract_policy(Q)

print("\n" + "="*60)
print("TREINAMENTO CONCLUÍDO!")
print("="*60)

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════

# Q-Table
print("\nQ-Table Final (primeiras 5 linhas):")
print("Estado | Q(UP)   Q(DOWN)  Q(LEFT)  Q(RIGHT)")
print("-" * 50)
for state in range(min(5, env.n_states)):
    print(f"{state:3d}    | {Q[state,0]:7.2f} {Q[state,1]:7.2f} "
          f"{Q[state,2]:7.2f} {Q[state,3]:7.2f}")

# Política
actions_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
print("\nPolítica Aprendida:")
for state in range(env.n_states):
    if state not in [env.goal_state] + env.obstacles:
        row, col = env._state_to_position(state)
        print(f"Estado {state:2d} (Posição [{row},{col}]): {actions_names[policy[state]]}")

# Plotar curvas de aprendizado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Recompensa por episódio
ax1 = axes[0]
ax1.plot(rewards_history, alpha=0.3, label='Recompensa')
window = 50
rewards_smooth = np.convolve(rewards_history, 
                             np.ones(window)/window, mode='valid')
ax1.plot(range(window-1, len(rewards_history)), rewards_smooth, 
         linewidth=2, label=f'Média Móvel ({window})')
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Recompensa Total')
ax1.set_title('Aprendizado - Recompensa por Episódio')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Steps por episódio
ax2 = axes[1]
ax2.plot(steps_history, alpha=0.3, label='Steps')
steps_smooth = np.convolve(steps_history, 
                           np.ones(window)/window, mode='valid')
ax2.plot(range(window-1, len(steps_history)), steps_smooth, 
         linewidth=2, label=f'Média Móvel ({window})')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Número de Steps')
ax2.set_title('Eficiência - Steps para Completar')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualizar política no grid
fig = env.render(Q=Q, policy=policy)
plt.show()

# Heatmap dos Q-values máximos
fig, ax = plt.subplots(figsize=(8, 6))
Q_max = np.max(Q, axis=1).reshape(env.grid_size, env.grid_size)
sns.heatmap(Q_max, annot=True, fmt='.2f', cmap='RdYlGn', 
            cbar_kws={'label': 'Max Q-value'}, ax=ax)
ax.set_title('Heatmap - Max Q-values por Estado')
ax.set_xlabel('Coluna')
ax.set_ylabel('Linha')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. TESTAR POLÍTICA APRENDIDA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TESTANDO POLÍTICA APRENDIDA")
print("="*60)

num_tests = 10
success_count = 0

for test in range(num_tests):
    state = env.reset()
    done = False
    steps = 0
    path = [state]

    while not done and steps < 20:
        action = policy[state]  # Usar política aprendida
        state, reward, done = env.step(action)
        path.append(state)
        steps += 1

    if state == env.goal_state:
        success_count += 1
        print(f"Teste {test+1:2d}: ✅ SUCESSO em {steps} steps | Caminho: {path}")
    else:
        print(f"Teste {test+1:2d}: ❌ FALHOU após {steps} steps")

success_rate = success_count / num_tests * 100
print(f"\n🎯 Taxa de Sucesso: {success_rate:.0f}% ({success_count}/{num_tests})")

print("\n💡 Observações:")
print("   • Q-Learning converge para política ótima")
print("   • Exploração (ε-greedy) permite descobrir melhores caminhos")
print("   • Q-values refletem recompensa esperada de cada ação")
print("   • Política aprendida escolhe caminho mais curto ao objetivo")
