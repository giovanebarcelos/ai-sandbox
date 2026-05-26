#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1821-35ResumoEProximaAula
Aula 18 - Reinforcement Learning
Curso: Inteligência Artificial - FAPA

FROZEN LAKE - Q-LEARNING EM AMBIENTE ESTOCÁSTICO
=================================================
Cenário: Robô navega por lago congelado (4×4 grid)
  S F F F
  F H F H
  F F F H
  H F F G

Legenda: S=Start, F=Frozen (seguro), H=Hole (buraco!), G=Goal (+1 recompensa)
Desafio: O ambiente é SLIPPERY (escorregadio): ao escolher ir à direita,
          há 33% de chance de escorregar para cima ou para baixo.

Isso demonstra Q-Learning em MDPs ESTOCÁSTICOS - muito mais próximo de problemas reais!

Instalação: pip install gymnasium matplotlib seaborn numpy
"""

import numpy as np
import gymnasium as gym   # gymnasium é a versão moderna do OpenAI gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# 1. AMBIENTE E HIPERPARÂMETROS
# ═══════════════════════════════════════════════════════════════════

# Criar ambiente com gymnasium (versão moderna do gym)
# is_slippery=True: a ação tomada pode resultar em movimento perpendicular (33%)
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode=None)
n_states = env.observation_space.n  # 16
n_actions = env.action_space.n      # 4

print("=" * 70)
print("FROZEN LAKE - Q-LEARNING COM AMBIENTE ESTOCÁSTICO")
print("=" * 70)
print(f"Estados: {n_states}  (16 posições no grid 4×4)")
print(f"Ações: {n_actions}   (0=Esquerda, 1=Baixo, 2=Direita, 3=Cima)")
print(f"Slippery: TRUE  (ação escolhida pode escorregar para lado)")
print(f"\nMapa 4×4:")
print("""
    S F F F    ← Linha 0: Start (estado 0..3)
    F H F H    ← Linha 1: Holes em estados 5 e 7
    F F F H    ← Linha 2: Hole em estado 11
    H F F G    ← Linha 3: Hole em 12, Goal em 15
""")

# Hiperparâmetros do Q-Learning
# alpha alto (0.8): aprende rápido mesmo com estocasticidade alta
# gamma (0.95): valoriza bem recompensas futuras (goal pode estar longe)
# epsilon alto e decay lento: muita exploração necessária no lago escorregadio
alpha = 0.8           # Learning rate: quão rápido atualiza os Q-values
gamma = 0.95          # Discount factor: importância das recompensas futuras
epsilon_start = 1.0   # Começa com 100% de exploração aleatória
epsilon_min = 0.01    # Garante no mínimo 1% de exploração sempre
epsilon_decay = 0.9995  # Decai lentamente: precisa explorar bastante
num_episodes = 15000  # Precisa de muitos episódios (ambiente difícil!)

# ═══════════════════════════════════════════════════════════════════
# 2. Q-LEARNING ALGORITHM
# ═══════════════════════════════════════════════════════════════════

def q_learning_frozen_lake(env, num_episodes, alpha, gamma, epsilon_start,
                           epsilon_min, epsilon_decay):
    """
    Q-Learning com decay exponencial de epsilon para FrozenLake.

    A estocasticidade do ambiente torna isso mais difícil que Grid World:
    - Mesma ação pode levar a estados diferentes
    - Q-values aprendem a MÉDIA das recompensas esperadas
    - Precisa de muito mais episódios para convergir
    """

    # Q-table: zeros iniciais → agente não sabe nada ainda
    # Shape: (16 estados, 4 ações) = 64 valores Q
    Q = np.zeros((n_states, n_actions))

    rewards_history = []   # Recompensa total por episódio (0 ou 1)
    success_history = []   # 1 se chegou ao Goal, 0 se caiu no buraco
    epsilon_history = []   # Evolução do epsilon ao longo do treino
    steps_history = []     # Quantos steps até terminar o episódio

    epsilon = epsilon_start

    for episode in range(num_episodes):
        # gymnasium: reset() retorna (state, info)
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            # ε-greedy: explora ou explota com base no epsilon atual
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Exploração aleatória
            else:
                action = np.argmax(Q[state])         # Exploitação da política atual

            # Passo no ambiente: gymnasium retorna 5 valores
            next_state, reward, done, truncated, info = env.step(action)

            # Q-Learning update (regra de Bellman):
            # Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
            # O termo entre colchetes é o "TD Error" (erro de diferença temporal)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

            # Evitar loops infinitos em casos patológicos
            if steps > 100:
                break

        # Decair epsilon gradualmente (mais exploração no início, menos no final)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(total_reward)
        # Recompensa +1 significa que chegou ao Goal
        success_history.append(1 if total_reward > 0 else 0)
        epsilon_history.append(epsilon)
        steps_history.append(steps)

        # Reportar progresso a cada 2000 episódios
        if (episode + 1) % 2000 == 0:
            recent_success = np.mean(success_history[-500:]) * 100
            recent_steps = np.mean(steps_history[-500:])
            print(f"Episódio {episode+1:5d} | Epsilon: {epsilon:.4f} | "
                  f"Taxa Sucesso (últimos 500): {recent_success:5.1f}% | "
                  f"Steps Médios: {recent_steps:.1f}")

    return Q, rewards_history, success_history, epsilon_history, steps_history

# ═══════════════════════════════════════════════════════════════════
# 3. TREINAR AGENTE
# ═══════════════════════════════════════════════════════════════════

Q, rewards_history, success_history, epsilon_history, steps_history = \
    q_learning_frozen_lake(env, num_episodes, alpha, gamma, 
                           epsilon_start, epsilon_min, epsilon_decay)

print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO!")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 4. ANÁLISE DOS RESULTADOS
# ═══════════════════════════════════════════════════════════════════

# Taxa de sucesso final
final_success_rate = np.mean(success_history[-1000:]) * 100
print(f"\n🎯 Taxa de Sucesso Final (últimos 1000 ep): {final_success_rate:.1f}%")

# Política aprendida
policy = np.argmax(Q, axis=1)
action_names = ['←', '↓', '→', '↑']
print("\n📋 Política Aprendida (melhor ação por estado):")
print("Grid 4×4:")
for row in range(4):
    row_str = "  "
    for col in range(4):
        state = row * 4 + col
        if state == 15:  # Goal
            row_str += "G  "
        elif state in [5, 7, 11, 12]:  # Holes
            row_str += "H  "
        else:
            row_str += f"{action_names[policy[state]]}  "
    print(row_str)

# Q-values summary
print("\n💎 Q-values Médios por Estado (Top 5):")
q_means = Q.mean(axis=1)
top_states = np.argsort(q_means)[-5:][::-1]
for state in top_states:
    print(f"  Estado {state:2d}: Q_mean = {q_means[state]:.4f} | "
          f"Melhor ação: {action_names[policy[state]]}")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Taxa de sucesso ao longo do tempo
ax1 = axes[0, 0]
window = 500
success_smooth = np.convolve(success_history, np.ones(window)/window, mode='valid')
ax1.plot(success_smooth, linewidth=2, color='green')
ax1.axhline(y=0.7, color='red', linestyle='--', label='Target 70%')
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Taxa de Sucesso')
ax1.set_title(f'Aprendizado - Taxa de Sucesso (Janela {window})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Epsilon decay
ax2 = axes[0, 1]
ax2.plot(epsilon_history, linewidth=1, color='orange')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Epsilon')
ax2.set_title('Epsilon-Greedy Decay')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 3. Steps por episódio
ax3 = axes[0, 2]
steps_smooth = np.convolve(steps_history, np.ones(window)/window, mode='valid')
ax3.plot(steps_smooth, linewidth=2, color='purple')
ax3.set_xlabel('Episódio')
ax3.set_ylabel('Número de Steps')
ax3.set_title(f'Eficiência - Steps para Completar (Janela {window})')
ax3.grid(True, alpha=0.3)

# 4. Heatmap Q-table (valores máximos)
ax4 = axes[1, 0]
Q_max = np.max(Q, axis=1).reshape(4, 4)
sns.heatmap(Q_max, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Max Q-value'}, ax=ax4)
ax4.set_title('Max Q-values por Estado (Grid 4×4)')
ax4.set_xlabel('Coluna')
ax4.set_ylabel('Linha')

# 5. Distribuição Q-values
ax5 = axes[1, 1]
Q_flat = Q.flatten()
ax5.hist(Q_flat, bins=50, edgecolor='black', alpha=0.7)
ax5.set_xlabel('Q-value')
ax5.set_ylabel('Frequência')
ax5.set_title('Distribuição de Q-values')
ax5.axvline(x=Q_flat.mean(), color='red', linestyle='--', 
            label=f'Média: {Q_flat.mean():.3f}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Policy visualization
ax6 = axes[1, 2]
policy_grid = policy.reshape(4, 4)
policy_display = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        state = i * 4 + j
        if state in [5, 7, 11, 12]:  # Holes
            policy_display[i, j] = -1
        elif state == 15:  # Goal
            policy_display[i, j] = 4
        else:
            policy_display[i, j] = policy_grid[i, j]

cmap_custom = plt.cm.colors.ListedColormap(['black', 'blue', 'green', 'red', 'yellow', 'gold'])
im = ax6.imshow(policy_display, cmap=cmap_custom, vmin=-1, vmax=4)
ax6.set_title('Política Aprendida (Visual)')
ax6.set_xticks(range(4))
ax6.set_yticks(range(4))
for i in range(4):
    for j in range(4):
        state = i * 4 + j
        if state in [5, 7, 11, 12]:
            text = 'H'
        elif state == 15:
            text = 'G'
        else:
            text = action_names[policy[state]]
        ax6.text(j, i, text, ha='center', va='center', 
                color='white', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. TESTAR POLÍTICA APRENDIDA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTANDO POLÍTICA APRENDIDA (100 episódios)")
print("="*70)

test_successes = 0
test_episodes = 100
test_steps_list = []

for test_ep in range(test_episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < 100:
        action = np.argmax(Q[state])  # Greedy (sem exploração)
        state, reward, done, truncated, info = env.step(action)
        steps += 1

    if reward > 0:
        test_successes += 1
        test_steps_list.append(steps)

test_success_rate = test_successes / test_episodes * 100
avg_steps = np.mean(test_steps_list) if test_steps_list else 0

print(f"✅ Taxa de Sucesso: {test_success_rate:.1f}% ({test_successes}/{test_episodes})")
print(f"📊 Steps Médios (sucessos): {avg_steps:.1f}")

print("\n💡 Análise do Ambiente Estocástico:")
print("   • FrozenLake é SLIPPERY: ação escolhida pode resultar em movimento lateral")
print("   • Isso torna o ambiente não-determinístico (mesma ação → resultados diferentes)")
print("   • Q-Learning aprende Q-values ESPERADOS considerando a estocasticidade")
print(f"   • Taxa de sucesso ~{test_success_rate:.0f}% é razoável dado o desafio elevado")
print("   • Ambiente determinístico (is_slippery=False) alcança ~100% de sucesso")
print("   • Isso mostra a diferença entre RL em ambientes determinísticos vs estocásticos")
print()
print("   Comparação esperada:")
print("   is_slippery=True  → ~60-75% de sucesso após 15.000 episódios")
print("   is_slippery=False → ~100% de sucesso após 3.000 episódios")

env.close()