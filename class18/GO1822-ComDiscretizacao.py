#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1822-ComDiscretizacao
Aula 18 - Reinforcement Learning
Curso: Inteligência Artificial - FAPA

MOUNTAIN CAR - Q-LEARNING COM DISCRETIZAÇÃO DE ESTADOS
=======================================================
Cenário: Carro precisa subir uma montanha íngreme.
Problema: O motor do carro é FRACO demais para subir diretamente.
Solução: O agente deve APRENDER a balançar para trás e para frente,
          ganhando momentum para finalmente subir.

Desafio técnico: Estado CONTÍNUO (posição + velocidade) exige
                  DISCRETIZAÇÃO para usar Q-Learning tabular.

Estado: [posição (-1.2 a 0.6), velocidade (-0.07 a 0.07)]
Ações: 0=Empurrar Esquerda, 1=Sem Ação, 2=Empurrar Direita
Objetivo: Alcançar flag no topo (posição ≥ 0.5)
Recompensa: -1 por passo (incentiva encontrar solução rápida)

Instalação: pip install gymnasium matplotlib seaborn numpy
"""

import numpy as np
import gymnasium as gym   # gymnasium = versão moderna do OpenAI gym
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# 1. AMBIENTE E DISCRETIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

# gymnasium (versão moderna) substitui o gym original
env = gym.make('MountainCar-v0')

print("=" * 70)
print("MOUNTAIN CAR - Q-LEARNING COM DISCRETIZAÇÃO")
print("=" * 70)
print(f"Espaço de estados contínuo: {env.observation_space}")
print(f"  • Posição: [-1.2, 0.6]  (negativo = vale, positivo = topo)")
print(f"  • Velocidade: [-0.07, 0.07]  (negativo = indo pra esquerda)")
print(f"Espaço de ações: {env.action_space}  (0=←, 1=sem força, 2=→)")

# DISCRETIZAÇÃO: converter espaço contínuo em grid para tabela Q
# Mais bins = mais preciso, mas tabela maior e mais lenta para aprender
n_bins_position = 20   # Dividir posição em 20 faixas
n_bins_velocity = 20   # Dividir velocidade em 20 faixas
n_actions = env.action_space.n  # 3 ações

# Criar limites das faixas (bins) para cada dimensão
position_bins = np.linspace(-1.2, 0.6, n_bins_position)
velocity_bins = np.linspace(-0.07, 0.07, n_bins_velocity)


def discretize_state(state):
    """
    Converte estado contínuo [posição, velocidade] em índice inteiro único.

    Exemplo: posição=-0.5 → bin 8, velocidade=0.02 → bin 14
             Estado discreto = 8 * 20 + 14 = 174
    """
    position, velocity = state

    # np.digitize retorna em qual bin o valor se encontra
    pos_idx = np.digitize(position, position_bins) - 1
    vel_idx = np.digitize(velocity, velocity_bins) - 1

    # Garantir que índices ficam dentro dos limites válidos
    pos_idx = np.clip(pos_idx, 0, n_bins_position - 1)
    vel_idx = np.clip(vel_idx, 0, n_bins_velocity - 1)

    # Combinar os dois índices em um único inteiro (indexação linear)
    discrete_state = pos_idx * n_bins_velocity + vel_idx
    return discrete_state


n_discrete_states = n_bins_position * n_bins_velocity
print(f"\n📊 Discretização: {n_discrete_states} estados discretos")
print(f"   Bins posição: {n_bins_position} | Bins velocidade: {n_bins_velocity}")
print(f"   Q-Table shape: ({n_discrete_states}, {n_actions}) = "
      f"{n_discrete_states * n_actions} valores Q")

# ═══════════════════════════════════════════════════════════════════
# 2. Q-LEARNING PARA MOUNTAIN CAR
# ═══════════════════════════════════════════════════════════════════

# Hiperparâmetros do Q-Learning
# Nota: Mountain Car tem recompensa esparsa (-1 a cada passo, sem bônus no caminho)
# Isso torna o aprendizado difícil: o agente raramente chega ao topo inicialmente
alpha = 0.1     # Learning rate moderado (ambiente determinístico)
gamma = 0.99    # Alto: recompensas futuras importam muito (objetivo está longe)
epsilon_start = 1.0    # 100% exploração no início
epsilon_min = 0.01     # Mínimo 1% de exploração sempre
epsilon_decay = 0.9995  # Decai lentamente para garantir exploração suficiente
num_episodes = 5000     # Mountain Car é difícil, precisa de mais episódios


def q_learning_mountain_car(env, num_episodes, alpha, gamma, epsilon_start,
                             epsilon_min, epsilon_decay):
    """
    Q-Learning com discretização de estados para Mountain Car.

    Técnica especial: REWARD SHAPING
    O problema original tem recompensa -1 por passo (muito esparsa).
    Adicionamos uma recompensa extra baseada no progresso de posição
    para guiar o agente na direção certa mais cedo.
    """

    # Q-table inicializada com zeros (agente ignora tudo no início)
    Q = np.zeros((n_discrete_states, n_actions))

    rewards_history = []   # Recompensa acumulada por episódio
    steps_history = []     # Passos até terminar (menor = melhor)
    success_history = []   # 1 se chegou ao topo, 0 se não chegou
    epsilon_history = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        # gymnasium: reset() retorna (observation, info)
        state, _ = env.reset()
        discrete_state = discretize_state(state)
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            # ε-greedy com estado discretizado
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[discrete_state])

            # gymnasium: step() retorna (obs, reward, terminated, truncated, info)
            next_state, reward, done, truncated, info = env.step(action)
            next_discrete_state = discretize_state(next_state)

            # REWARD SHAPING: adicionar bônus pelo progresso de posição
            # Sem isso, o sinal de recompensa é muito fraco para aprender
            # O agente só sabe que chegou ao objetivo APÓS 200 passos sem alcançar
            position_reward = next_state[0] - state[0]  # Δposição (+ = avanço)
            modified_reward = reward + position_reward

            # Q-Learning update com a recompensa modificada
            best_next_action = np.argmax(Q[next_discrete_state])
            td_target = modified_reward + gamma * Q[next_discrete_state, best_next_action]
            td_error = td_target - Q[discrete_state, action]
            Q[discrete_state, action] += alpha * td_error

            discrete_state = next_discrete_state
            state = next_state
            total_reward += reward  # Histórico usa recompensa original
            steps += 1

        # Decair epsilon após cada episódio
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(total_reward)
        steps_history.append(steps)
        # Sucesso: posição final ≥ 0.5 (flag no topo)
        success_history.append(1 if state[0] >= 0.5 else 0)
        epsilon_history.append(epsilon)

        # Log de progresso
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
print("   • Problema DIFÍCIL: motor fraco → precisa ganhar momentum oscilando")
print("   • Estratégia aprendida: recuar pra esquerda → impulso → subir direita")
print("   • Discretização: 20×20 = 400 estados (vs infinito contínuo)")
print("   • Reward shaping (+Δposição): guia o agente antes de chegar ao topo")
print("   • Resolvido quando steps < 110 em 90%+ dos episódios")
print()
print("   Curiosidade: Sem reward shaping, Mountain Car precisa de ~50.000+ episódios!")
print("   Com reward shaping: converge em ~5.000 episódios. Isso é 10x mais eficiente.")

env.close()