#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1817-26ProjetoDqnCartpoleSetup
Aula 18 - Reinforcement Learning
Curso: Inteligência Artificial - FAPA

DQN (DEEP Q-NETWORK) NO CARTPOLE
==================================
Cenário: Equilibrar uma vara vertical em cima de um carrinho móvel.
         Se a vara inclinar demais (>15°) ou o carrinho sair da trilha, o episódio acaba.

Estado: 4 valores contínuos
  [posição_carro, velocidade_carro, ângulo_vara, velocidade_angular_vara]

Ações: 2 ações discretas
  0 = Empurrar esquerda
  1 = Empurrar direita

Recompensa: +1 a cada passo que a vara permanece equilibrada.
Objetivo: Média ≥ 195 passos em 100 episódios consecutivos = RESOLVIDO!

Por que DQN em vez de Q-Learning?
  Q-Learning guarda uma tabela Q[estado, ação]. Com estados CONTÍNUOS
  (posição pode ser qualquer valor real), a tabela seria infinita.
  DQN usa uma REDE NEURAL para aproximar Q(s,a) → funciona para estados contínuos!

Inovações do DQN (DeepMind, 2015):
  1. Experience Replay: reutiliza memórias passadas para estabilizar treino
  2. Target Network: segunda rede para calcular targets → evita "alvo móvel"

Instalação: pip install gymnasium tensorflow keras numpy matplotlib
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from collections import deque
import random

# ═══════════════════════════════════════════════════════════════════
# 1. HIPERPARÂMETROS
# ═══════════════════════════════════════════════════════════════════

EPISODES = 500              # Número máximo de episódios de treinamento
BATCH_SIZE = 32             # Amostras por mini-batch do replay buffer
GAMMA = 0.99                # Discount factor: valorizar recompensas futuras
EPSILON_START = 1.0         # Começa explorando 100% aleatoriamente
EPSILON_MIN = 0.01          # Nunca explorar menos que 1%
EPSILON_DECAY = 0.995       # Multiplicar epsilon a cada episódio
LEARNING_RATE = 0.001       # Taxa de aprendizado do Adam optimizer
TARGET_UPDATE_FREQ = 10     # Atualizar target network a cada N episódios
REPLAY_BUFFER_SIZE = 10000  # Capacidade do Experience Replay Buffer


# ═══════════════════════════════════════════════════════════════════
# 2. AGENTE DQN
# ═══════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Agente DQN com Experience Replay e Target Network.

    Estrutura:
      - self.model:        Q-network principal (atualizada a cada step)
      - self.target_model: Target network (atualizada a cada TARGET_UPDATE_FREQ ep)
      - self.replay_buffer: Deque com experiências (s, a, r, s', done)
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Experience Replay Buffer: FIFO de capacidade REPLAY_BUFFER_SIZE
        # deque com maxlen descarta automaticamente experiências antigas
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Epsilon inicia em 1.0 (exploração total) e decai com o tempo
        self.epsilon = EPSILON_START

        # Criar as duas redes neurais (Q-network e Target network)
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Target começa com os mesmos pesos da Q-network
        self.update_target_model()

    def _build_model(self):
        """
        Rede neural: recebe estado (4 valores) → prediz Q-values para cada ação.

        Arquitetura simples mas eficaz:
          Entrada (4) → Dense(64, ReLU) → Dense(64, ReLU) → Saída (2 Q-values)

        Ativação linear na saída: Q-values podem ser qualquer valor real (sem bounds).
        Loss MSE: erro quadrático entre Q-predito e Q-target (regressão).
        """
        model = keras.Sequential([
            # Camada 1: extrai features dos estados
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            # Camada 2: combina features para inferir valores de ação
            layers.Dense(64, activation='relu'),
            # Saída: Q-value para cada ação (linear = sem restrição de valores)
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse'  # Mean Squared Error: (Q_pred - Q_target)²
        )
        return model

    def update_target_model(self):
        """Copia pesos da Q-network para a Target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Armazena uma transição (s, a, r, s', done) no replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Seleciona ação usando política ε-greedy:
          - Com probabilidade ε: ação ALEATÓRIA (exploração)
          - Com probabilidade 1-ε: ação com maior Q-value (exploitação)
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)  # Exploração
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])                  # Exploitação

    def replay(self):
        """
        Treina a rede neural com um mini-batch aleatório do replay buffer.

        Por que aleatório?
          Amostrar aleatoriamente quebra correlação temporal entre experiências
          consecutivas, tornando o treinamento mais estável.

        Algoritmo DQN update:
          Para cada (s, a, r, s', done) no batch:
            Se done: Q_target = r  (sem recompensas futuras)
            Senão:   Q_target = r + γ * max_a' Q_target(s', a')
                     (Target network calcula o Q do próximo estado)
        """
        # Só treina quando o buffer tem amostras suficientes
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Amostrar mini-batch aleatório
        batch = random.sample(self.replay_buffer, BATCH_SIZE)

        # Desempacotar o batch em arrays numpy (para operações vetorizadas)
        states = np.vstack([x[0] for x in batch])        # (32, 4)
        actions = np.array([x[1] for x in batch])        # (32,)
        rewards = np.array([x[2] for x in batch])        # (32,)
        next_states = np.vstack([x[3] for x in batch])   # (32, 4)
        dones = np.array([x[4] for x in batch])          # (32,) - bool

        # Predições atuais do Q-network (usadas como base para atualizar apenas a ação tomada)
        targets = self.model.predict(states, verbose=0)  # (32, 2)

        # Q-values do próximo estado usando a TARGET network (mais estável)
        q_next = self.target_model.predict(next_states, verbose=0)  # (32, 2)

        # Atualizar apenas o Q-value da ação tomada (os outros ficam iguais)
        for i in range(BATCH_SIZE):
            if dones[i]:
                # Episódio terminou: sem recompensas futuras
                targets[i][actions[i]] = rewards[i]
            else:
                # Bellman: Q_target = r + γ * max(Q_target(s'))
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(q_next[i])

        # Treinar a rede para se aproximar dos Q-targets calculados
        self.model.fit(states, targets, epochs=1, verbose=0)

    def decay_epsilon(self):
        """Reduz epsilon gradualmente até o mínimo definido."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ═══════════════════════════════════════════════════════════════════
# 3. LOOP DE TREINAMENTO
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("DQN - CARTPOLE-V1")
print("=" * 70)
print(f"Episódios máximos: {EPISODES}")
print(f"Critério de sucesso: média ≥ 195 em 100 episódios")
print(f"Hiperparâmetros: γ={GAMMA}, α={LEARNING_RATE}, ε_decay={EPSILON_DECAY}")
print()

# Criar ambiente CartPole (gymnasium)
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # 4 (pos, vel, ângulo, vel_ang)
action_size = env.action_space.n              # 2 (esq, dir)

print(f"Estado: {state_size} dimensões | Ações: {action_size}")

# Instanciar agente DQN
agent = DQNAgent(state_size, action_size)
agent.model.summary()

# Histórico para visualização
reward_history = []        # Recompensa de cada episódio
avg_reward_history = []    # Média móvel de 100 episódios
epsilon_history = []       # Evolução do epsilon

print("\n" + "-" * 70)
print("INICIANDO TREINAMENTO...")
print("-" * 70)

for episode in range(EPISODES):
    # gymnasium: reset() retorna (observation, info)
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])  # Formato para a rede: (1, 4)
    total_reward = 0

    for step in range(500):  # CartPole para no máximo 500 steps
        # 1. Selecionar ação (ε-greedy)
        action = agent.act(state)

        # 2. Executar ação no ambiente
        # gymnasium: step() retorna (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. Preparar próximo estado
        next_state = np.reshape(next_state, [1, state_size])

        # 4. Armazenar transição no replay buffer
        agent.remember(state, action, reward, next_state, done)

        # 5. Treinar com mini-batch do replay buffer
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            break  # Vara caiu ou carro saiu da trilha

    # Armazenar métricas do episódio
    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-100:])
    avg_reward_history.append(avg_reward)
    epsilon_history.append(agent.epsilon)

    # Decair epsilon após cada episódio
    agent.decay_epsilon()

    # Atualizar target network periodicamente (estabiliza treinamento)
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()

    # Log de progresso a cada 10 episódios
    if episode % 10 == 0:
        print(f"Ep {episode:3d}/{EPISODES} | Reward: {total_reward:3.0f} | "
              f"Avg(100): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")

    # Critério de sucesso: média ≥ 195 nos últimos 100 episódios
    if avg_reward >= 195 and len(reward_history) >= 100:
        print(f"\n🎉 PROBLEMA RESOLVIDO no episódio {episode}! Avg = {avg_reward:.2f}")
        agent.model.save('dqn_cartpole.keras')
        print("✅ Modelo salvo: dqn_cartpole.keras")
        break

env.close()

# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZAÇÃO DOS RESULTADOS DE TREINAMENTO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VISUALIZANDO CURVAS DE APRENDIZADO")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DQN - CartPole: Curvas de Aprendizado', fontsize=14, fontweight='bold')

episodes_range = range(len(reward_history))

# --- Gráfico 1: Reward por episódio com média móvel ---
ax1 = axes[0, 0]
ax1.plot(episodes_range, reward_history, alpha=0.3, color='steelblue', label='Reward/Episódio')
ax1.plot(episodes_range, avg_reward_history, color='red', linewidth=2,
         label='Média Móvel (100 ep)')
ax1.axhline(y=195, color='green', linestyle='--', linewidth=1.5, label='Critério (195)')
ax1.axhline(y=500, color='gold', linestyle=':', linewidth=1, label='Máximo (500)')
ax1.set_xlabel('Episódio')
ax1.set_ylabel('Total Reward')
ax1.set_title('Reward por Episódio\n(vermelha = média 100 eps)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Anotar episódio de solução (se resolvido)
if max(avg_reward_history) >= 195:
    solved_ep = next(i for i, v in enumerate(avg_reward_history) if v >= 195)
    ax1.axvline(x=solved_ep, color='purple', linestyle='--', alpha=0.7)
    ax1.text(solved_ep + 2, 50, f'Resolvido\n(ep {solved_ep})',
             fontsize=8, color='purple')

# --- Gráfico 2: Epsilon decay ---
ax2 = axes[0, 1]
ax2.plot(episodes_range, epsilon_history, color='orange', linewidth=2)
ax2.fill_between(episodes_range, 0, epsilon_history, alpha=0.2, color='orange')
ax2.set_xlabel('Episódio')
ax2.set_ylabel('Epsilon (ε)')
ax2.set_title('Decaimento do Epsilon\n(exploração → exploitação)')
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.text(len(reward_history) * 0.05, 0.9,
         f"ε inicial: {EPSILON_START}\nε final: {epsilon_history[-1]:.3f}",
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# --- Gráfico 3: Distribuição dos rewards ---
ax3 = axes[1, 0]
ax3.hist(reward_history, bins=30, edgecolor='black', alpha=0.7, color='#4CAF50')
ax3.axvline(x=np.mean(reward_history), color='red', linestyle='--',
            linewidth=2, label=f"Média: {np.mean(reward_history):.1f}")
ax3.axvline(x=np.median(reward_history), color='blue', linestyle='--',
            linewidth=2, label=f"Mediana: {np.median(reward_history):.1f}")
ax3.axvline(x=195, color='green', linestyle=':', linewidth=1.5, label='Meta: 195')
ax3.set_xlabel('Total Reward')
ax3.set_ylabel('Frequência')
ax3.set_title('Distribuição de Rewards\n(histograma de todos os episódios)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Gráfico 4: Fases do aprendizado (early, mid, late) ---
ax4 = axes[1, 1]
n = len(reward_history)
fase1 = reward_history[:n // 3]
fase2 = reward_history[n // 3: 2 * n // 3]
fase3 = reward_history[2 * n // 3:]
nomes_fases = ['Início\n(exploração)', 'Meio\n(transição)', 'Final\n(exploitação)']
medias = [np.mean(f) for f in [fase1, fase2, fase3]]
stds = [np.std(f) for f in [fase1, fase2, fase3]]
cores_fases = ['#F44336', '#FF9800', '#4CAF50']

bars = ax4.bar(nomes_fases, medias, yerr=stds, color=cores_fases,
               alpha=0.8, edgecolor='black', capsize=6)
ax4.axhline(y=195, color='green', linestyle='--', label='Meta: 195')
ax4.set_ylabel('Reward Médio ± Desvio Padrão')
ax4.set_title('Fases do Aprendizado DQN\n(início vs. fim do treinamento)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')
for bar, media, std in zip(bars, medias, stds):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 3,
             f'{media:.0f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('dqn_cartpole_treino.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: dqn_cartpole_treino.png")

# ═══════════════════════════════════════════════════════════════════
# 5. TESTAR O AGENTE TREINADO (sem exploração aleatória)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TESTANDO AGENTE TREINADO (50 episódios, ε=0 - sem exploração)")
print("=" * 70)

# Carregar o ambiente de teste
env_test = gym.make('CartPole-v1')

test_rewards = []
for test_ep in range(50):
    state = env_test.reset()[0]
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(500):
        # Modo de teste: sem epsilon (sempre usa a melhor ação)
        q_values = agent.model.predict(state, verbose=0)
        action = np.argmax(q_values[0])  # Greedy (sem exploração)

        next_state, reward, terminated, truncated, _ = env_test.step(action)
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        if terminated or truncated:
            break

    test_rewards.append(total_reward)

env_test.close()

test_avg = np.mean(test_rewards)
test_std = np.std(test_rewards)
test_solved = sum(r >= 475 for r in test_rewards)

print(f"\n📊 Resultados do Teste:")
print(f"   Reward Médio:     {test_avg:.2f} ± {test_std:.2f}")
print(f"   Reward Mínimo:    {min(test_rewards):.0f}")
print(f"   Reward Máximo:    {max(test_rewards):.0f}")
print(f"   Ep com ≥475 pts:  {test_solved}/50 ({test_solved/50*100:.0f}%)")

print("\n💡 Conceitos-chave do DQN:")
print("   ✅ Experience Replay: quebra correlação temporal → treino mais estável")
print("   ✅ Target Network: 'alvo fixo' → evita divergência nos Q-values")
print("   ✅ ε-greedy: exploração suficiente no início → convergência mais rápida")
print("   ✅ Rede neural: aprende Q(s,a) para estados CONTÍNUOS (impossível c/ tabela)")
print()
print("   DQN resolve CartPole em ~150-300 episódios.")
print("   Q-Learning tabular seria impossível (estados reais = infinitos).")
