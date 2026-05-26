"""
GO1810 - Loop de Treinamento DQN Completo
==========================================
Demonstra o loop de treinamento DQN com simulação sem dependências pesadas.
Requer apenas numpy.

O loop de treinamento DQN integra todos os componentes:
1. ε-greedy para coleta de experiências
2. Armazenar no Replay Buffer
3. Amostrar mini-batch e calcular targets com Q_target
4. Treinar Q_network com backprop
5. Atualizar Q_target a cada C episódios
6. Decair ε após cada episódio

Ambiente simulado: Mountain Car 1D simplificado
(carro precisa ganhar momentum para subir a montanha)
"""

import numpy as np
from collections import deque
import random


class MountainCar1D:
    """
    Versao simplificada do Mountain Car para demonstracao.
    Estado: [posicao, velocidade]
    Acoes: 0=esquerda, 1=nada, 2=direita
    """

    def __init__(self):
        self.state_size = 2
        self.action_size = 3
        self.pos = 0.0
        self.vel = 0.0
        self.step_count = 0

    def reset(self):
        self.pos = np.random.uniform(-0.6, -0.4)
        self.vel = 0.0
        self.step_count = 0
        return np.array([self.pos, self.vel])

    def step(self, action: int) -> tuple:
        forca = (action - 1) * 0.001   # -0.001, 0, +0.001
        gravidade = -0.0025 * np.cos(3 * self.pos)

        self.vel = np.clip(self.vel + forca + gravidade, -0.07, 0.07)
        self.pos = np.clip(self.pos + self.vel, -1.2, 0.6)

        if self.pos == -1.2:
            self.vel = 0.0

        done = self.pos >= 0.5
        recompensa = 0.0 if done else -1.0
        self.step_count += 1
        done = done or self.step_count >= 200

        return np.array([self.pos, self.vel]), recompensa, done


class DQNSimulado:
    """
    DQN simplificado usando arrays numpy (sem TensorFlow).
    Aproxima Q(s,a) com uma rede linear de 2 camadas.
    Para demonstração didática — não treinável para convergência real.
    """

    def __init__(self, state_size: int, action_size: int):
        np.random.seed(42)
        self.state_size = state_size
        self.action_size = action_size
        # Pesos da rede: simplificado como mapeamento linear
        self.W1 = np.random.randn(state_size, 16) * 0.1
        self.W2 = np.random.randn(16, action_size) * 0.1

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass simplificado."""
        h = np.tanh(state @ self.W1)
        return h @ self.W2

    def get_weights(self) -> list:
        return [self.W1.copy(), self.W2.copy()]

    def set_weights(self, weights: list) -> None:
        self.W1 = weights[0].copy()
        self.W2 = weights[1].copy()

    def train_step(self, states, targets, lr=0.001):
        """Atualização simplificada de gradiente (descida de gradiente)."""
        batch_loss = 0.0
        for s, t in zip(states, targets):
            pred = self.predict(s)
            error = pred - t
            batch_loss += np.mean(error ** 2)
            # Backprop simplificado
            h = np.tanh(s @ self.W1)
            grad_W2 = np.outer(h, error)
            self.W2 -= lr * grad_W2
        return batch_loss / len(states)


def treinar_dqn(num_episodes: int = 100, batch_size: int = 32,
                gamma: float = 0.99) -> dict:
    """
    Executa o loop completo de treinamento DQN.
    """
    env = MountainCar1D()
    Q_network = DQNSimulado(env.state_size, env.action_size)
    Q_target = DQNSimulado(env.state_size, env.action_size)
    Q_target.set_weights(Q_network.get_weights())

    replay_buffer = deque(maxlen=5000)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99

    recompensas_por_episodio = []
    passos_por_episodio = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        total_reward = 0
        steps = 0

        for step_ep in range(200):
            # ε-greedy
            if np.random.random() < epsilon:
                action = np.random.choice(env.action_size)
            else:
                q_values = Q_network.predict(state[0])
                action = int(np.argmax(q_values))

            # Executar ação
            next_state, reward, done = env.step(action)
            next_state_r = np.reshape(next_state, [1, env.state_size])

            # Armazenar no replay buffer
            replay_buffer.append(
                (state[0], action, reward, next_state_r[0], done)
            )
            total_reward += reward
            steps += 1
            state = next_state_r

            # Treinar se buffer tem dados suficientes
            if len(replay_buffer) >= batch_size:
                batch = random.sample(list(replay_buffer), batch_size)

                states_b = np.array([b[0] for b in batch])
                actions_b = np.array([b[1] for b in batch])
                rewards_b = np.array([b[2] for b in batch])
                next_states_b = np.array([b[3] for b in batch])
                dones_b = np.array([b[4] for b in batch])

                # Calcular targets
                targets = Q_network.predict(states_b[0]).reshape(1, -1)
                for i in range(len(batch)):
                    t = Q_network.predict(states_b[i]).copy()
                    if dones_b[i]:
                        t[actions_b[i]] = rewards_b[i]
                    else:
                        t[actions_b[i]] = (
                            rewards_b[i] + gamma * np.max(Q_target.predict(next_states_b[i]))
                        )
                    if i == 0:
                        targets = [t]
                    else:
                        targets.append(t)

                Q_network.train_step(states_b, targets, lr=0.001)

            if done:
                break

        # Atualizar target network a cada 10 episódios
        if (episode + 1) % 10 == 0:
            Q_target.set_weights(Q_network.get_weights())

        # Decair epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recompensas_por_episodio.append(total_reward)
        passos_por_episodio.append(steps)

    return {
        "recompensas": recompensas_por_episodio,
        "passos": passos_por_episodio,
        "epsilon_final": epsilon,
        "Q_network": Q_network,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GO1810 - LOOP DE TREINAMENTO DQN COMPLETO")
    print("=" * 60)

    print("\nEXECUTANDO TREINAMENTO DQN (100 episodios)...")
    print("(versao simplificada para demonstracao — sem TensorFlow)")
    print()

    random.seed(42)
    np.random.seed(42)

    resultado = treinar_dqn(num_episodes=100, batch_size=32)
    recompensas = resultado["recompensas"]
    passos = resultado["passos"]

    # Mostrar progresso
    print("Recompensa media por bloco de 20 episodios:")
    for bloco in range(5):
        inicio = bloco * 20
        fim = inicio + 20
        media = np.mean(recompensas[inicio:fim])
        media_passos = np.mean(passos[inicio:fim])
        barra = "#" * int(abs(media) / 4)
        print(f"  Ep {inicio+1:3d}-{fim:3d}: reward={media:7.1f}  passos={media_passos:5.1f}  {barra}")

    print()
    print("Ultimos 10 episodios:")
    for ep in range(90, 100):
        print(f"  Ep {ep+1:3d}: reward={recompensas[ep]:7.1f}  passos={passos[ep]:3d}  "
              f"epsilon={resultado['epsilon_final']:.3f}")

    print()
    print("COMPONENTES DO TREINAMENTO DQN:")
    componentes = [
        ("ε-greedy    ", "Balanceia exploracao e exploitacao"),
        ("Replay Buffer", "Quebra correlacao temporal das amostras"),
        ("Target Network", "Targets estaveis (sem alvo movel)"),
        ("Batch Training", "Eficiencia e estabilidade numerica"),
        ("ε decay     ", "Mais exploracao no inicio, exploitacao no fim"),
    ]
    for nome, desc in componentes:
        print(f"  {nome}: {desc}")

    print()
    print("  Para DQN completo com convergencia real:")
    print("  Ver GO1817-26ProjetoDqnCartpoleSetup.py (com TensorFlow/Gymnasium)")
