"""
GO1819 - Exercício 2: Double DQN no CartPole
=============================================
Implementa o Double DQN com explicação das duas variantes.
Requer: pip install gymnasium tensorflow matplotlib

Double DQN (Van Hasselt et al. 2015) corrige o viés de superestimação do DQN:

  DQN padrão:
    target = r + γ * max_a' Q_target(s', a')
    Problema: max do Q_target tende a SUPERESTIMAR (viés positivo)

  Double DQN:
    best_action = argmax_a' Q_network(s', a')   ← Q_network escolhe a ação
    target = r + γ * Q_target(s', best_action)  ← Q_target avalia essa ação
    Resultado: menor viés, aprendizado mais estável

Regra Double DQN do slide:
    best_action = np.argmax(Q_network.predict(s'))
    target = r + gamma * Q_target.predict(s')[best_action]
"""

import sys
import subprocess
import numpy as np


def instalar_deps():
    for pkg in ["gymnasium", "tensorflow", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def calcular_target_dqn(Q_network, Q_target, next_state, reward: float,
                        done: bool, gamma: float = 0.99) -> float:
    """
    Calcula target para DQN padrão (com superestimação).
    target = r + γ * max Q_target(s', a')
    """
    if done:
        return reward
    q_next = Q_target.predict(next_state, verbose=0)[0]
    return reward + gamma * np.max(q_next)


def calcular_target_double_dqn(Q_network, Q_target, next_state, reward: float,
                                done: bool, gamma: float = 0.99) -> float:
    """
    Calcula target para Double DQN (sem superestimação).
    1. Q_network escolhe a melhor ação (sem viés de target network)
    2. Q_target AVALIA essa ação específica
    """
    if done:
        return reward
    # Double DQN: separar seleção e avaliação
    best_action = np.argmax(Q_network.predict(next_state, verbose=0)[0])
    target_value = Q_target.predict(next_state, verbose=0)[0][best_action]
    return reward + gamma * target_value


def treinar_double_dqn(num_episodes: int = 300, batch_size: int = 32,
                       gamma: float = 0.99) -> dict:
    """Treina Double DQN no CartPole-v1."""
    import gymnasium as gym
    from tensorflow import keras
    from keras import layers
    from collections import deque
    import random

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_size, activation='linear'),
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
        return model

    Q_network = build_model()
    Q_target = build_model()
    Q_target.set_weights(Q_network.get_weights())

    replay = deque(maxlen=10000)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    rewards_hist = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for _ in range(500):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_network.predict(state, verbose=0)[0])

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_state = np.reshape(next_state, [1, state_size])
            replay.append((state[0], action, reward, next_state[0], done))
            total_reward += reward
            state = next_state

            if len(replay) >= batch_size:
                batch = random.sample(list(replay), batch_size)
                states_b = np.array([b[0] for b in batch])
                actions_b = np.array([b[1] for b in batch])
                rewards_b = np.array([b[2] for b in batch])
                next_b = np.array([b[3] for b in batch])
                dones_b = np.array([b[4] for b in batch])

                targets = Q_network.predict(states_b, verbose=0)
                for i in range(batch_size):
                    ns = next_b[i:i+1]
                    # DOUBLE DQN: separar seleção e avaliação
                    best_a = np.argmax(Q_network.predict(ns, verbose=0)[0])
                    if dones_b[i]:
                        targets[i][actions_b[i]] = rewards_b[i]
                    else:
                        targets[i][actions_b[i]] = (
                            rewards_b[i] + gamma * Q_target.predict(ns, verbose=0)[0][best_a]
                        )
                Q_network.fit(states_b, targets, epochs=1, verbose=0)

            if done:
                break

        if (episode + 1) % 10 == 0:
            Q_target.set_weights(Q_network.get_weights())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_hist.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards_hist[-50:])
            print(f"  Ep {episode+1:4d} | Avg(50): {avg:7.1f} | ε={epsilon:.3f}")

    env.close()
    return {"rewards": rewards_hist, "Q_network": Q_network}


def salvar_grafico(rewards_hist: list) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rewards_hist, alpha=0.3, color='steelblue', label='Reward')
        n = 50
        avg = [np.mean(rewards_hist[max(0, i-n):i+1]) for i in range(len(rewards_hist))]
        ax.plot(avg, color='red', linewidth=2, label=f'Média móvel ({n} ep)')
        ax.axhline(y=195, color='green', linestyle='--', label='Resolvido (195)')
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Total Reward")
        ax.set_title("Double DQN no CartPole-v1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("GO1819_Exercicio2_DoubleDQN.png", dpi=120, bbox_inches='tight')
        print("  Grafico salvo: GO1819_Exercicio2_DoubleDQN.png")
    except Exception as e:
        print(f"  Grafico nao salvo: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXERCICIO 2 - DOUBLE DQN NO CARTPOLE")
    print("=" * 60)

    print("\nDQN vs Double DQN:")
    print()
    print("  DQN (padrao):")
    print("    target = r + γ * max Q_target(s')")
    print("    Problema: max superstima (escolhe e avalia com mesma rede)")
    print()
    print("  Double DQN:")
    print("    best_action = argmax Q_network(s')  # Q_network ESCOLHE")
    print("    target = r + γ * Q_target(s')[best_action]  # Q_target AVALIA")
    print("    Resultado: menor bias, aprendizado mais estavel")

    instalar_deps()

    try:
        import gymnasium
        import tensorflow

        print("\nTreinando Double DQN (300 episodios)...")
        resultado = treinar_double_dqn(num_episodes=300)

        rewards = resultado["rewards"]
        print(f"\nResultados:")
        print(f"  Recompensa maxima: {max(rewards):.1f}")
        print(f"  Media ultimos 50:  {np.mean(rewards[-50:]):.1f}")
        print(f"  Resolvido (>=195): {'SIM' if np.mean(rewards[-100:]) >= 195 else 'Continuar treinando'}")

        salvar_grafico(rewards)

    except ImportError as e:
        print(f"\nDependencia faltando: {e}")
        print("Execute: pip install gymnasium tensorflow matplotlib")
        print()
        print("Logica do Double DQN (pseudocodigo):")
        print("  for each transition in batch:")
        print("      best_action = argmax(Q_network.predict(s'))")
        print("      q_val = Q_target.predict(s')[best_action]")
        print("      target = r + gamma * q_val  (se nao done)")
