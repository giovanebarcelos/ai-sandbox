# GO1818-Exercício1QlearningNoFrozenlake
import gymnasium as gym
import numpy as np


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)

    # TODO:
    # 1. Criar Q-table: Q = np.zeros((n_states, n_actions))
    # 2. Implementar loop de treinamento com ε-greedy
    # 3. Update rule: Q[s,a] += alpha * (r + gamma * max Q[s'] - Q[s,a])
    # 4. Treinar 10.000 episódios
    # 5. Testar política aprendida
    # 6. Comparar is_slippery=True (estocástico) vs False (determinístico)

    # Sucesso: chegar no objetivo (G) em > 70% das tentativas
