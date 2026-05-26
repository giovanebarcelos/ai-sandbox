"""
GO1805 - Pseudocódigo do Q-Learning com Exemplo Executável
============================================================
Demonstra o algoritmo Q-Learning completo em um ambiente simples.
Requer apenas numpy.

O pseudocódigo clássico do Q-Learning (Watkins, 1989):
  Inicializar Q(s,a) = 0 para todos (s,a)
  Para cada episódio:
    s = estado inicial
    Enquanto não terminou:
      Escolher a usando ε-greedy em Q(s,·)
      Executar a, observar r, s'
      Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
      s ← s'
    Decair ε

Propriedade fundamental: Q-Learning converge para Q* (ótimo global)
se cada par (s,a) é visitado infinitas vezes e α decai adequadamente.
"""

import numpy as np


class GridWorld1D:
    """
    Ambiente Grid World 1D: [0] → [1] → [2] → [3=OBJETIVO]
    Ações: 0=esquerda, 1=direita
    Recompensa: -1 por passo, +10 ao chegar no estado 3
    """

    def __init__(self):
        self.n_estados = 4
        self.n_acoes = 2
        self.estado_objetivo = 3

    def reset(self) -> int:
        """Inicia no estado 0."""
        return 0

    def step(self, estado: int, acao: int) -> tuple:
        """
        Executa ação e retorna (prox_estado, recompensa, terminado).
        acao: 0=esquerda, 1=direita
        """
        if acao == 1:  # direita
            prox = min(estado + 1, self.n_estados - 1)
        else:           # esquerda
            prox = max(estado - 1, 0)

        terminado = prox == self.estado_objetivo
        recompensa = +10.0 if terminado else -1.0

        return prox, recompensa, terminado


def q_learning(env: GridWorld1D, num_episodios: int = 500,
               alpha: float = 0.1, gamma: float = 0.99) -> tuple:
    """
    Treina com Q-Learning e retorna (Q_table, historico_recompensas).
    """
    Q = np.zeros((env.n_estados, env.n_acoes))
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    recompensas_por_episodio = []

    for episodio in range(num_episodios):
        s = env.reset()
        recompensa_total = 0

        for _ in range(100):  # limite de passos por episódio
            # Política ε-greedy
            if np.random.random() < epsilon:
                a = np.random.choice(env.n_acoes)
            else:
                a = int(np.argmax(Q[s]))

            # Executar ação
            s_prox, r, done = env.step(s, a)

            # Atualização Q-Learning
            Q[s, a] = Q[s, a] + alpha * (
                r + gamma * np.max(Q[s_prox]) - Q[s, a]
            )

            recompensa_total += r
            s = s_prox

            if done:
                break

        # Decair epsilon após cada episódio
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recompensas_por_episodio.append(recompensa_total)

    return Q, recompensas_por_episodio


def extrair_politica(Q: np.ndarray) -> dict:
    """Extrai a política ótima (ação com maior Q) para cada estado."""
    acoes = {0: "esq", 1: "dir"}
    return {s: acoes[int(np.argmax(Q[s]))] for s in range(len(Q))}


if __name__ == "__main__":
    print("=" * 60)
    print("GO1805 - PSEUDOCODIGO Q-LEARNING EXECUTAVEL")
    print("=" * 60)

    print("\nPSEUDOCODIGO:")
    print()
    print("  Q = zeros(n_estados, n_acoes)")
    print("  epsilon = 1.0")
    print("  Para cada episodio:")
    print("    s = env.reset()")
    print("    done = False")
    print("    Enquanto not done:")
    print("      if random < epsilon: a = random_action()")
    print("      else:                a = argmax Q[s]")
    print("      s', r, done = env.step(a)")
    print("      Q[s,a] += alpha * (r + gamma * max Q[s'] - Q[s,a])")
    print("      s = s'")
    print("    epsilon = max(0.01, epsilon * 0.995)")

    # ─── Executar Q-Learning ──────────────────────────────────
    print()
    print("─" * 60)
    print("EXECUTANDO Q-LEARNING: GRID WORLD 1D")
    print("─" * 60)

    np.random.seed(42)
    env = GridWorld1D()
    Q, historico = q_learning(env, num_episodios=500)

    # Mostrar convergência
    print("\n  Recompensa media por periodo:")
    n = len(historico)
    periodos = [historico[:n//5], historico[n//5:2*n//5],
                historico[2*n//5:3*n//5], historico[3*n//5:4*n//5],
                historico[4*n//5:]]
    nomes = ["ep 1-100", "ep 101-200", "ep 201-300", "ep 301-400", "ep 401-500"]
    for nome, periodo in zip(nomes, periodos):
        media = np.mean(periodo)
        barra = "#" * int((media + 4) * 3)  # normalizar para visualização
        print(f"  {nome}: {media:6.2f}  {barra}")

    # Q-table final
    print()
    print("  Q-table convergeida:")
    print(f"  {'Estado':>8} | {'Q(esq)':>10} | {'Q(dir)':>10} | {'Acao Otima':>10}")
    print("  " + "-" * 48)
    nomes_acoes = {0: "esq", 1: "dir"}
    for s in range(env.n_estados):
        melhor = nomes_acoes[int(np.argmax(Q[s]))]
        print(f"  {s:>8} | {Q[s, 0]:>10.3f} | {Q[s, 1]:>10.3f} | {melhor:>10}")

    # Política ótima
    politica = extrair_politica(Q)
    print()
    print("  Politica otima extraida:")
    print("  [0] → [1] → [2] → [3=OBJETIVO]")
    for s, a in politica.items():
        print(f"    π*({s}) = {a}")

    # Testar política
    print()
    print("  Testando politica aprendida (episodio greedy):")
    s = env.reset()
    caminho = [s]
    for _ in range(10):
        a = int(np.argmax(Q[s]))
        s, r, done = env.step(s, a)
        caminho.append(s)
        if done:
            break
    print(f"    Caminho: {caminho}")
    print(f"    Chegou ao objetivo: {'SIM' if caminho[-1] == env.estado_objetivo else 'NAO'}")
    print()
    print("  Ver GO1806 para aplicação em Grid World 2D completo.")
