"""
GO1804 - Política ε-greedy (Exploração vs. Exploração)
=======================================================
Demonstra o dilema Exploração vs. Exploração e a estratégia ε-greedy.
Requer apenas numpy.

Dilema fundamental do RL:
  EXPLOITATION: usar o conhecimento atual (tomar a melhor ação conhecida)
  EXPLORATION : tentar novas ações (descobrir ações potencialmente melhores)

Estratégia ε-greedy:
  com probabilidade ε → Explorar: escolher ação aleatória
  com probabilidade 1-ε → Explotar: escolher argmax_a Q(s,a)

ε decay: ε começa alto (explorar muito no início, quando Q está errado)
         e decai gradualmente (explotar mais conforme Q melhora).
"""

import numpy as np


def epsilon_greedy(Q_s: np.ndarray, epsilon: float) -> int:
    """
    Política ε-greedy para o estado s.
    Q_s: vetor de Q-values para o estado s (uma entrada por ação).
    Retorna o índice da ação escolhida.
    """
    if np.random.random() < epsilon:
        # EXPLORAÇÃO: ação aleatória (aprender sobre ações pouco visitadas)
        return np.random.choice(len(Q_s))
    else:
        # EXPLORAÇÃO: melhor ação conhecida (usar conhecimento atual)
        return int(np.argmax(Q_s))


def simular_bandit(n_bracos: int = 5, n_passos: int = 1000,
                   epsilon: float = 0.1) -> dict:
    """
    Simula o problema do Multi-Armed Bandit (k braços).
    Cada braço tem uma recompensa média diferente (desconhecida pelo agente).

    Retorna estatísticas da exploração.
    """
    np.random.seed(42)
    # Recompensas verdadeiras dos braços (desconhecidas pelo agente)
    recompensas_verdadeiras = np.array([1.5, 2.0, 1.0, 3.0, 2.5])[:n_bracos]

    # Q-values estimados (iniciam em 0)
    Q = np.zeros(n_bracos)
    contagens = np.zeros(n_bracos, dtype=int)

    recompensas_recebidas = []
    acoes_tomadas = []

    for t in range(n_passos):
        # Escolher ação via ε-greedy
        acao = epsilon_greedy(Q, epsilon)

        # Receber recompensa (com ruído gaussiano)
        recompensa = recompensas_verdadeiras[acao] + np.random.randn() * 0.5

        # Atualizar estimativa incremental: Q_novo = Q_antigo + (r - Q_antigo) / n
        contagens[acao] += 1
        Q[acao] += (recompensa - Q[acao]) / contagens[acao]

        recompensas_recebidas.append(recompensa)
        acoes_tomadas.append(acao)

    return {
        "Q_final": Q,
        "recompensas_verdadeiras": recompensas_verdadeiras,
        "contagens": contagens,
        "recompensa_media": np.mean(recompensas_recebidas),
        "acao_otima": int(np.argmax(recompensas_verdadeiras)),
        "acao_aprendida": int(np.argmax(Q)),
        "taxa_otima": sum(a == np.argmax(recompensas_verdadeiras)
                         for a in acoes_tomadas) / n_passos,
    }


def comparar_epsilons(n_passos: int = 500) -> list:
    """Compara diferentes valores de ε no mesmo bandit."""
    resultados = []
    for eps in [0.0, 0.01, 0.1, 0.3, 1.0]:
        res = simular_bandit(n_bracos=5, n_passos=n_passos, epsilon=eps)
        resultados.append((eps, res))
    return resultados


if __name__ == "__main__":
    print("=" * 60)
    print("GO1804 - POLITICA E-GREEDY: EXPLORACAO vs EXPLOITACAO")
    print("=" * 60)

    # ─── Código da política ε-greedy ─────────────────────────
    print("\nCODIGO DA POLITICA E-GREEDY:")
    print()
    print("  if random() < ε:")
    print("      acao = random_action()   # Explorar (ε prob)")
    print("  else:")
    print("      acao = argmax_a Q(s,a)  # Exploitar (1-ε prob)")

    # ─── Exemplo numérico simples ─────────────────────────────
    print()
    print("─" * 60)
    print("EXEMPLO NUMERICO: escolher acao com e-greedy")
    print("─" * 60)

    Q_estado = np.array([2.5, 4.1, 1.8, 3.3])  # Q-values para 4 ações
    epsilon = 0.1

    print(f"\n  Q(s, *) = {Q_estado}  (4 acoes)")
    print(f"  ε = {epsilon}")
    print(f"  Melhor acao (argmax): {np.argmax(Q_estado)} (Q={np.max(Q_estado)})")
    print()

    np.random.seed(7)
    resultados_sorteio = []
    for _ in range(10):
        acao = epsilon_greedy(Q_estado, epsilon)
        tipo = "EXPLORAR" if acao != np.argmax(Q_estado) else "EXPLOITAR"
        resultados_sorteio.append((acao, tipo))

    print("  10 amostras da política:")
    for i, (a, tipo) in enumerate(resultados_sorteio, 1):
        print(f"    [{i:2d}] acao={a}  ({tipo})")

    exploradas = sum(1 for _, t in resultados_sorteio if t == "EXPLORAR")
    print(f"\n  Explorou {exploradas}/10 vezes (esperado ~{epsilon*10:.0f})")

    # ─── Comparação de epsilons no bandit ─────────────────────
    print()
    print("─" * 60)
    print("COMPARACAO: Diferentes valores de ε (Multi-Armed Bandit)")
    print("─" * 60)

    comparacoes = comparar_epsilons(n_passos=500)
    print(f"\n  {'ε':>8} | {'Recomp.Média':>13} | {'Taxa ótima':>11} | Comentário")
    print("  " + "-" * 65)
    for eps, res in comparacoes:
        comentario = {
            0.0: "Gananciosa pura — fica presa na primeira boa",
            0.01: "Quase gananciosa — explora raramente",
            0.1:  "Equilibrio tipico — recomendado",
            0.3:  "Muito explorador — demora a convergir",
            1.0:  "Aleatoria pura — nunca converge",
        }.get(eps, "")
        otima = "*" if res["acao_aprendida"] == res["acao_otima"] else " "
        print(f"  {eps:>8.2f} | {res['recompensa_media']:>13.3f} | "
              f"{res['taxa_otima']:>10.1%}{otima} | {comentario}")

    print()
    print("  * = encontrou a ação ótima")
    print()
    print("─" * 60)
    print("E-DECAY: melhor estrategia pratica")
    print("─" * 60)
    print()
    print("  epsilon_inicio = 1.0   # Explorar muito no inicio")
    print("  epsilon_min    = 0.01  # Sempre explorar um pouco")
    print("  decay          = 0.995 # Decair a cada episodio")
    print()
    print("  Intuicao: no inicio Q e ruim (explorar muito)")
    print("  Com o tempo Q melhora (exploitar mais)")
    print()
    print("  Ver GO1806 para e-decay em acao no Grid World.")
