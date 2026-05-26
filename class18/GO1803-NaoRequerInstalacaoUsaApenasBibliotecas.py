"""
GO1803 - Q-Learning: Regra de Atualização TD
=============================================
Demonstra a regra de atualização do Q-Learning com exemplo numérico.
Requer apenas numpy.

A regra TD (Temporal Difference) do Q-Learning:
  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
                         └─────────── TD Error ──────────┘

Onde:
  α (alpha) : taxa de aprendizado (0 < α ≤ 1)
  γ (gamma) : fator de desconto (0 ≤ γ < 1)
  r         : recompensa imediata recebida
  s'        : próximo estado
  TD Error  : diferença entre estimativa atual e estimativa melhorada

Intuição: "aprende" movendo Q(s,a) em direção ao alvo r + γ max Q(s',a'),
que é a estimativa da Bellman para Q*(s,a).
"""

import numpy as np


def atualizar_q(Q: np.ndarray, s: int, a: int, r: float,
                s_prox: int, alpha: float, gamma: float) -> tuple:
    """
    Aplica uma atualização Q-Learning.
    Retorna (Q_novo, td_error, alvo).
    """
    # Alvo: r + γ max_a' Q(s', a')  — a "melhor estimativa atual"
    alvo = r + gamma * np.max(Q[s_prox])

    # TD Error: discrepância entre estimativa atual e alvo
    td_error = alvo - Q[s, a]

    # Atualização: move Q(s,a) em direção ao alvo
    Q_novo = Q.copy()
    Q_novo[s, a] = Q[s, a] + alpha * td_error

    return Q_novo, td_error, alvo


def demonstrar_atualizacoes_sequenciais():
    """
    Simula várias atualizações Q-Learning no Grid World 1D.
    Mostra como Q(s,a) converge para Q*.
    """
    # Grid 1D: 4 estados, 2 ações (0=esquerda, 1=direita)
    n_estados, n_acoes = 4, 2
    Q = np.zeros((n_estados, n_acoes))

    alpha, gamma = 0.5, 0.9

    # Trajetória de exemplo: ir da esquerda (s=0) até o objetivo (s=3)
    # (estado_atual, ação, recompensa, próximo_estado)
    trajetoria = [
        (0, 1, -1, 1),   # s=0, direita → s=1, r=-1
        (1, 1, -1, 2),   # s=1, direita → s=2, r=-1
        (2, 1, +10, 3),  # s=2, direita → s=3(objetivo!), r=+10
    ]

    historico = []
    for s, a, r, s_prox in trajetoria:
        Q_antes = Q[s, a]
        Q, td_error, alvo = atualizar_q(Q, s, a, r, s_prox, alpha, gamma)
        historico.append({
            "s": s, "a": a, "r": r, "s_prox": s_prox,
            "Q_antes": Q_antes,
            "alvo": alvo,
            "td_error": td_error,
            "Q_depois": Q[s, a],
        })

    return Q, historico


if __name__ == "__main__":
    print("=" * 60)
    print("GO1803 - Q-LEARNING: REGRA DE ATUALIZACAO TD")
    print("=" * 60)

    # ─── Formula ──────────────────────────────────────────────
    print("\nFORMULA:")
    print()
    print("  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]")
    print("                         └────────── TD Error ──────────┘")
    print()
    print("  α (alpha) = taxa de aprendizado  [0.1 a 0.5 típico]")
    print("  γ (gamma) = fator de desconto    [0.9 a 0.99 típico]")
    print("  TD Error  = alvo - estimativa atual")

    # ─── Exemplo numérico manual ─────────────────────────────
    print()
    print("─" * 60)
    print("EXEMPLO NUMERICO MANUAL:")
    print("─" * 60)

    Q_manual = np.array([[0.0, 0.0],   # estado 0
                          [0.0, 0.0],   # estado 1
                          [0.0, 0.0],   # estado 2
                          [0.0, 0.0]])  # estado 3 (objetivo)

    alpha, gamma = 0.5, 0.9
    s, a, r, s_prox = 2, 1, +10, 3  # estado 2, ação direita, r=+10, vai p/ estado 3

    alvo = r + gamma * np.max(Q_manual[s_prox])
    td_err = alvo - Q_manual[s, a]
    Q_novo_manual = Q_manual[s, a] + alpha * td_err

    print(f"\n  Situacao: s={s}, a={a}(direita), r={r}, s'={s_prox}")
    print(f"  Q({s},{a}) antes = {Q_manual[s, a]:.2f}")
    print()
    print(f"  Passo 1: Alvo = r + γ · max Q(s', a')")
    print(f"           Alvo = {r} + {gamma} · max{list(Q_manual[s_prox])}")
    print(f"           Alvo = {r} + {gamma} · {np.max(Q_manual[s_prox]):.2f}")
    print(f"           Alvo = {alvo:.2f}")
    print()
    print(f"  Passo 2: TD Error = Alvo - Q({s},{a})")
    print(f"           TD Error = {alvo:.2f} - {Q_manual[s, a]:.2f} = {td_err:.2f}")
    print()
    print(f"  Passo 3: Q({s},{a}) ← Q({s},{a}) + α · TD Error")
    print(f"           Q({s},{a}) ← {Q_manual[s, a]:.2f} + {alpha} · {td_err:.2f}")
    print(f"           Q({s},{a}) ← {Q_novo_manual:.2f}")

    # ─── Sequência de atualizações ────────────────────────────
    print()
    print("─" * 60)
    print("ATUALIZACOES SEQUENCIAIS NO EPISODIO")
    print("─" * 60)

    Q_seq, historico = demonstrar_atualizacoes_sequenciais()

    nomes_acoes = {0: "esq", 1: "dir"}
    for i, h in enumerate(historico):
        print(f"\n  Passo {i + 1}: s={h['s']}, a={nomes_acoes[h['a']]}, "
              f"r={h['r']:+4.0f}, s'={h['s_prox']}")
        print(f"    Q({h['s']},{h['a']}) = {h['Q_antes']:6.3f}")
        print(f"    Alvo = {h['alvo']:6.3f}  |  TD Error = {h['td_error']:+.3f}")
        print(f"    Q({h['s']},{h['a']}) ← {h['Q_depois']:6.3f}")

    print()
    print("  Q-table apos o episodio:")
    print(f"  {'Estado':>8} | {'Esquerda':>10} | {'Direita':>10}")
    print("  " + "-" * 35)
    for s in range(4):
        print(f"  {s:>8} | {Q_seq[s, 0]:>10.3f} | {Q_seq[s, 1]:>10.3f}")

    print()
    print("  Após muitos episodios, Q converge para Q* (Bellman).")
    print("  Politica otima: π*(s) = argmax_a Q*(s,a)")
    print()
    print("  Ver GO1806 para Q-Learning completo no Grid World.")
