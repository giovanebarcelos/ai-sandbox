"""
GO1812 - Advantage Function: A(s,a) = Q(s,a) - V(s)
=====================================================
Demonstra a Advantage Function com exemplo numérico concreto.
Requer apenas numpy.

A função de vantagem mede QUANTO MELHOR é tomar a ação 'a' no estado 's'
em comparação com a ação média (dada pela política atual).

  A(s,a) = Q(s,a) - V(s)

Interpretação:
  A(s,a) > 0: ação 'a' é melhor que a média → reforçar esta ação
  A(s,a) < 0: ação 'a' é pior que a média  → desencorajar esta ação
  A(s,a) = 0: ação 'a' é exatamente a média

Por que usar Advantage em vez de Q(s,a) direto?
  - Normaliza o sinal de aprendizado pelo valor de referência V(s)
  - Reduz variância do gradiente (sem mudar o gradiente esperado!)
  - Estado com V(s) = 100: Q(s,a) = 101 parece bom, mas só é marginalmente melhor
"""

import numpy as np


def calcular_advantage(Q: dict, V: dict, estado: int) -> dict:
    """
    Calcula A(s,a) para todas as ações no estado dado.

    Q: {(estado, acao): q_value}
    V: {estado: v_value}
    """
    acoes = [a for (s, a) in Q.keys() if s == estado]
    return {
        a: Q[(estado, a)] - V[estado]
        for a in acoes
    }


def demonstrar_advantage_grid_world():
    """
    Demonstra A(s,a) no Grid World 2x3.
    """
    # Q* e V* após convergência do Q-Learning no Grid World
    # Grid: estados 0-5, estado 5 = objetivo
    Q = {
        # estado 2 (direita-topo): próximo do objetivo
        (2, 'cima'):    -2.0,
        (2, 'baixo'):    8.5,   # vai para estado 5 (objetivo)!
        (2, 'esquerda'): 3.1,
        (2, 'direita'):  -2.0,
        # estado 1 (centro-topo): moderado
        (1, 'cima'):    -2.0,
        (1, 'baixo'):    2.0,
        (1, 'esquerda'): 1.5,
        (1, 'direita'):  3.1,
        # estado 0 (esquerda-topo): longe do objetivo
        (0, 'cima'):    -2.0,
        (0, 'baixo'):    1.0,
        (0, 'esquerda'): 0.5,
        (0, 'direita'):  1.5,
    }

    V = {
        0: 1.25,  # média das Q(0,a) pela política atual
        1: 1.15,
        2: 1.90,  # V = média ponderada das Q-values
    }

    return Q, V


def advantage_como_td_error(gamma: float = 0.99) -> None:
    """
    Mostra que A(s,a) pode ser aproximado pelo TD Error:
    A(s,a) ≈ r + γ V(s') - V(s)  (Advantage com bootstrap)

    Isso evita estimar Q(s,a) separadamente — basta V(s).
    """
    # Exemplo numérico
    exemplos = [
        {"s": 2, "a": "baixo", "r": +10, "s_prox": 5, "V_s": 4.1, "V_sp": 0.0},
        {"s": 2, "a": "esq",   "r": -1,  "s_prox": 1, "V_s": 4.1, "V_sp": 3.0},
        {"s": 1, "a": "dir",   "r": -1,  "s_prox": 2, "V_s": 3.0, "V_sp": 4.1},
    ]
    print("  Advantage via TD Error: A(s,a) ≈ r + γ*V(s') - V(s)")
    print(f"  {'Estado':>8} {'Acao':>10} {'A(s,a)':>10} {'Interpretacao'}")
    print("  " + "-" * 55)
    for ex in exemplos:
        td = ex["r"] + gamma * ex["V_sp"] - ex["V_s"]
        interp = "MELHOR" if td > 0.5 else ("PIOR" if td < -0.5 else "NEUTRO")
        print(f"  {ex['s']:>8} {ex['a']:>10} {td:>10.3f}  {interp}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1812 - ADVANTAGE FUNCTION: A(s,a) = Q(s,a) - V(s)")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  A(s,a) = Q(s,a) - V(s)")
    print()
    print("  Onde:")
    print("    Q(s,a) = valor de tomar acao 'a' no estado 's'")
    print("    V(s)   = valor medio de 's' seguindo a politica atual")
    print("    A(s,a) = O QUANTO 'a' e MELHOR (ou pior) que a media")

    # ─── Exemplo numérico ─────────────────────────────────────
    print()
    print("─" * 60)
    print("EXEMPLO NUMERICO: Grid World 2x3")
    print("─" * 60)

    Q, V = demonstrar_advantage_grid_world()

    for estado in [2, 1, 0]:
        advantages = calcular_advantage(Q, V, estado)
        print(f"\n  Estado {estado}:  V(s) = {V[estado]:.2f}")
        print(f"  {'Acao':>10} | {'Q(s,a)':>8} | {'A(s,a)':>8} | Decisao")
        print("  " + "-" * 50)
        for a in sorted(advantages.keys()):
            q = Q[(estado, a)]
            adv = advantages[a]
            decisao = "REFORCAR" if adv > 0.5 else ("INIBIR" if adv < -0.5 else "neutro")
            print(f"  {a:>10} | {q:>8.2f} | {adv:>8.3f} | {decisao}")

    # ─── Advantage via TD Error ───────────────────────────────
    print()
    print("─" * 60)
    print("ADVANTAGE APROXIMADO POR TD ERROR (mais pratico):")
    print("─" * 60)
    print()
    advantage_como_td_error()

    print()
    print("─" * 60)
    print("POR QUE USAR ADVANTAGE?")
    print("─" * 60)
    print()
    print("  Sem Advantage (usar Q direto):")
    print("    Q(s, boa_acao)  = 101  → gradiente = +1")
    print("    Q(s, ruim_acao) = 100  → gradiente = 0?")
    print("    (Dificil distinguir 'bom' de 'médio' com valores absolutos)")
    print()
    print("  Com Advantage:")
    print("    V(s) = 100  (baseline)")
    print("    A(s, boa_acao)  = +1  → reforcar")
    print("    A(s, ruim_acao) = 0   → neutro")
    print("    (Sinal claro: relativo ao valor médio do estado)")
    print()
    print("  Advantage e usado em: A2C, A3C, PPO, GAE (Generalized A.E.)")
