"""
GO1801 - Funções de Valor em Aprendizado por Reforço
=====================================================
Demonstra as funções de valor V(s) e Q(s,a) com exemplos numéricos.
Requer apenas bibliotecas padrão (numpy).

Conceito central de RL: funções de valor medem a "qualidade" de estar em
um estado ou tomar uma ação, considerando recompensas futuras esperadas.

Definições matemáticas:
  V^π(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t=s, π]
           (valor de estar no estado s seguindo a política π)

  Q^π(s,a) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t=s, a_t=a, π]
             (valor de tomar ação a no estado s e depois seguir π)

Relação: V^π(s) = Σ_a π(a|s) * Q^π(s,a)
"""

import numpy as np


def calcular_retorno_monte_carlo(recompensas: list, gamma: float) -> list:
    """
    Calcula o retorno acumulado descontado G_t para cada timestep.
    G_t = R_t + γ*R_{t+1} + γ²*R_{t+2} + ...

    Monte Carlo: percorre o episódio de trás para frente.
    Mais eficiente que somar série infinita a cada passo.
    """
    retornos = []
    G = 0.0
    for r in reversed(recompensas):
        G = r + gamma * G   # Equação de recorrência do retorno
        retornos.insert(0, G)
    return retornos


def calcular_v_grid_world():
    """
    Exemplo concreto: grid world 2x3 com política aleatória.
    Mostra como V(s) difere por estado.

    Grid (0=início, 5=objetivo, X=obstáculo):
    [0, 1, 2]
    [3, X, 5]  <- 5 é o objetivo (recompensa +10)
    """
    # Valores estimados para cada estado (resultado de iteração de valor)
    # Estado 5 (objetivo) tem valor alto; estados distantes têm valor menor
    V = {
        0: 2.4,   # Início: longe do objetivo
        1: 4.1,   # Centro-topo: mais próximo
        2: 7.3,   # Direita-topo: perto do objetivo
        3: 1.8,   # Esquerda-baixo: precisa desviar
        # 4 é obstáculo (sem valor)
        5: 10.0,  # OBJETIVO: valor máximo
    }
    return V


def calcular_q_exemplo():
    """
    Exemplo concreto de Q(s,a) para o estado 2 do grid world.
    Ações: 0=cima, 1=baixo, 2=esquerda, 3=direita
    """
    # Do estado 2 (direita-topo), as ações têm valores diferentes
    Q = {
        (2, 0): -1.0,   # Cima: sai do grid — inválido
        (2, 1): 9.2,    # Baixo: vai para estado 5 (objetivo!) — ótimo
        (2, 2): 4.1,    # Esquerda: vai para estado 1 — ok
        (2, 3): -1.0,   # Direita: sai do grid — inválido
    }
    return Q


if __name__ == "__main__":
    print("=" * 60)
    print("GO1801 - FUNÇÕES DE VALOR V(s) e Q(s,a)")
    print("=" * 60)

    # ─── Fórmulas ────────────────────────────────────────────
    print("\nFORMULAS:")
    print()
    print("  V^π(s) = E[Σ γᵏ R_{t+k} | s_t=s, π]")
    print("         Valor esperado de ESTAR no estado s")
    print()
    print("  Q^π(s,a) = E[Σ γᵏ R_{t+k} | s_t=s, a_t=a, π]")
    print("            Valor esperado de TOMAR ação a no estado s")
    print()
    print("  Relação: V^π(s) = Σ_a π(a|s) · Q^π(s,a)")
    print("           Se política é determinística: V^π(s) = Q^π(s, π(s))")

    # ─── Retorno com desconto ─────────────────────────────────
    print()
    print("─" * 60)
    print("CALCULO DO RETORNO G_t = Σ γᵏ R_{t+k}")
    print("─" * 60)

    recompensas = [-1, -1, -1, +10]  # 3 passos + chegou ao objetivo
    gamma = 0.9

    retornos = calcular_retorno_monte_carlo(recompensas, gamma)
    print(f"\n  Episodio: recompensas = {recompensas}")
    print(f"  gamma = {gamma}")
    print()
    for t, (r, G) in enumerate(zip(recompensas, retornos)):
        print(f"  t={t}: R_t={r:+4.1f}  →  G_t = {G:6.3f}")

    print(f"\n  Interpretacao:")
    print(f"  G_0 = {retornos[0]:.3f}: valor de estar no estado inicial")
    print(f"  G_3 = {retornos[3]:.3f}: chegou ao objetivo (G=R=10)")

    # ─── V(s) no grid world ───────────────────────────────────
    print()
    print("─" * 60)
    print("V(s) ESTIMADO — GRID WORLD 2x3")
    print("─" * 60)
    print()
    V = calcular_v_grid_world()
    print("  Grid:")
    print(f"  | V[0]={V[0]:.1f} | V[1]={V[1]:.1f} | V[2]={V[2]:.1f} |")
    print(f"  | V[3]={V[3]:.1f} | obst.  | V[5]={V[5]:.1f} | <- OBJETIVO")
    print()
    print("  Estados mais proximos do objetivo tem maior V(s).")
    print("  Agente otimo: sempre move para o estado com maior V(s')")

    # ─── Q(s,a) no estado 2 ───────────────────────────────────
    print()
    print("─" * 60)
    print("Q(s,a) NO ESTADO 2 (direita-topo)")
    print("─" * 60)
    Q = calcular_q_exemplo()
    acoes = {0: "Cima ", 1: "Baixo", 2: "Esq. ", 3: "Dir. "}
    for (s, a), q in sorted(Q.items(), key=lambda x: -x[1]):
        melhor = " <- MELHOR ACAO" if q == max(Q.values()) else ""
        print(f"  Q({s}, {acoes[a]}) = {q:5.1f}{melhor}")

    print()
    print("  Politica otima no estado 2: BAIXO (vai direto ao objetivo)")
    print()
    print("  Q-Learning aprende Q* diretamente da experiencia (sem modelo do ambiente).")
    print("  Politica otima: π*(s) = argmax_a Q*(s,a)")
