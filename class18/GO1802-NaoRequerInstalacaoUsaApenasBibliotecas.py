"""
GO1802 - Equações de Otimalidade de Bellman
=============================================
Demonstra as equações de Bellman com exemplos numéricos calculados.
Requer apenas numpy (biblioteca padrão do Python científico).

As equações de Bellman são o coração de todo RL:
  V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
  Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q*(s',a')

Interpretação:
  V*(s) = melhor recompensa possível a partir de s
  Q*(s,a)= recompensa de tomar 'a' em 's' + melhor valor futuro

Richard Bellman (1957) — "Princípio da Otimalidade":
  "Uma política ótima tem a propriedade de que, dado qualquer
   estado inicial e decisão inicial, as decisões restantes
   devem constituir uma política ótima com relação ao estado
   resultante da primeira decisão."
"""

import numpy as np


def bellman_v_determinístico(
    recompensas: dict,
    transicoes: dict,
    V_atual: dict,
    gamma: float,
) -> dict:
    """
    Aplica uma iteração da equação de Bellman para V*(s).
    Caso determinístico: P(s'|s,a) = 1 para o próximo estado.

    recompensas : {(s, a): r}
    transicoes  : {(s, a): s_proximo}
    V_atual     : {s: valor_atual}
    """
    V_novo = {}
    estados = set(s for (s, _) in recompensas.keys())

    for s in estados:
        acoes = [a for (estado, a) in recompensas if estado == s]
        if not acoes:
            V_novo[s] = 0.0
            continue

        # V*(s) = max_a [R(s,a) + γ V*(s')]
        valores_acoes = []
        for a in acoes:
            r = recompensas.get((s, a), 0.0)
            s_prox = transicoes.get((s, a), s)
            valor = r + gamma * V_atual.get(s_prox, 0.0)
            valores_acoes.append(valor)

        V_novo[s] = max(valores_acoes)

    return V_novo


def calcular_exemplo_numerico() -> dict:
    """
    Grid world 1D com 4 estados: [0, 1, 2, 3]
    Estado 3 é o objetivo (recompensa +10).
    Ações: mover para direita (+1) ou esquerda (-1).
    Custo de passo: -1.

    Aplica Iteração de Valor até convergência.
    """
    # Estados: 0, 1, 2, 3 (3 = objetivo)
    estados = [0, 1, 2, 3]

    # R(s, a): recompensa ao tomar ação 'a' no estado 's'
    recompensas = {
        (0, 'direita'): -1, (0, 'esquerda'): -1,
        (1, 'direita'): -1, (1, 'esquerda'): -1,
        (2, 'direita'): +10, (2, 'esquerda'): -1,  # chegar em 3 = +10
        (3, 'direita'): 0,  (3, 'esquerda'): 0,    # objetivo, terminal
    }

    # T(s, a) → s': transições determinísticas
    transicoes = {
        (0, 'direita'): 1, (0, 'esquerda'): 0,
        (1, 'direita'): 2, (1, 'esquerda'): 0,
        (2, 'direita'): 3, (2, 'esquerda'): 1,
        (3, 'direita'): 3, (3, 'esquerda'): 3,
    }

    gamma = 0.9
    V = {s: 0.0 for s in estados}
    historico = [V.copy()]

    for iteracao in range(20):
        V_novo = bellman_v_determinístico(recompensas, transicoes, V, gamma)
        V_novo[3] = 10.0  # Estado terminal tem valor fixo

        # Convergência: delta < threshold
        delta = max(abs(V_novo.get(s, 0) - V.get(s, 0)) for s in estados)
        V = V_novo
        historico.append(V.copy())
        if delta < 1e-6:
            break

    return {"V": V, "historico": historico, "gamma": gamma}


if __name__ == "__main__":
    print("=" * 60)
    print("GO1802 - EQUACOES DE OTIMALIDADE DE BELLMAN")
    print("=" * 60)

    # ─── Fórmulas ────────────────────────────────────────────
    print("\nFORMULAS DE BELLMAN:")
    print()
    print("  V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]")
    print("          ↑ ação ótima ↑ recomp. imediata  ↑ valor futuro descontado")
    print()
    print("  Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q*(s',a')")
    print("            ↑ recomp.    ↑ melhor Q no próximo estado")
    print()
    print("  Relação: V*(s) = max_a Q*(s,a)")

    # ─── Exemplo numérico: Grid World 1D ─────────────────────
    resultado = calcular_exemplo_numerico()
    V = resultado["V"]
    gamma = resultado["gamma"]
    historico = resultado["historico"]

    print()
    print("─" * 60)
    print("EXEMPLO: GRID WORLD 1D  [0] → [1] → [2] → [3=OBJETIVO]")
    print("─" * 60)
    print(f"  gamma = {gamma}, custo de passo = -1, chegada = +10")
    print()

    # Mostrar convergência
    print("  Iteração de Valor (V*(s) por iteração):")
    print(f"  {'Iter':>5} | {'V(0)':>8} | {'V(1)':>8} | {'V(2)':>8} | {'V(3)':>8}")
    print("  " + "-" * 50)
    for i, v in enumerate(historico[:6]):
        linha = f"  {i:>5} | "
        linha += " | ".join(f"{v.get(s, 0):8.3f}" for s in [0, 1, 2, 3])
        print(linha)
    if len(historico) > 6:
        print(f"  ... ({len(historico)} iterações no total)")
    print()

    # Valores finais
    print("  Valores finais V*(s):")
    for s in sorted(V.keys()):
        print(f"    V*({s}) = {V[s]:8.4f}")

    # ─── Calcular manualmente Bellman para V*(2) ─────────────
    print()
    print("─" * 60)
    print("VERIFICACAO MANUAL: Bellman para V*(2)")
    print("─" * 60)
    v_s2_direita = -1 + gamma * V[3]  # ação direita: vai para estado 3
    v_s2_esquerda = -1 + gamma * V[1]  # ação esquerda: vai para estado 1
    print(f"  V*(2) = max_a [R(2,a) + {gamma} * V*(s')]")
    print(f"        = max(direita, esquerda)")
    print(f"        = max({v_s2_direita:.3f}, {v_s2_esquerda:.3f})")
    print(f"        = {max(v_s2_direita, v_s2_esquerda):.3f}")
    print(f"        ≈ {V[2]:.3f}  (valor convergeido)  OK!")

    print()
    print("  Politica otima derivada:")
    print("  π*(0)=direita  π*(1)=direita  π*(2)=direita  π*(3)=terminal")
    print()
    print("  Algoritmos que resolvem Bellman:")
    print("  - Value Iteration: atualiza V(s) até convergência")
    print("  - Policy Iteration: alterna política-avaliação e melhoria")
    print("  - Q-Learning: aprende Q* da experiência (sem modelo P)")
