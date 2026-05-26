"""
GO1814 - PPO: Proximal Policy Optimization com Clipping
=========================================================
Demonstra a função objetivo com clipping do PPO com exemplos numéricos.
Requer apenas numpy.

PPO (Schulman et al., 2017 — OpenAI) resolve o problema de Policy Gradient:
"Quanto atualizar a política a cada step sem destruir o aprendizado?"

Problema: gradientes muito grandes → política muda demais → instabilidade.
Solução PPO: CLIPAR a razão de probabilidades para limitar o update.

Função objetivo PPO (CLIP):
  ratio = π_θ_new(a|s) / π_θ_old(a|s)
  L^CLIP(θ) = min(ratio * A(s,a), clip(ratio, 1-ε, 1+ε) * A(s,a))

Interpretação: o clipping garante que o ratio não fique muito longe de 1.0
(a nova política não diverge demais da antiga por episódio).
"""

import numpy as np


def ppo_clipped_objective(
    ratio: float,
    advantage: float,
    epsilon: float = 0.2,
) -> dict:
    """
    Calcula o objetivo PPO clippado para um par (ratio, advantage).

    ratio     = π_new(a|s) / π_old(a|s)
    advantage = A(s,a) — estimado pelo Critic
    epsilon   = limite de clipping (0.2 é o padrão do paper)
    """
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

    termo_nao_clippado = ratio * advantage
    termo_clippado = clipped_ratio * advantage

    L_clip = min(termo_nao_clippado, termo_clippado)

    return {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "advantage": advantage,
        "L_nao_clippado": termo_nao_clippado,
        "L_clippado": termo_clippado,
        "L_clip": L_clip,
        "clipping_ativo": ratio != clipped_ratio,
    }


def demonstrar_cenarios_ppo(epsilon: float = 0.2) -> None:
    """
    Demonstra os 4 cenários possíveis do clipping PPO.
    """
    cenarios = [
        # (ratio, advantage, descricao)
        (0.7,  +2.0, "ratio < 1-ε, advantage > 0: update conservador (ação boa)"),
        (1.5,  +2.0, "ratio > 1+ε, advantage > 0: clipping ativo (impede update excessivo)"),
        (1.5,  -2.0, "ratio > 1+ε, advantage < 0: update conservador (ação ruim)"),
        (0.7,  -2.0, "ratio < 1-ε, advantage < 0: clipping ativo (impede update excessivo)"),
        (1.0,  +1.5, "ratio = 1.0: sem mudança na política"),
        (1.15, +1.5, "ratio dentro do intervalo [0.8, 1.2]: normal"),
    ]

    print(f"\n  ε = {epsilon} → intervalo [{1-epsilon:.1f}, {1+epsilon:.1f}]")
    print()
    print(f"  {'ratio':>6} | {'Adv':>6} | {'Ratio_clip':>10} | "
          f"{'L(nao-clip)':>11} | {'L(clip)':>8} | {'L_PPO':>7} | Clip?")
    print("  " + "-" * 80)

    for ratio, adv, desc in cenarios:
        res = ppo_clipped_objective(ratio, adv, epsilon)
        clip_str = "SIM" if res["clipping_ativo"] else "---"
        print(f"  {ratio:>6.2f} | {adv:>6.1f} | {res['clipped_ratio']:>10.2f} | "
              f"{res['L_nao_clippado']:>11.3f} | {res['L_clippado']:>8.3f} | "
              f"{res['L_clip']:>7.3f} | {clip_str}")


def visualizar_curva_ppo(epsilon: float = 0.2) -> None:
    """
    Mostra a curva de L^CLIP em função do ratio para advantage > 0 e < 0.
    """
    ratios = np.linspace(0.4, 1.6, 13)

    print()
    print(f"  CURVA L^CLIP vs ratio (advantage = +2.0):")
    print(f"  (valores clippados marcados com *)")
    print()
    for r in ratios:
        res = ppo_clipped_objective(float(r), 2.0, epsilon)
        marker = "*" if res["clipping_ativo"] else " "
        barra = "#" * int(abs(res["L_clip"]) * 5)
        print(f"  ratio={r:.2f}{marker} | L={res['L_clip']:+6.2f} | {barra}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1814 - PPO: CLIPPING DO OBJETIVO")
    print("=" * 60)

    print("\nFORMULA PPO CLIP:")
    print()
    print("  ratio = π_θ_new(a|s) / π_θ_old(a|s)")
    print("  ratio_clip = clip(ratio, 1-ε, 1+ε)   # ε=0.2 tipico")
    print()
    print("  L^CLIP(θ) = min(ratio * A(s,a), ratio_clip * A(s,a))")
    print()
    print("  O 'min' garante update CONSERVADOR:")
    print("  - Advantage > 0: não deixa ratio crescer demais (1+ε)")
    print("  - Advantage < 0: não deixa ratio cair demais (1-ε)")

    # ─── Cenários ────────────────────────────────────────────
    print()
    print("─" * 60)
    print("CENARIOS DE CLIPPING (ε = 0.2):")
    print("─" * 60)
    demonstrar_cenarios_ppo(epsilon=0.2)

    # ─── Curva visual ─────────────────────────────────────────
    print()
    print("─" * 60)
    print("VISUALIZACAO DA CURVA L^CLIP:")
    print("─" * 60)
    visualizar_curva_ppo(epsilon=0.2)

    print()
    print("─" * 60)
    print("POR QUE PPO E TAN POPULAR?")
    print("─" * 60)
    print()
    print("  Algoritmos anteriores:")
    print("  - REINFORCE: alta variancia, lento")
    print("  - TRPO (Trust Region Policy Opt.): eficaz mas complexo")
    print()
    print("  PPO: simples + eficaz + estavel = melhor tradeoff (2017)")
    print("  Usado em: ChatGPT RLHF, robótica, jogos, finančas")
    print()
    print("  PPO requer:")
    print("  1. Coletar dados com π_old")
    print("  2. Fazer K epochs com os mesmos dados (eficiencia)")
    print("  3. Calcular ratio = π_new / π_old para cada acao")
    print("  4. Aplicar clipping + atualizar redes")
    print()
    print("  Ver stable-baselines3 para PPO pronto (GO1820).")
