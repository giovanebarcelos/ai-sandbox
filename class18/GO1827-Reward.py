"""
GO1827 - Função de Recompensa para Robótica (Navegação)
=========================================================
Demonstra e calcula uma função de recompensa multi-objetivo para
navegação robótica. Requer apenas numpy.

Conceito: Reward Shaping — como projetar recompensas que levam o agente
a aprender o comportamento desejado. Recompensas ruins → comportamentos ruins.

Função de recompensa para navegação:
  reward = -distancia_objetivo      (minimizar distância)
           - 10 * colisão           (evitar bater)
           + 100 * chegou_objetivo  (chegar ao destino)
           - 0.01 * tempo           (minimizar tempo)
"""

import numpy as np


def calcular_recompensa_navegacao(
    pos_agente: np.ndarray,
    pos_objetivo: np.ndarray,
    colisao: bool,
    chegou: bool,
    tempo: int,
) -> tuple:
    """
    Calcula a recompensa de navegação robótica.

    Retorna (recompensa_total, componentes).
    """
    distancia = float(np.linalg.norm(pos_agente - pos_objetivo))

    r_distancia = -distancia
    r_colisao = -10.0 if colisao else 0.0
    r_chegou = +100.0 if chegou else 0.0
    r_tempo = -0.01 * tempo

    total = r_distancia + r_colisao + r_chegou + r_tempo

    return total, {
        "distancia": r_distancia,
        "colisao": r_colisao,
        "chegou": r_chegou,
        "tempo": r_tempo,
        "total": total,
    }


def simular_trajetoria() -> None:
    """
    Simula uma trajetória do robô e mostra a evolução da recompensa.
    """
    np.random.seed(42)
    pos_objetivo = np.array([5.0, 5.0])
    pos_atual = np.array([0.0, 0.0])

    print("  Trajetória do robô (10 passos):")
    print(f"  {'Passo':>6} | {'Pos Agente':>15} | {'Dist':>6} | "
          f"{'Colisão':>8} | {'Recomp.':>8}")
    print("  " + "-" * 60)

    for t in range(10):
        # Mover em direção ao objetivo com algum ruído
        direcao = pos_objetivo - pos_atual
        direcao = direcao / (np.linalg.norm(direcao) + 1e-8)
        pos_atual = pos_atual + direcao * 0.7 + np.random.randn(2) * 0.1

        dist = np.linalg.norm(pos_atual - pos_objetivo)
        colisao = np.random.random() < 0.05   # 5% chance de colisão
        chegou = dist < 0.5

        recomp, comp = calcular_recompensa_navegacao(
            pos_atual, pos_objetivo, colisao, chegou, t
        )

        print(f"  {t+1:>6} | ({pos_atual[0]:5.2f}, {pos_atual[1]:5.2f}) | "
              f"{dist:6.3f} | {'SIM' if colisao else 'nao':>8} | {recomp:8.3f}")

        if chegou:
            print("  ** OBJETIVO ALCANCADO! **")
            break


if __name__ == "__main__":
    print("=" * 60)
    print("GO1827 - FUNCAO DE RECOMPENSA: NAVEGACAO ROBOTICA")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = -distancia_objetivo      # Aproximar do destino")
    print("           - 10 * colisao           # Evitar bater")
    print("           + 100 * chegou_objetivo  # Chegou!")
    print("           - 0.01 * tempo           # Minimizar tempo")

    print()
    print("─" * 60)
    print("EXEMPLOS DE CENARIOS:")
    print("─" * 60)

    cenarios = [
        ("Andando normalmente",      np.array([2.0, 1.0]), np.array([5.0, 5.0]), False, False, 10),
        ("Colidiu com obstáculo",    np.array([1.0, 1.0]), np.array([5.0, 5.0]), True,  False,  3),
        ("Chegou ao objetivo!",      np.array([5.1, 5.0]), np.array([5.0, 5.0]), False, True,  50),
        ("Chegou mas demorou muito", np.array([5.1, 5.0]), np.array([5.0, 5.0]), False, True, 500),
    ]

    for desc, pos_ag, pos_obj, col, cheg, t in cenarios:
        recomp, comp = calcular_recompensa_navegacao(pos_ag, pos_obj, col, cheg, t)
        print(f"\n  [{desc}]")
        for nome, val in comp.items():
            if nome != "total":
                print(f"    {nome:10s}: {val:+8.2f}")
        print(f"    {'TOTAL':10s}: {recomp:+8.2f}")

    print()
    print("─" * 60)
    print("TRAJETORIA SIMULADA:")
    print("─" * 60)
    print()
    simular_trajetoria()

    print()
    print("─" * 60)
    print("BOAS PRATICAS NO DESIGN DE RECOMPENSAS:")
    print("─" * 60)
    print("  + Normalizar escalas: componentes na mesma ordem de magnitude")
    print("  + Recompensas esparsas vs densas: densas facilitam aprendizado")
    print("  + Evitar reward hacking: agente maximize recompensa de formas inesperadas")
    print("  + Ex: 'sobreviver' sem penalidade de tempo -> agente fica parado")
