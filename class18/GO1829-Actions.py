"""
GO1829 - Ações para Direção Autônoma
======================================
Demonstra o espaço de ações para um agente de direção autônoma.
Requer apenas numpy.

Conceito: o espaço de ações define O QUE o agente pode fazer.
Um bom espaço de ações:
  - Cobre todos os comportamentos necessários
  - Não tem ações redundantes (desperdício de capacidade)
  - É tratável (poucas ações facilita o aprendizado)

Para direção autônoma: ações discretas são mais simples de aprender;
ações contínuas (esterço, aceleração exata) são mais realistas.
"""

import numpy as np


ACOES_DISCRETAS = {
    0: "stay_lane",
    1: "change_left",
    2: "change_right",
    3: "accelerate",
    4: "brake",
}

ACOES_DESCRICOES = {
    "stay_lane":     "Manter faixa e velocidade atual",
    "change_left":   "Sinalizar e mudar para faixa esquerda",
    "change_right":  "Sinalizar e mudar para faixa direita",
    "accelerate":    "Aumentar velocidade +5 km/h (até 120)",
    "brake":         "Reduzir velocidade -10 km/h (até 0)",
}


def aplicar_acao(estado: dict, acao: int) -> tuple:
    """
    Aplica a ação ao estado e retorna (novo_estado, recompensa, done).
    """
    novo_estado = estado.copy()
    recompensa = 0.0

    nome_acao = ACOES_DISCRETAS[acao]

    if nome_acao == "stay_lane":
        recompensa = 0.1  # Pequena recompensa por estabilidade

    elif nome_acao == "change_left":
        if estado["left_lane_clear"] and estado["lane"] < 3:
            novo_estado["lane"] += 1
            recompensa = 0.5  # Ultrapassou carro lento
        else:
            recompensa = -1.0  # Tentou mudar sem espaço

    elif nome_acao == "change_right":
        if estado["right_lane_clear"] and estado["lane"] > 1:
            novo_estado["lane"] -= 1
            recompensa = 0.2  # Liberou faixa rápida
        else:
            recompensa = -1.0

    elif nome_acao == "accelerate":
        novo_estado["ego_speed"] = min(estado["ego_speed"] + 5, 120.0)
        # Recompensa por alcançar velocidade alvo
        diff_novo = abs(novo_estado["ego_speed"] - estado["target_speed"])
        diff_ant = abs(estado["ego_speed"] - estado["target_speed"])
        recompensa = 0.3 if diff_novo < diff_ant else -0.1

    elif nome_acao == "brake":
        novo_estado["ego_speed"] = max(estado["ego_speed"] - 10, 0.0)
        # Penalizar frenagem desnecessária, recompensar frenagem de segurança
        if estado["front_car_distance"] < 20:
            recompensa = +2.0  # Frenagem de emergência necessária
        else:
            recompensa = -0.5  # Frenagem desnecessária

    # Penalidade de colisão
    done = False
    if novo_estado["front_car_distance"] < 5:
        recompensa = -50.0
        done = True

    return novo_estado, recompensa, done


if __name__ == "__main__":
    print("=" * 60)
    print("GO1829 - ACOES PARA DIRECAO AUTONOMA")
    print("=" * 60)

    print("\nESPACO DE ACOES (Discrete):")
    print()
    for idx, nome in ACOES_DISCRETAS.items():
        print(f"  Acao {idx}: {nome:15s} — {ACOES_DESCRICOES[nome]}")

    # Demonstrar cada ação em um estado inicial
    estado_inicial = {
        "ego_speed": 65.0,
        "lane": 2,
        "front_car_distance": 50.0,
        "front_car_speed": 55.0,
        "left_lane_clear": True,
        "right_lane_clear": False,
        "target_speed": 70.0,
    }

    print()
    print("─" * 60)
    print("RESULTADO DE CADA ACAO NO ESTADO ATUAL:")
    print("─" * 60)
    print(f"\n  Estado atual: vel={estado_inicial['ego_speed']} km/h, "
          f"faixa={estado_inicial['lane']}, "
          f"carro_frente={estado_inicial['front_car_distance']}m")

    print()
    print(f"  {'Acao':>20} | {'Nova vel':>9} | {'Nova faixa':>11} | "
          f"{'Recomp.':>8} | Resultado")
    print("  " + "-" * 70)

    for idx in sorted(ACOES_DISCRETAS.keys()):
        novo_est, recomp, done = aplicar_acao(estado_inicial, idx)
        nome = ACOES_DISCRETAS[idx]
        print(f"  {nome:>20} | {novo_est['ego_speed']:>8.1f} | "
              f"{novo_est['lane']:>11} | {recomp:>+8.2f} | "
              f"{'COLISÃO!' if done else 'OK'}")

    print()
    print("─" * 60)
    print("ACOES DISCRETAS vs CONTINUAS:")
    print("─" * 60)
    print()
    print("  Discretas (aqui):")
    print("  + Simples de aprender (Q-Learning, DQN)")
    print("  - Menos realistas (velocidade muda em degraus)")
    print()
    print("  Contínuas (PPO, SAC, TD3):")
    print("  + Mais realistas: esterco=[−1, +1], aceleracao=[0, 1]")
    print("  - Exige algoritmos de policy gradient")
    print()
    print("  Na prática: veículos autônomos usam ações contínuas")
    print("  (aceleração, freio, esterço) com PPO ou SAC.")
