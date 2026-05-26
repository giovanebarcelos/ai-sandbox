"""
GO1830 - Função de Recompensa para Direção Autônoma
====================================================
Demonstra e calcula a função de recompensa multi-componente para AD.
Requer apenas numpy.

Projetar recompensas para direção autônoma é difícil porque precisamos
balancear múltiplos objetivos simultaneamente:
1. Velocidade (eficiência)
2. Segurança (sem colisões)
3. Conforto (sem frenadas/acelerações bruscas)
4. Regras (não trocar de faixa desnecessariamente)
5. Eficiência de ultrapassagem

Reward shaping: cada componente tem um peso que reflete sua prioridade.
"""

import numpy as np


def calcular_recompensa_direcao(
    vel_atual: float,
    vel_alvo: float,
    colisao: bool,
    jerk: float,           # Variação brusca de velocidade
    mudou_faixa: bool,
    ultrapassou_carro_lento: bool,
) -> dict:
    """
    Calcula recompensa multi-componente para direção autônoma.

    Retorna dicionário com cada componente e o total.
    """
    # Recompensa por velocidade (quanto mais perto da alvo, melhor)
    diff_vel = abs(vel_atual - vel_alvo) / vel_alvo   # normalizado
    r_velocidade = 1.0 * (1.0 - diff_vel)   # Entre 0 e 1

    # Penalidade por colisão (catastrófica)
    r_colisao = -10.0 if colisao else 0.0

    # Penalidade por movimento brusco (conforto)
    r_jerk = -0.5 * jerk

    # Penalidade por troca de faixa desnecessária
    r_faixa = -0.1 if mudou_faixa else 0.0

    # Bônus por eficiência (ultrapassou carro lento)
    r_ultrapassagem = +0.5 if ultrapassou_carro_lento else 0.0

    total = r_velocidade + r_colisao + r_jerk + r_faixa + r_ultrapassagem

    return {
        "velocidade": r_velocidade,
        "colisao": r_colisao,
        "jerk": r_jerk,
        "troca_faixa": r_faixa,
        "ultrapassagem": r_ultrapassagem,
        "total": total,
    }


def simular_episodio(n_passos: int = 10) -> None:
    """Simula um episódio de direção com variação de situações."""
    np.random.seed(42)
    vel = 60.0
    vel_alvo = 80.0

    print("  Simulação de episódio de direção:")
    print(f"  {'Passo':>5} | {'Vel':>5} | {'Colisão':>8} | "
          f"{'Jerk':>6} | {'Troca':>6} | {'Ultrap':>7} | {'Reward':>8}")
    print("  " + "-" * 60)

    total_reward = 0
    for t in range(n_passos):
        # Variar situação
        vel = min(max(vel + np.random.randn() * 5, 0), 120)
        colisao = t == 4  # Colisão no passo 4
        jerk = abs(np.random.randn()) * 0.5
        mudou_faixa = t in (2, 6, 9)
        ultrapassou = t in (3, 7)

        comp = calcular_recompensa_direcao(
            vel, vel_alvo, colisao, jerk, mudou_faixa, ultrapassou
        )
        total_reward += comp["total"]
        print(f"  {t+1:>5} | {vel:>5.1f} | {'SIM' if colisao else 'nao':>8} | "
              f"{jerk:>6.3f} | {'SIM' if mudou_faixa else 'nao':>6} | "
              f"{'SIM' if ultrapassou else 'nao':>7} | {comp['total']:>+8.3f}")

    print(f"\n  Total: {total_reward:+.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1830 - RECOMPENSA PARA DIRECAO AUTONOMA")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = + 1.0 * approach_target_speed   # Velocidade certa")
    print("           - 10.0 * colisao                # Evitar acidente!")
    print("           - 0.5 * jerk                    # Dirigir suavemente")
    print("           - 0.1 * mudar_faixa_desnec.     # Nao ficar trocando")
    print("           + 0.5 * ultrapassar_carro_lento # Eficiência")

    print()
    print("─" * 60)
    print("CENARIOS:")
    print("─" * 60)

    cenarios = [
        ("Dirigindo bem (na vel. alvo)",    80.0, 80.0, False, 0.1, False, False),
        ("Colisão frontal",                  70.0, 80.0, True,  0.5, False, False),
        ("Frenada brusca (alta jerk)",       50.0, 80.0, False, 2.0, False, False),
        ("Ultrapassagem eficiente",          90.0, 80.0, False, 0.2, True,  True),
        ("Abaixo da velocidade",             40.0, 80.0, False, 0.1, False, False),
    ]

    for desc, vel, vel_alvo, col, jerk, troca, ultra in cenarios:
        comp = calcular_recompensa_direcao(vel, vel_alvo, col, jerk, troca, ultra)
        print(f"\n  [{desc}]")
        for k, v in comp.items():
            if k != "total":
                print(f"    {k:15s}: {v:+.3f}")
        print(f"    {'TOTAL':15s}: {comp['total']:+.3f}")

    print()
    print("─" * 60)
    print("SIMULACAO DO EPISODIO:")
    print("─" * 60)
    print()
    simular_episodio(n_passos=10)
