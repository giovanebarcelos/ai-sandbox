"""
GO1833 - Recompensa para Otimização de Data Center
====================================================
Demonstra a função de recompensa para otimização de data center com RL.
Requer apenas numpy.

A recompensa balanceia múltiplos objetivos:
  - Latência: resposta rápida para os clientes
  - Custo energético: reduzir PUE e consumo total
  - SLA (Service Level Agreement): cumprir contratos de disponibilidade
  - Eficiência de recursos: não desperdiçar capacidade computacional
"""

import numpy as np


def calcular_recompensa_datacenter(
    latencia_ms: float,            # Latência média das requisições (ms)
    custo_energia: float,          # Custo de energia normalizado [0, 1]
    sla_violations: int,           # Número de violações de SLA
    resource_efficiency: float,    # Eficiência [0, 1]
) -> dict:
    """
    Calcula recompensa multi-objetivo para gestão de data center.

    Pesos refletem prioridades:
    - SLA é crítico (peso 100): violações são catastróficas
    - Latência (peso 10): impacto direto no usuário
    - Custo (peso 1): importante mas secundário
    - Eficiência (peso 1): bônus por uso racional
    """
    r_latencia = -latencia_ms * 10.0 / 1000.0    # Escalar para [-1, 0] range
    r_custo = -custo_energia * 1.0
    r_sla = -sla_violations * 100.0
    r_eficiencia = resource_efficiency * 1.0

    total = r_latencia + r_custo + r_sla + r_eficiencia

    return {
        "latencia": r_latencia,
        "custo_energia": r_custo,
        "sla_violations": r_sla,
        "resource_efficiency": r_eficiencia,
        "total": total,
    }


def simular_ciclo_operacional(n_ciclos: int = 8) -> None:
    """Simula um ciclo diário de operação do data center."""
    np.random.seed(7)
    print("  Simulação de ciclo operacional (horário):")
    print(f"  {'Hora':>6} | {'Latência':>9} | {'Custo':>7} | "
          f"{'SLAs':>5} | {'Efic.':>6} | {'Reward':>8}")
    print("  " + "-" * 55)

    total_reward = 0
    horas = [f"{h:02d}:00" for h in range(8, 8 + n_ciclos)]
    cargas = [40, 65, 80, 95, 90, 70, 55, 35]  # % de carga por hora

    for hora, carga in zip(horas, cargas):
        latencia = carga * 0.3 + np.random.randn() * 5
        custo = carga / 100.0 + np.random.randn() * 0.05
        sla_v = max(0, int((carga - 85) / 5 + np.random.poisson(0.5)))
        efic = 0.7 + (carga / 200.0) + np.random.randn() * 0.05

        comp = calcular_recompensa_datacenter(
            max(5, latencia), max(0, min(1, custo)),
            sla_v, max(0, min(1, efic)),
        )
        total_reward += comp["total"]
        flag = " !" if sla_v > 0 else ""
        print(f"  {hora:>6} | {latencia:>8.1f}ms | {custo:>7.3f} | "
              f"{sla_v:>5} | {efic:>6.3f} | {comp['total']:>+8.2f}{flag}")

    print(f"\n  Recompensa acumulada: {total_reward:+.2f}")
    print(f"  (! = violação de SLA)")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1833 - RECOMPENSA PARA DATA CENTER")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = - latency * 10          # Baixa latencia")
    print("           - cost * 1              # Baixo custo energetico")
    print("           - sla_violations * 100  # Evitar quebrar SLA!")
    print("           + resource_efficiency   # Usar bem os recursos")

    print()
    print("─" * 60)
    print("CENARIOS:")
    print("─" * 60)

    cenarios = [
        ("Operacao normal",        20.0, 0.5, 0, 0.8),
        ("Alta carga, eficiente",  50.0, 0.8, 0, 0.9),
        ("Violacao de SLA!",       120.0, 0.9, 3, 0.5),
        ("Subutilizado",            5.0, 0.3, 0, 0.2),
    ]

    for desc, lat, custo, sla, efic in cenarios:
        comp = calcular_recompensa_datacenter(lat, custo, sla, efic)
        print(f"\n  [{desc}]")
        for k, v in comp.items():
            if k != "total":
                print(f"    {k:20s}: {v:+.2f}")
        print(f"    {'TOTAL':20s}: {comp['total']:+.2f}")

    print()
    print("─" * 60)
    print("SIMULACAO CICLO OPERACIONAL:")
    print("─" * 60)
    print()
    simular_ciclo_operacional()
