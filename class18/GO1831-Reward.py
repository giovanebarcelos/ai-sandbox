"""
GO1831 - Recompensa para Ventilação Mecânica (RL Médico)
=========================================================
Demonstra função de recompensa para controle de ventilador mecânico.
Requer apenas numpy.

Contexto: RL aplicado à medicina — controlar ventilador mecânico de UTI
para otimizar parâmetros de ventilação (PEEP, FiO2, frequência, volume corrente).

Objetivo: maximizar oxigenação (SpO2) enquanto minimiza dano pulmonar
(barotrauma) e tempo em ventilação (quanto mais cedo sair, melhor).

Referência: Prasad et al. 2017 - "A Reinforcement Learning Approach
to Weaning of Mechanical Ventilation in Intensive Care Units"
"""

import numpy as np


def calcular_recompensa_ventilacao(
    spo2_melhora: float,       # Δ SpO2 (%) — quanto a oxigenação melhorou
    risco_dano_pulmao: float,  # [0, 1] — risco de barotrauma
    dias_ventilacao: int,      # Número de dias em ventilação
    sucesso_desmame: bool,     # Conseguiu retirar ventilador
    paciente_vivo: bool = True,
) -> dict:
    """
    Calcula a recompensa para controle de ventilação mecânica.

    Recompensa multi-objetivo:
    + SpO2_improvement : aumentar saturação de oxigênio
    - lung_damage_risk : evitar barotrauma (pressão excessiva)
    - ventilator_days  : minimizar tempo em ventilação
    + 100 if weaning_success : sucesso em retirar ventilação
    """
    if not paciente_vivo:
        return {"total": -1000.0, "nota": "Paciente não sobreviveu"}

    r_spo2 = spo2_melhora * 10.0           # Escalar o impacto da oxigenação
    r_dano = -risco_dano_pulmao * 50.0     # Penalizar dano pulmonar fortemente
    r_dias = -dias_ventilacao * 0.5        # Pequena penalidade diária
    r_desmame = 100.0 if sucesso_desmame else 0.0

    total = r_spo2 + r_dano + r_dias + r_desmame

    return {
        "spo2": r_spo2,
        "dano_pulmao": r_dano,
        "dias_ventilacao": r_dias,
        "sucesso_desmame": r_desmame,
        "total": total,
    }


def simular_trajetoria_paciente(dias: int = 7) -> None:
    """Simula a evolução de um paciente em ventilação."""
    np.random.seed(42)
    spo2_base = 88.0

    print("  Evolução simulada do paciente (7 dias):")
    print(f"  {'Dia':>4} | {'SpO2':>6} | {'Δ SpO2':>7} | "
          f"{'Risco':>7} | {'Reward':>8}")
    print("  " + "-" * 45)

    total_reward = 0
    for dia in range(1, dias + 1):
        delta_spo2 = np.random.uniform(-0.5, 1.5)  # Tendência de melhora
        spo2 = spo2_base + delta_spo2
        risco = max(0, np.random.uniform(0.1, 0.4) - dia * 0.02)  # Diminui com tempo
        sucesso = dia == dias and spo2 > 95

        comp = calcular_recompensa_ventilacao(
            spo2_melhora=max(0, delta_spo2),
            risco_dano_pulmao=risco,
            dias_ventilacao=dia,
            sucesso_desmame=sucesso,
        )
        total_reward += comp["total"]
        desmame_str = " DESMAME!" if sucesso else ""
        print(f"  {dia:>4} | {spo2:>6.1f} | {delta_spo2:>+7.2f} | "
              f"{risco:>7.3f} | {comp['total']:>+8.2f}{desmame_str}")
        spo2_base = spo2

    print(f"\n  Recompensa total: {total_reward:+.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1831 - RECOMPENSA PARA VENTILACAO MECANICA")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = + SpO2_improvement  # Aumentar saturacao de O2")
    print("           - lung_damage_risk  # Evitar barotrauma")
    print("           - ventilator_days   # Minimizar tempo em ventilacao")
    print("           + 100 if weaning_success  # Sucesso no desmame")

    print()
    print("─" * 60)
    print("CENARIOS:")
    print("─" * 60)

    cenarios = [
        ("SpO2 melhorou (+2%), baixo risco",  2.0, 0.1, 3, False),
        ("SpO2 estável, alto risco dano",     0.0, 0.9, 5, False),
        ("Desmame bem-sucedido!",             1.0, 0.1, 7, True),
        ("SpO2 piorou, dia 1",               -1.0, 0.3, 1, False),
    ]

    for desc, spo2, risco, dias, desmame in cenarios:
        comp = calcular_recompensa_ventilacao(spo2, risco, dias, desmame)
        print(f"\n  [{desc}]")
        for k, v in comp.items():
            if k not in ("total", "nota"):
                print(f"    {k:20s}: {v:+.2f}")
        print(f"    {'TOTAL':20s}: {comp['total']:+.2f}")

    print()
    print("─" * 60)
    print("SIMULACAO (7 DIAS):")
    print("─" * 60)
    print()
    simular_trajetoria_paciente(dias=7)

    print()
    print("─" * 60)
    print("DESAFIOS DO RL NA MEDICINA:")
    print("─" * 60)
    print("  + Potencial de personalizar tratamento por paciente")
    print("  + Aprende padrões em grandes bases de dados de UTI")
    print("  - Dados escassos (poucos pacientes = poucas amostras)")
    print("  - Causalidade vs correlação nos dados históricos")
    print("  - Regulação: decisões devem ser explicáveis (XAI)")
    print("  - Off-policy: treinar em histórico sem testar em pacientes reais")
