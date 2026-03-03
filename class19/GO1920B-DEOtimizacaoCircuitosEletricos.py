# GO1920B-DEOtimizacaoCircuitosEletricos
import numpy as np

def opamp_objective(params):
    """Simular amplificador operacional (modelo simplificado)"""
    # 6 parâmetros: W/L ratios dos transistores
    W1_L1, W2_L2, W3_L3, W4_L4, W5_L5, Ibias = params

    # Modelo simplificado (na indústria: simulador SPICE)
    gain_db = 20 * np.log10(W1_L1 * W3_L3 / (W2_L2 + 0.1))  # Simplificado
    power_mw = Ibias * (W1_L1 + W2_L2 + W3_L3) / 10
    area_um2 = (W1_L1 + W2_L2 + W3_L3 + W4_L4 + W5_L5) * 100

    # Restrições (penalidades)
    penalty = 0
    if gain_db < 60:  # Ganho mínimo 60 dB
        penalty += 1000 * (60 - gain_db)
    if power_mw > 5:  # Consumo máximo 5 mW
        penalty += 1000 * (power_mw - 5)

    # Objetivo multi-critério (pesos ajustáveis)
    objective = -gain_db + 10*power_mw + 0.01*area_um2 + penalty

    return objective

# Espaço de busca (μm)
bounds = [
    (1, 50),   # W1/L1
    (1, 50),   # W2/L2
    (1, 100),  # W3/L3 (estágio saída)
    (1, 50),   # W4/L4
    (1, 50),   # W5/L5
    (10, 500)  # Ibias (μA)
]

# Otimizar
best_circuit, best_obj, _ = differential_evolution(
    opamp_objective, bounds, pop_size=40, max_iter=100, F=0.7, CR=0.8
)

print(f"🔌 Melhor projeto de circuito:")
print(f"  W/L ratios: {best_circuit[:5]}")
print(f"  Corrente bias: {best_circuit[5]:.1f} μA")
