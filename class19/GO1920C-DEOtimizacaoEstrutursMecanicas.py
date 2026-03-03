# GO1920C-DEOtimizacaoEstrutursMecanicas
def truss_objective(cross_sections):
    """
    Otimizar seções transversais de vigas em treliça

    Critérios:
    - Minimizar peso total
    - Resistir carga 100 kN
    - Deflexão < 10 cm
    """
    # Análise elementos finitos simplificada (na indústria: Abaqus/ANSYS)
    weight = np.sum(cross_sections) * 7850 * 10  # densidade aço * comprimento

    # Rigidez (simplificado)
    stiffness = np.sum(cross_sections**2) * 210e9  # Módulo Young aço
    deflection_cm = 100000 / stiffness  # Carga / rigidez

    # Penalidade se deflexão > 10 cm
    penalty = max(0, deflection_cm - 10) * 10000

    return weight + penalty

# 20 vigas na treliça
bounds = [(0.0001, 0.01)] * 20  # Área seção (m²)

best_design, best_weight, _ = differential_evolution(
    truss_objective, bounds, pop_size=60, max_iter=150
)

print(f"✈️ Projeto ótimo de asa:")
print(f"  Peso total: {best_weight:.2f} kg")
print(f"  Seções críticas: {best_design[:5]}")
