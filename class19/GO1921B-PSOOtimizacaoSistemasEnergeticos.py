# GO1921B-PSOOtimizacaoSistemasEnergeticos
def wind_farm_objective(positions):
    """
    Otimizar layout parque eólico (10 turbinas)

    positions: [x1, y1, x2, y2, ..., x10, y10] (20 dimensões)

    Critérios:
    - Maximizar energia (espalhadas captura mais vento)
    - Evitar wake (turbina a jusante perde 30-50% eficiência)
    - Distância mínima 200m entre turbinas
    """
    n_turbines = 10
    coords = positions.reshape(n_turbines, 2)

    # Energia base (simplificado)
    total_energy = 0
    for i in range(n_turbines):
        # Energia base de cada turbina: 2.5 MW
        energy_i = 2.5

        # Reduzir energia por wake (turbinas downwind)
        for j in range(n_turbines):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])

                # Se turbina j está upwind de i (simplificado: coordenada x)
                if coords[j, 0] < coords[i, 0] and dist < 500:  # 500m raio wake
                    wake_loss = 0.3 * (1 - dist / 500)  # Perda 0-30%
                    energy_i *= (1 - wake_loss)

        total_energy += energy_i

    # Penalizar distâncias < 200m
    penalty = 0
    for i in range(n_turbines):
        for j in range(i+1, n_turbines):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 200:
                penalty += 1000 * (200 - dist)

    # Objetivo: MAXIMIZAR energia → minimizar NEGATIVO
    return -total_energy + penalty

# Área do parque: 2km x 2km
bounds = [(0, 2000), (0, 2000)] * 10  # 10 turbinas (x, y)

pso_wind = PSO(wind_farm_objective, bounds, n_particles=60, max_iter=150, w=0.5, c1=2.0, c2=2.0)
best_layout, best_obj = pso_wind.optimize()

coords = best_layout.reshape(10, 2)
energia_total_mw = -best_obj

print(f"💨 Layout Ótimo Parque Eólico:")
print(f"  Energia total: {energia_total_mw:.2f} MW")
print(f"  Energia/turbina: {energia_total_mw / 10:.2f} MW (vs 2.5 MW ideal)")
print(f"  Eficiência: {energia_total_mw / 25 * 100:.1f}%")

# Visualizar layout
plt.figure(figsize=(8, 8))
plt.scatter(coords[:, 0], coords[:, 1], s=500, c='blue', alpha=0.6, edgecolors='black')
for i, (x, y) in enumerate(coords):
    plt.text(x, y, f'T{i+1}', ha='center', va='center', fontsize=12, color='white', weight='bold')

# Desenhar círculos de wake (500m raio)
for x, y in coords:
    circle = plt.Circle((x, y), 500, color='red', fill=False, linestyle='--', alpha=0.3)
    plt.gca().add_patch(circle)

plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.xlabel('X (metros)')
plt.ylabel('Y (metros)')
plt.title('Layout Parque Eólico Otimizado (PSO)')
plt.grid(alpha=0.3)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
