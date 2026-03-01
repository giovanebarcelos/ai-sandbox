# GO0807-16SomParaDetecçãoDeAnomalias
# DETECÇÃO DE ANOMALIAS COM SOM

# Ideia: Pontos distantes do BMU são potenciais anomalias

# CALCULAR DISTÂNCIA QUANTIZATION ERROR

def quantization_error(som, data):
    """Calcula erro de quantização para cada ponto"""
    errors = []
    for x in data:
        winner = som.winner(x)
        winner_weights = som.get_weights()[winner[0], winner[1]]
        error = np.linalg.norm(x - winner_weights)
        errors.append(error)
    return np.array(errors)

qe = quantization_error(som, X)

# IDENTIFICAR ANOMALIAS

# Threshold: média + 2*std (99% dos dados normais)
threshold = qe.mean() + 2 * qe.std()

anomalies_idx = np.where(qe > threshold)[0]
normal_idx = np.where(qe <= threshold)[0]

print("="*60)
print("DETECÇÃO DE ANOMALIAS")
print("="*60)
print(f"Total de pontos: {len(X)}")
print(f"Anomalias detectadas: {len(anomalies_idx)} ({len(anomalies_idx)/len(X)*100:.1f}%)")
print(f"Threshold: {threshold:.3f}")

# VISUALIZAR

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de erros
ax1.hist(qe[normal_idx], bins=50, alpha=0.7, label='Normal', color='blue')
ax1.hist(qe[anomalies_idx], bins=20, alpha=0.7, label='Anomalias', color='red')
ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
ax1.set_xlabel('Quantization Error')
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição do Erro de Quantização')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mapa de anomalias
anomaly_map = np.zeros(som_shape)
normal_map = np.zeros(som_shape)

for idx in anomalies_idx:
    winner = som.winner(X[idx])
    anomaly_map[winner[0], winner[1]] += 1

for idx in normal_idx:
    winner = som.winner(X[idx])
    normal_map[winner[0], winner[1]] += 1

# Plot com transparência para ver sobreposição
ax2.pcolormesh(normal_map.T, cmap='Blues', alpha=0.6, edgecolors='gray', linewidth=0.5)
im = ax2.pcolormesh(anomaly_map.T, cmap='Reds', alpha=0.8, edgecolors='black', linewidth=1)
plt.colorbar(im, ax=ax2, label='Anomalias')
ax2.set_title('Mapa de Anomalias no SOM')
ax2.set_xlabel('Neurônio X')
ax2.set_ylabel('Neurônio Y')

plt.tight_layout()
plt.show()

# ANALISAR ANOMALIAS

print("\n" + "="*60)
print("EXEMPLOS DE ANOMALIAS")
print("="*60)

anomalous_customers = df.iloc[anomalies_idx].head(5)
print("\nTop 5 clientes anômalos:")
print(anomalous_customers[['idade', 'renda', 'gastos_anuais', 'frequencia_compra']])

print("\nPossíveis razões:")
print("  • Renda muito alta/baixa para o perfil")
print("  • Gastos incompatíveis com renda")
print("  • Comportamento de compra atípico")
print("  • Dados incorretos ou fraudulentos")
