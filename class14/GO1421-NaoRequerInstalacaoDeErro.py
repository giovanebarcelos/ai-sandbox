# GO1421-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# TIME SERIES ANOMALY DETECTION COM LSTM
# Detectar anomalias em séries temporais usando reconstruction error
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

print("🚨 TIME SERIES ANOMALY DETECTION COM LSTM")
print("=" * 70)

# ─── 1. GERAR SÉRIE TEMPORAL COM ANOMALIAS ───
print("\n📊 Gerando série temporal sintética...")

np.random.seed(42)
n_points = 1000

# Série temporal normal (senoidal + ruído)
time = np.arange(n_points)
normal_series = 10 * np.sin(2 * np.pi * time / 100) + np.random.randn(n_points) * 0.5

# Injetar anomalias (5% dos pontos)
anomaly_indices = np.random.choice(n_points, size=int(0.05 * n_points), replace=False)
anomaly_series = normal_series.copy()

for idx in anomaly_indices:
    # Spike aleatório (+/- 5 a 10)
    anomaly_series[idx] += np.random.choice([-1, 1]) * np.random.uniform(5, 10)

print(f"  Pontos totais: {n_points}")
print(f"  Anomalias injetadas: {len(anomaly_indices)} ({len(anomaly_indices)/n_points:.1%})")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados...")

# Normalizar
scaler = StandardScaler()
data_scaled = scaler.fit_transform(anomaly_series.reshape(-1, 1))

# Criar sequências
window_size = 20

def create_sequences(data, window):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
    return np.array(X)

X = create_sequences(data_scaled, window_size)

print(f"  X shape: {X.shape}")  # (samples, window_size, 1)

# Split: treinar apenas com dados "normais"
# Na prática, separamos período inicial (sem anomalias conhecidas)
split = int(0.7 * len(X))
X_train = X[:split]
X_test = X[split:]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 3. CONSTRUIR AUTOENCODER LSTM ───
print("\n🏗️ Construindo LSTM Autoencoder...")

# Autoencoder: Encoder → Latent → Decoder → Reconstruction
latent_dim = 10

model = Sequential([
    # ENCODER
    LSTM(32, activation='relu', input_shape=(window_size, 1), return_sequences=True),
    LSTM(latent_dim, activation='relu', return_sequences=False),

    # DECODER
    RepeatVector(window_size),
    LSTM(latent_dim, activation='relu', return_sequences=True),
    LSTM(32, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
], name='LSTM_Autoencoder')

model.compile(optimizer='adam', loss='mse')

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando autoencoder...")

history = model.fit(
    X_train, X_train,  # Autoencoder: input = output
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"  Final loss: {history.history['loss'][-1]:.6f}")

# ─── 5. CALCULAR RECONSTRUCTION ERROR ───
print("\n🔍 Calculando reconstruction error...")

# Reconstruir
X_test_pred = model.predict(X_test, verbose=0)

# Erro de reconstrução (MAE)
reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=(1, 2))

print(f"  Erro shape: {reconstruction_error.shape}")
print(f"  Erro médio: {reconstruction_error.mean():.6f}")
print(f"  Erro std: {reconstruction_error.std():.6f}")

# ─── 6. DEFINIR THRESHOLD PARA ANOMALIAS ───
print("\n🎯 Definindo threshold...")

# Threshold = média + 2.5 * std
threshold = reconstruction_error.mean() + 2.5 * reconstruction_error.std()

print(f"  Threshold: {threshold:.6f}")

# Detectar anomalias
anomalies_detected = reconstruction_error > threshold
n_anomalies_detected = anomalies_detected.sum()

print(f"  Anomalias detectadas: {n_anomalies_detected} ({n_anomalies_detected/len(reconstruction_error):.1%})")

# ─── 7. VISUALIZAR RESULTADOS ───
print("\n📊 Visualizando resultados...")

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Plot 1: Série temporal com anomalias reais
axes[0].plot(time, anomaly_series, label='Série', linewidth=1.5)
axes[0].scatter(anomaly_indices, anomaly_series[anomaly_indices], 
               color='red', label='Anomalias Reais', s=50, zorder=5)
axes[0].set_title('Série Temporal com Anomalias Injetadas', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Tempo')
axes[0].set_ylabel('Valor')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Reconstruction Error
test_time = time[split+window_size:]
axes[1].plot(test_time, reconstruction_error, label='Reconstruction Error', linewidth=1.5)
axes[1].axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
axes[1].fill_between(test_time, 0, threshold, alpha=0.2, color='green', label='Normal')
axes[1].fill_between(test_time, threshold, reconstruction_error.max(), alpha=0.2, color='red', label='Anomaly Zone')
axes[1].set_title('Reconstruction Error (Test Set)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Tempo')
axes[1].set_ylabel('Erro')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: Detecções
axes[2].plot(test_time, anomaly_series[split+window_size:], label='Série', linewidth=1.5)
anomaly_time = test_time[anomalies_detected]
anomaly_values = anomaly_series[split+window_size:][anomalies_detected]
axes[2].scatter(anomaly_time, anomaly_values, color='red', label='Detectadas', s=50, zorder=5)
axes[2].set_title('Anomalias Detectadas pelo Modelo', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Tempo')
axes[2].set_ylabel('Valor')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle('Time Series Anomaly Detection com LSTM Autoencoder', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('anomaly_detection_lstm.png', dpi=150)
print("✅ Resultados salvos: anomaly_detection_lstm.png")

# ─── 8. MÉTRICAS DE AVALIAÇÃO ───
print("\n" + "="*70)
print("📊 MÉTRICAS DE DETECÇÃO")
print("="*70)

# Criar labels reais para test set
test_anomaly_labels = np.isin(np.arange(split+window_size, n_points), anomaly_indices)

# True Positives, False Positives, etc.
tp = np.sum(anomalies_detected & test_anomaly_labels)
fp = np.sum(anomalies_detected & ~test_anomaly_labels)
fn = np.sum(~anomalies_detected & test_anomaly_labels)
tn = np.sum(~anomalies_detected & ~test_anomaly_labels)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"  True Positives: {tp}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Negatives: {tn}")
print(f"\n  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1-Score: {f1:.3f}")

print("\n💡 COMO FUNCIONA:")
print("  1. Treinar autoencoder com dados normais")
print("  2. Modelo aprende a reconstruir padrões normais")
print("  3. Anomalias têm alto erro de reconstrução")
print("  4. Threshold: define fronteira normal/anômalo")

print("\n📚 APLICAÇÕES:")
print("  • Monitoramento de servidores (CPU, memória)")
print("  • Fraude em transações financeiras")
print("  • Detecção de falhas em sensores IoT")
print("  • Segurança (detecção de intrusão)")

print("\n✅ ANOMALY DETECTION COMPLETO!")
