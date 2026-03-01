# GO1431-NãoRequerInstalaçãoDeErro
# ══════════════════════════════════════════════════════════════════
# TIME SERIES ANOMALY DETECTION - LSTM AUTOENCODER
# Detectar anomalias em séries temporais
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt

print("🚨 TIME SERIES ANOMALY DETECTION")
print("=" * 70)

# ─── 1. GERAR SÉRIE TEMPORAL ───
print("\n📊 Gerando série temporal com anomalias...")

np.random.seed(42)

def generate_timeseries(n_samples=500, seq_length=50):
    # Série normal: onda senoidal + ruído
    t = np.linspace(0, 10, seq_length)
    normal_series = []

    for _ in range(n_samples):
        freq = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)

        series = amplitude * np.sin(2 * np.pi * freq * t + phase)
        series += np.random.normal(0, 0.1, seq_length)

        normal_series.append(series)

    return np.array(normal_series)

def generate_anomalies(n_samples=50, seq_length=50):
    # Anomalias: spikes, platôs, tendências abruptas
    anomalies = []

    for _ in range(n_samples):
        anom_type = np.random.choice(['spike', 'plateau', 'trend'])

        if anom_type == 'spike':
            series = np.random.normal(0, 0.2, seq_length)
            spike_pos = np.random.randint(10, seq_length-10)
            series[spike_pos:spike_pos+5] += np.random.uniform(3, 5)

        elif anom_type == 'plateau':
            series = np.sin(np.linspace(0, 10, seq_length))
            plateau_start = np.random.randint(10, seq_length-20)
            series[plateau_start:plateau_start+15] = np.random.uniform(2, 3)

        else:  # trend
            series = np.sin(np.linspace(0, 10, seq_length))
            trend_start = np.random.randint(10, seq_length-20)
            series[trend_start:] += np.linspace(0, 3, seq_length-trend_start)

        anomalies.append(series)

    return np.array(anomalies)

X_normal = generate_timeseries(800)
X_anomaly = generate_anomalies(100)

print(f"  Normal series: {X_normal.shape}")
print(f"  Anomalies: {X_anomaly.shape}")

# Reshape para LSTM: (samples, timesteps, features)
X_normal = X_normal.reshape(-1, 50, 1)
X_anomaly = X_anomaly.reshape(-1, 50, 1)

# Split
from sklearn.model_selection import train_test_split

X_train, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)

print(f"  Train: {X_train.shape}")
print(f"  Test normal: {X_test_normal.shape}")

# ─── 2. LSTM AUTOENCODER ───
print("\n🏗️ Construindo LSTM Autoencoder...")

seq_length = 50
latent_dim = 32

# Encoder
inputs = Input(shape=(seq_length, 1))
encoded = LSTM(latent_dim, activation='relu')(inputs)

# Decoder
decoded = RepeatVector(seq_length)(encoded)
decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, outputs, name='LSTM_Autoencoder')

autoencoder.compile(optimizer='adam', loss='mse')

print(f"  Parâmetros: {autoencoder.count_params():,}")

# ─── 3. TREINAR (SÓ COM NORMAIS) ───
print("\n🚀 Treinando com séries normais...")

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"  Final loss: {history.history['loss'][-1]:.6f}")

# ─── 4. DETECTAR ANOMALIAS ───
print("\n🔍 Detectando anomalias...")

# Reconstruir
recon_normal = autoencoder.predict(X_test_normal, verbose=0)
recon_anomaly = autoencoder.predict(X_anomaly, verbose=0)

# Erro de reconstrução
error_normal = np.mean((X_test_normal - recon_normal)**2, axis=(1,2))
error_anomaly = np.mean((X_anomaly - recon_anomaly)**2, axis=(1,2))

print(f"  MSE normal: {error_normal.mean():.6f} ± {error_normal.std():.6f}")
print(f"  MSE anomaly: {error_anomaly.mean():.6f} ± {error_anomaly.std():.6f}")

# Threshold
threshold = error_normal.mean() + 3 * error_normal.std()

print(f"  Threshold: {threshold:.6f}")

# Classificar
pred_normal = error_normal > threshold
pred_anomaly = error_anomaly > threshold

false_positive_rate = pred_normal.mean()
true_positive_rate = pred_anomaly.mean()

print(f"  False positives: {false_positive_rate*100:.1f}%")
print(f"  True positives (anomalies detected): {true_positive_rate*100:.1f}%")

# ─── 5. VISUALIZAR ───
print("\n📊 Visualizando detecção...")

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Normais
for i in range(3):
    axes[0, i].plot(X_test_normal[i].squeeze(), label='Original', linewidth=2)
    axes[0, i].plot(recon_normal[i].squeeze(), label='Reconstruction', linewidth=2, linestyle='--')
    axes[0, i].set_title(f'Normal (MSE={error_normal[i]:.4f})', fontsize=10, color='green')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)

# Anômalas
for i in range(3):
    axes[1, i].plot(X_anomaly[i].squeeze(), label='Original', linewidth=2)
    axes[1, i].plot(recon_anomaly[i].squeeze(), label='Reconstruction', linewidth=2, linestyle='--')
    axes[1, i].set_title(f'Anomaly (MSE={error_anomaly[i]:.4f})', fontsize=10, color='red')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle('LSTM Autoencoder: Anomaly Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('timeseries_anomaly_detection.png', dpi=150)
print("✅ Detecção salva: timeseries_anomaly_detection.png")

# Distribuição de erros
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(error_normal, bins=30, alpha=0.7, label='Normal', color='green')
ax.hist(error_anomaly, bins=30, alpha=0.7, label='Anomaly', color='red')
ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timeseries_anomaly_distribution.png', dpi=150)
print("✅ Distribuição salva: timeseries_anomaly_distribution.png")

print("\n💡 TIME SERIES ANOMALY DETECTION:")
print("  • LSTM Autoencoder: Capturar padrões temporais")
print("  • Unsupervised: Treinar só com dados normais")
print("  • Reconstruction Error: Alto erro = anomalia")
print("  • Threshold: Baseado em estatísticas do normal")

print("\n🎯 TIPOS DE ANOMALIAS:")
print("  • Point Anomaly: Valor isolado anormal")
print("  • Contextual: Anormal em contexto específico")
print("  • Collective: Sequência de pontos anormais")

print("\n🏆 APLICAÇÕES:")
print("  • IoT: Sensores industriais, falhas em equipamentos")
print("  • Finance: Fraude, trading anormal")
print("  • Healthcare: Sinais vitais, ECG anormal")
print("  • Network: Detecção de intrusão")

print("\n✅ TIME SERIES ANOMALY DETECTION COMPLETO!")
