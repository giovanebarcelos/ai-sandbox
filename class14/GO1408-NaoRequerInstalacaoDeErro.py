# GO1408-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# DETECÇÃO DE ANOMALIAS EM SÉRIES TEMPORAIS - LSTM AUTOENCODER
# Dataset: Dados de sensores industriais
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ─── 1. GERAR DADOS DE SENSORES COM ANOMALIAS ───
print("🏭 Gerando dados sintéticos de sensores industriais...")

np.random.seed(42)

# Parâmetros
n_samples = 10000
timesteps = 50

# Gerar dados normais (padrão senoidal com ruído)
time = np.arange(n_samples)
normal_data = []

for i in range(n_samples):
    # Padrão senoidal combinado
    sensor1 = 10 * np.sin(2 * np.pi * (i + np.arange(timesteps)) / 100) + np.random.normal(0, 0.5, timesteps)
    sensor2 = 15 * np.cos(2 * np.pi * (i + np.arange(timesteps)) / 150) + np.random.normal(0, 0.7, timesteps)
    sensor3 = 20 * np.sin(2 * np.pi * (i + np.arange(timesteps)) / 200) + np.random.normal(0, 1, timesteps)

    combined = np.column_stack([sensor1, sensor2, sensor3])
    normal_data.append(combined)

normal_data = np.array(normal_data)

print(f"  Dados normais: {normal_data.shape} (samples, timesteps, features)")

# Gerar dados anômalos
n_anomalies = 200
anomaly_data = []
anomaly_indices = []

for i in range(n_anomalies):
    idx = np.random.randint(0, n_samples)
    anomaly_indices.append(idx)

    # Tipos de anomalias
    anomaly_type = np.random.choice(['spike', 'drop', 'drift', 'noise'])

    sensor1 = normal_data[idx, :, 0].copy()
    sensor2 = normal_data[idx, :, 1].copy()
    sensor3 = normal_data[idx, :, 2].copy()

    if anomaly_type == 'spike':
        # Pico repentino
        spike_pos = np.random.randint(10, 40)
        sensor1[spike_pos:spike_pos+5] += np.random.uniform(20, 40)
    elif anomaly_type == 'drop':
        # Queda repentina
        drop_pos = np.random.randint(10, 40)
        sensor2[drop_pos:drop_pos+5] -= np.random.uniform(20, 40)
    elif anomaly_type == 'drift':
        # Deriva gradual
        sensor3 += np.linspace(0, 30, timesteps)
    else:  # noise
        # Ruído excessivo
        sensor1 += np.random.normal(0, 5, timesteps)
        sensor2 += np.random.normal(0, 5, timesteps)

    combined = np.column_stack([sensor1, sensor2, sensor3])
    anomaly_data.append(combined)

anomaly_data = np.array(anomaly_data)

print(f"  Dados anômalos: {anomaly_data.shape}")
print(f"  Índices com anomalias: {len(anomaly_indices)}")

# Inserir anomalias no dataset
data_with_anomalies = normal_data.copy()
for i, idx in enumerate(anomaly_indices):
    data_with_anomalies[idx] = anomaly_data[i]

# Labels (0=normal, 1=anomalia)
labels = np.zeros(n_samples)
labels[anomaly_indices] = 1

print(f"\n  Total de amostras: {len(labels)}")
print(f"  Anomalias: {labels.sum():.0f} ({labels.mean()*100:.2f}%)")

# Visualizar exemplos
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Normal
axes[0, 0].plot(normal_data[0, :, 0], label='Sensor 1', linewidth=2)
axes[0, 0].plot(normal_data[0, :, 1], label='Sensor 2', linewidth=2)
axes[0, 0].plot(normal_data[0, :, 2], label='Sensor 3', linewidth=2)
axes[0, 0].set_title('Dados Normais', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Spike anomaly
spike_idx = anomaly_indices[0]
axes[0, 1].plot(data_with_anomalies[spike_idx, :, 0], label='Sensor 1', linewidth=2)
axes[0, 1].plot(data_with_anomalies[spike_idx, :, 1], label='Sensor 2', linewidth=2)
axes[0, 1].plot(data_with_anomalies[spike_idx, :, 2], label='Sensor 3', linewidth=2)
axes[0, 1].set_title('Anomalia: Spike', fontsize=12, fontweight='bold', color='red')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Drop anomaly
drop_idx = anomaly_indices[50] if len(anomaly_indices) > 50 else anomaly_indices[1]
axes[1, 0].plot(data_with_anomalies[drop_idx, :, 0], label='Sensor 1', linewidth=2)
axes[1, 0].plot(data_with_anomalies[drop_idx, :, 1], label='Sensor 2', linewidth=2)
axes[1, 0].plot(data_with_anomalies[drop_idx, :, 2], label='Sensor 3', linewidth=2)
axes[1, 0].set_title('Anomalia: Drop', fontsize=12, fontweight='bold', color='red')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Drift anomaly
drift_idx = anomaly_indices[100] if len(anomaly_indices) > 100 else anomaly_indices[2]
axes[1, 1].plot(data_with_anomalies[drift_idx, :, 0], label='Sensor 1', linewidth=2)
axes[1, 1].plot(data_with_anomalies[drift_idx, :, 1], label='Sensor 2', linewidth=2)
axes[1, 1].plot(data_with_anomalies[drift_idx, :, 2], label='Sensor 3', linewidth=2)
axes[1, 1].set_title('Anomalia: Drift', fontsize=12, fontweight='bold', color='red')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensor_data_examples.png', dpi=150)
print("\n  ✓ Exemplos salvos: sensor_data_examples.png")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados para LSTM Autoencoder...")

# Normalizar (por feature)
scaler = StandardScaler()
data_reshaped = data_with_anomalies.reshape(-1, 3)
data_scaled = scaler.fit_transform(data_reshaped)
data_scaled = data_scaled.reshape(n_samples, timesteps, 3)

# Split (80% treino - apenas dados normais, 20% teste - com anomalias)
train_size = int(0.8 * n_samples)

# Treino: apenas dados NORMAIS
train_indices = [i for i in range(train_size) if labels[i] == 0]
X_train = data_scaled[train_indices]

# Teste: todos os dados (normais + anômalos)
X_test = data_scaled[train_size:]
y_test = labels[train_size:]

print(f"\n  Treino: {X_train.shape[0]} (apenas normais)")
print(f"  Teste: {X_test.shape[0]} ({y_test.sum():.0f} anomalias)")

# ─── 3. CONSTRUIR LSTM AUTOENCODER ───
print("\n🔨 Construindo LSTM Autoencoder...")

# Encoder
encoder_inputs = Input(shape=(timesteps, 3))
encoder = LSTM(32, activation='relu', return_sequences=True)(encoder_inputs)
encoder = LSTM(16, activation='relu', return_sequences=False)(encoder)

# Decoder
decoder = RepeatVector(timesteps)(encoder)
decoder = LSTM(16, activation='relu', return_sequences=True)(decoder)
decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(3))(decoder)

# Autoencoder completo
autoencoder = Model(inputs=encoder_inputs, outputs=decoder, name='LSTM_Autoencoder')

autoencoder.compile(optimizer='adam', loss='mse')

print(f"  Parâmetros: {autoencoder.count_params():,}")
autoencoder.summary()

# ─── 4. TREINAR AUTOENCODER ───
print("\n🚀 Treinando autoencoder (apenas com dados normais)...")

callbacks = [EarlyStopping(patience=10, restore_best_weights=True, verbose=1)]

history = autoencoder.fit(
    X_train, X_train,  # Input = Output (reconstrução)
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# ─── 5. CALCULAR ERRO DE RECONSTRUÇÃO ───
print("\n📊 Calculando erro de reconstrução no teste...")

# Reconstruir dados
X_test_pred = autoencoder.predict(X_test, verbose=0)

# Calcular erro de reconstrução (MSE por amostra)
reconstruction_errors = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

print(f"  Erro médio (normais): {reconstruction_errors[y_test == 0].mean():.6f}")
print(f"  Erro médio (anomalias): {reconstruction_errors[y_test == 1].mean():.6f}")

# ─── 6. DEFINIR THRESHOLD E DETECTAR ANOMALIAS ───
print("\n🎯 Definindo threshold para detecção...")

# Threshold: percentil 95 dos erros de reconstrução dos dados normais do treino
X_train_pred = autoencoder.predict(X_train, verbose=0)
train_reconstruction_errors = np.mean(np.square(X_train - X_train_pred), axis=(1, 2))
threshold = np.percentile(train_reconstruction_errors, 95)

print(f"  Threshold (95° percentil): {threshold:.6f}")

# Detectar anomalias (erro > threshold)
y_pred = (reconstruction_errors > threshold).astype(int)

# Métricas
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomalia']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusão:")
print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

# ─── 7. VISUALIZAR RESULTADOS ───
print("\n📈 Gerando visualizações...")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 1. Histórico de treinamento
axes[0].plot(history.history['loss'], label='Treino Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validação Loss', linewidth=2)
axes[0].set_title('Histórico de Treinamento - Autoencoder', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Distribuição dos erros de reconstrução
axes[1].hist(reconstruction_errors[y_test == 0], bins=50, alpha=0.7, 
             label='Normal', color='green', edgecolor='black')
axes[1].hist(reconstruction_errors[y_test == 1], bins=50, alpha=0.7, 
             label='Anomalia', color='red', edgecolor='black')
axes[1].axvline(x=threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.4f})')
axes[1].set_title('Distribuição dos Erros de Reconstrução', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Erro de Reconstrução (MSE)')
axes[1].set_ylabel('Frequência')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors)
roc_auc = auc(fpr, tpr)

axes[2].plot(fpr, tpr, color='darkorange', linewidth=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[2].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random')
axes[2].set_title('ROC Curve - Detecção de Anomalias', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Taxa de Falsos Positivos')
axes[2].set_ylabel('Taxa de Verdadeiros Positivos')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_detection_results.png', dpi=150)
print("  ✓ Resultados salvos: anomaly_detection_results.png")

# ─── 8. VISUALIZAR RECONSTRUÇÕES ───
print("\n🔍 Visualizando reconstruções específicas...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Normal detectado corretamente (TN)
normal_idx = np.where((y_test == 0) & (y_pred == 0))[0][0]
axes[0, 0].plot(X_test[normal_idx, :, 0], label='Original', linewidth=2, alpha=0.7)
axes[0, 0].plot(X_test_pred[normal_idx, :, 0], label='Reconstruído', 
                linewidth=2, linestyle='--', alpha=0.7)
axes[0, 0].set_title(f'Normal (TN) - Erro: {reconstruction_errors[normal_idx]:.4f}', 
                     fontsize=12, fontweight='bold', color='green')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Anomalia detectada corretamente (TP)
anomaly_idx = np.where((y_test == 1) & (y_pred == 1))[0]
if len(anomaly_idx) > 0:
    anomaly_idx = anomaly_idx[0]
    axes[0, 1].plot(X_test[anomaly_idx, :, 0], label='Original', linewidth=2, alpha=0.7)
    axes[0, 1].plot(X_test_pred[anomaly_idx, :, 0], label='Reconstruído', 
                    linewidth=2, linestyle='--', alpha=0.7)
    axes[0, 1].set_title(f'Anomalia (TP) - Erro: {reconstruction_errors[anomaly_idx]:.4f}', 
                         fontsize=12, fontweight='bold', color='red')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Falso Positivo (FP)
fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
if len(fp_idx) > 0:
    fp_idx = fp_idx[0]
    axes[1, 0].plot(X_test[fp_idx, :, 0], label='Original', linewidth=2, alpha=0.7)
    axes[1, 0].plot(X_test_pred[fp_idx, :, 0], label='Reconstruído', 
                    linewidth=2, linestyle='--', alpha=0.7)
    axes[1, 0].set_title(f'Falso Positivo (FP) - Erro: {reconstruction_errors[fp_idx]:.4f}', 
                         fontsize=12, fontweight='bold', color='orange')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Falso Negativo (FN)
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
if len(fn_idx) > 0:
    fn_idx = fn_idx[0]
    axes[1, 1].plot(X_test[fn_idx, :, 0], label='Original', linewidth=2, alpha=0.7)
    axes[1, 1].plot(X_test_pred[fn_idx, :, 0], label='Reconstruído', 
                    linewidth=2, linestyle='--', alpha=0.7)
    axes[1, 1].set_title(f'Falso Negativo (FN) - Erro: {reconstruction_errors[fn_idx]:.4f}', 
                         fontsize=12, fontweight='bold', color='purple')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_reconstructions.png', dpi=150)
print("  ✓ Reconstruções salvas: anomaly_reconstructions.png")

# ─── 9. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ DETECÇÃO DE ANOMALIAS CONCLUÍDA!")
print("="*70)

print(f"\n📊 Estatísticas:")
print(f"  Total de amostras teste: {len(y_test)}")
print(f"  Anomalias reais: {y_test.sum():.0f} ({y_test.mean()*100:.2f}%)")
print(f"  Anomalias detectadas: {y_pred.sum()}")

print(f"\n🎯 Performance:")
print(f"  Threshold: {threshold:.6f}")
print(f"  AUC-ROC: {roc_auc:.4f}")
print(f"  Verdadeiros Positivos (TP): {cm[1, 1]}")
print(f"  Falsos Positivos (FP): {cm[0, 1]}")
print(f"  Falsos Negativos (FN): {cm[1, 0]}")

print("\n📁 Arquivos gerados:")
print("  • sensor_data_examples.png - Exemplos de dados normais e anômalos")
print("  • anomaly_detection_results.png - Métricas e ROC curve")
print("  • anomaly_reconstructions.png - Reconstruções específicas")

print("\n💡 Insights:")
print("  • Autoencoder aprende padrão normal durante treino")
print("  • Anomalias têm alto erro de reconstrução")
print("  • Threshold baseado em percentil dos dados de treino")
print("  • Detecta spikes, drops, drifts e ruído excessivo")

print("\n🔧 Melhorias possíveis:")
print("  • Testar Variational Autoencoder (VAE)")
print("  • Usar GRU no lugar de LSTM")
print("  • Implementar attention mechanism")
print("  • Ensemble de múltiplos autoencoders")
print("  • Threshold adaptativo baseado em janela temporal")
