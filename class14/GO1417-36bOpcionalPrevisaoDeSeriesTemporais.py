# GO1417-36bOpcionalPrevisãoDeSériesTemporais
# ═══════════════════════════════════════════════════════════════════
# PREVISÃO DE SÉRIES TEMPORAIS MULTIVARIADAS COM LSTM
# Prever múltiplas variáveis simultaneamente (temperatura, umidade, pressão)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("📈 PREVISÃO DE SÉRIES TEMPORAIS MULTIVARIADAS")
print("=" * 70)

# ─── 1. GERAR DADOS SINTÉTICOS ───
print("\n🔢 Gerando séries temporais sintéticas...")

# 1000 timesteps
n_steps = 1000
time = np.arange(n_steps)

# Temperatura (ciclo diário + tendência)
temperature = 20 + 10 * np.sin(2 * np.pi * time / 24) + 0.01 * time + np.random.randn(n_steps) * 2

# Umidade (inversamente correlacionada com temperatura)
humidity = 70 - 5 * np.sin(2 * np.pi * time / 24) - 0.005 * time + np.random.randn(n_steps) * 3

# Pressão (ciclo semanal)
pressure = 1013 + 10 * np.sin(2 * np.pi * time / 168) + np.random.randn(n_steps) * 2

# Criar DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'pressure': pressure
})

print(f"  Dataset shape: {df.shape}")
print(f"  Colunas: {list(df.columns)}")
print(f"\n  Estatísticas:")
print(df.describe())

# ─── 2. NORMALIZAR DADOS ───
print("\n🔧 Normalizando dados...")

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

print(f"  Dados normalizados: {df_scaled.shape}")

# ─── 3. CRIAR SEQUÊNCIAS ───
print("\n📦 Criando sequências...")

def create_sequences(data, seq_length, pred_steps=1):
    """
    Cria sequências para treino
    data: array (timesteps, features)
    seq_length: tamanho da janela de entrada
    pred_steps: quantos timesteps prever à frente
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_steps + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_steps])
    return np.array(X), np.array(y)

SEQ_LENGTH = 24  # Usar 24 horas para prever próximas
PRED_STEPS = 6   # Prever próximas 6 horas

data_array = df_scaled.values

X, y = create_sequences(data_array, SEQ_LENGTH, PRED_STEPS)

print(f"  X shape: {X.shape}  # (samples, seq_length, features)")
print(f"  y shape: {y.shape}  # (samples, pred_steps, features)")

# ─── 4. SPLIT TRAIN/TEST ───
train_size = int(0.8 * len(X))

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 5. CONSTRUIR MODELO ───
print("\n🏗️ Construindo modelo LSTM multivariado...")

n_features = X.shape[2]  # 3 features (temp, humidity, pressure)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(PRED_STEPS * n_features),  # Prever múltiplos timesteps e features
], name='MultivariateLSTM')

# Reshape output para (pred_steps, features)
model.add(tf.keras.layers.Reshape((PRED_STEPS, n_features)))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ─── 6. TREINAR ───
print("\n🚀 Treinando modelo...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# ─── 7. AVALIAR ───
print("\n📊 Avaliando modelo...")

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss (MSE): {test_loss:.6f}")
print(f"  Test MAE: {test_mae:.6f}")

# ─── 8. FAZER PREDIÇÕES ───
print("\n🔮 Fazendo predições...")

y_pred = model.predict(X_test)

print(f"  Predições shape: {y_pred.shape}")

# Desnormalizar
y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, n_features)).reshape(y_test.shape)
y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, n_features)).reshape(y_pred.shape)

# ─── 9. VISUALIZAR RESULTADOS ───
print("\n📈 Gerando visualizações...")

# Pegar uma amostra aleatória para visualizar
sample_idx = 50

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

feature_names = ['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)']
colors = ['red', 'blue', 'green']

for i, (feature, color) in enumerate(zip(feature_names, colors)):
    # True values
    axes[i].plot(range(PRED_STEPS), y_test_denorm[sample_idx, :, i], 
                'o-', label='True', color=color, linewidth=2, markersize=8)

    # Predicted values
    axes[i].plot(range(PRED_STEPS), y_pred_denorm[sample_idx, :, i], 
                's--', label='Predicted', color='orange', linewidth=2, markersize=8)

    axes[i].set_title(f'{feature} - Previsão 6 horas à frente', 
                     fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Horas à frente')
    axes[i].set_ylabel(feature)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multivariate_forecast.png', dpi=150)
print("✅ Visualização salva: multivariate_forecast.png")

# ─── 10. MÉTRICAS POR FEATURE ───
print("\n" + "="*70)
print("📊 MÉTRICAS POR FEATURE")
print("="*70)

for i, feature in enumerate(feature_names):
    # MAE por feature
    mae = np.mean(np.abs(y_test_denorm[:, :, i] - y_pred_denorm[:, :, i]))

    # RMSE por feature
    rmse = np.sqrt(np.mean((y_test_denorm[:, :, i] - y_pred_denorm[:, :, i])**2))

    # MAPE por feature
    mape = np.mean(np.abs((y_test_denorm[:, :, i] - y_pred_denorm[:, :, i]) / 
                          (y_test_denorm[:, :, i] + 1e-8))) * 100

    print(f"\n{feature}:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

print("\n✅ PREVISÃO MULTIVARIADA COMPLETA!")

print("\n💡 CONCEITOS:")
print("  • Multivariado: Prever múltiplas variáveis simultaneamente")
print("  • Seq_length=24: Usar 24h de histórico")
print("  • Pred_steps=6: Prever 6h à frente")
print("  • Correlações: Modelo aprende relações entre variáveis")

print("\n📚 APLICAÇÕES:")
print("  • Previsão meteorológica")
print("  • Mercado financeiro (múltiplos ativos)")
print("  • Monitoramento industrial (sensores)")
print("  • Smart grids (consumo energético)")
