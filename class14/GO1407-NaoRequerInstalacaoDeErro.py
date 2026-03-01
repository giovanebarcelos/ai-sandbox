# GO1407-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# LSTM PARA PREVISÃO DE SÉRIES TEMPORAIS
# Exemplo: Previsão de preços de ações
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ─── 1. GERAR DADOS SINTÉTICOS (ou usar yfinance para dados reais) ───
# Série temporal sintética com tendência + sazonalidade + ruído
np.random.seed(42)
t = np.arange(0, 1000)
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.randn(1000) * 5
prices = 100 + trend + seasonal + noise

# Criar DataFrame
df = pd.DataFrame({'Price': prices})

# Visualizar
plt.figure(figsize=(14, 5))
plt.plot(df['Price'])
plt.title('Série Temporal de Preços (Sintética)')
plt.xlabel('Dia')
plt.ylabel('Preço')
plt.grid(True)
plt.show()

# ─── 2. PRÉ-PROCESSAMENTO ───
# Normalizar para [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Price']])

# ─── 3. CRIAR JANELAS DESLIZANTES (SLIDING WINDOW) ───
def create_sequences(data, seq_length):
    """
    Cria sequências para LSTM
    Input: últimos seq_length valores
    Output: próximo valor
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Usar últimos 60 dias para prever próximo

X, y = create_sequences(scaled_data, seq_length)

print(f"X shape: {X.shape}")  # (940, 60, 1)
print(f"y shape: {y.shape}")  # (940, 1)

# Split train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ─── 4. CONSTRUIR MODELO LSTM ───
model_ts = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),

    LSTM(50, return_sequences=False),
    Dropout(0.2),

    Dense(25, activation='relu'),
    Dense(1)  # Predição: próximo valor
], name='stock_prediction_lstm')

model_ts.compile(optimizer='adam', loss='mse', metrics=['mae'])

model_ts.summary()

# ─── 5. TREINAR ───
print("\nTreinando LSTM para séries temporais...")
history_ts = model_ts.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ─── 6. PREDIÇÕES ───
train_predictions = model_ts.predict(X_train)
test_predictions = model_ts.predict(X_test)

# Denormalizar
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_inv = scaler.inverse_transform(y_train)
y_test_inv = scaler.inverse_transform(y_test)

# ─── 7. MÉTRICAS ───
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_mae = mean_absolute_error(y_train_inv, train_predictions)
test_mae = mean_absolute_error(y_test_inv, test_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))

print("\n" + "="*50)
print("MÉTRICAS DE PREVISÃO")
print("="*50)
print(f"Train MAE:  {train_mae:.2f}")
print(f"Test MAE:   {test_mae:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE:  {test_rmse:.2f}")

# ─── 8. VISUALIZAR PREDIÇÕES ───
plt.figure(figsize=(16, 6))

# Plot completo
plt.subplot(1, 2, 1)
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[seq_length:len(train_predictions)+seq_length, :] = train_predictions

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predictions)+(seq_length*2):len(scaled_data), :] = test_predictions

plt.plot(scaler.inverse_transform(scaled_data), label='Real', linewidth=2)
plt.plot(train_plot, label='Train Predictions', alpha=0.7)
plt.plot(test_plot, label='Test Predictions', alpha=0.7)
plt.title('Previsão de Séries Temporais - Visão Geral')
plt.xlabel('Dia')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)

# Zoom no teste
plt.subplot(1, 2, 2)
test_start = len(train_predictions) + seq_length
plt.plot(y_test_inv, label='Real', linewidth=2)
plt.plot(test_predictions, label='Predição LSTM', linestyle='--', linewidth=2)
plt.title('Zoom: Conjunto de Teste')
plt.xlabel('Timestep')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ─── 9. PREVISÃO MULTI-STEP (Próximos 30 dias) ───
def predict_future(model, last_sequence, steps, scaler):
    """
    Prediz próximos N steps recursivamente
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        # Predizer próximo valor
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(pred)

        # Atualizar sequência: remove primeiro, adiciona predição
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, 0] = pred

    # Denormalizar
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Usar últimos 60 valores do teste
last_sequence = X_test[-1]
future_predictions = predict_future(model_ts, last_sequence, 30, scaler)

# Visualizar
plt.figure(figsize=(12, 6))
historical = scaler.inverse_transform(scaled_data[-100:])
plt.plot(range(len(historical)), historical, label='Histórico', linewidth=2)
plt.plot(range(len(historical), len(historical)+30), future_predictions, 
         label='Previsão 30 dias', linestyle='--', marker='o', linewidth=2)
plt.axvline(x=len(historical), color='red', linestyle=':', label='Hoje')
plt.title('Previsão para os Próximos 30 Dias')
plt.xlabel('Dia')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nPrevisão para os próximos 5 dias:")
for i, price in enumerate(future_predictions[:5], 1):
    print(f"  Dia +{i}: ${price[0]:.2f}")
