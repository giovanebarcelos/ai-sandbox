# GO1403-PrevisaoDeConsumoDeEnergiaComLSTM
# ═══════════════════════════════════════════════════════════════════
# PREVISÃO DE CONSUMO DE ENERGIA COM LSTM
# Dataset: Consumo horário de energia elétrica
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ─── 1. GERAR DADOS SINTÉTICOS DE CONSUMO DE ENERGIA ───
print("📊 Gerando dados sintéticos de consumo de energia...")

np.random.seed(42)
n_hours = 24 * 365  # 1 ano de dados horários

# Componentes da série temporal
time = np.arange(n_hours)

# 1. Tendência anual (crescimento)
trend = 0.0001 * time + 50

# 2. Sazonalidade diária (pico durante dia, baixo à noite)
daily_season = 20 * np.sin(2 * np.pi * time / 24)

# 3. Sazonalidade semanal (menor consumo fim de semana)
weekly_season = 5 * np.sin(2 * np.pi * time / (24 * 7))

# 4. Sazonalidade anual (maior consumo no verão - ar condicionado)
yearly_season = 10 * np.sin(2 * np.pi * time / (24 * 365) - np.pi/2)

# 5. Ruído aleatório
noise = np.random.normal(0, 2, n_hours)

# 6. Eventos especiais aleatórios (picos de consumo)
special_events = np.zeros(n_hours)
event_indices = np.random.choice(n_hours, size=20, replace=False)
special_events[event_indices] = np.random.uniform(10, 30, 20)

# Série temporal completa
energy_consumption = trend + daily_season + weekly_season + yearly_season + noise + special_events

# Criar DataFrame
dates = pd.date_range(start='2024-01-01', periods=n_hours, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'consumption_kwh': energy_consumption,
    'hour': dates.hour,
    'day_of_week': dates.dayofweek,
    'month': dates.month
})

print(f"  Total de horas: {len(df)}")
print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
print(f"  Consumo médio: {df['consumption_kwh'].mean():.2f} kWh")
print(f"  Consumo min/max: {df['consumption_kwh'].min():.2f} / {df['consumption_kwh'].max():.2f} kWh")

# Visualizar série temporal
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Série completa
axes[0].plot(df['timestamp'], df['consumption_kwh'], alpha=0.7, linewidth=0.8)
axes[0].set_title('Consumo de Energia - 1 Ano Completo', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Data')
axes[0].set_ylabel('Consumo (kWh)')
axes[0].grid(True, alpha=0.3)

# Zoom em 1 semana
week_data = df.iloc[:24*7]
axes[1].plot(week_data['timestamp'], week_data['consumption_kwh'], linewidth=2)
axes[1].set_title('Consumo de Energia - Primeira Semana (Detalhe)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Data/Hora')
axes[1].set_ylabel('Consumo (kWh)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('energy_timeseries.png', dpi=150)
print("  ✓ Série temporal salva: energy_timeseries.png")

# ─── 2. PREPARAR DADOS PARA LSTM ───
print("\n🔧 Preparando dados para LSTM...")

# Normalizar dados
scaler = MinMaxScaler(feature_range=(0, 1))
consumption_scaled = scaler.fit_transform(df[['consumption_kwh']]).flatten()

# Criar sequências (lookback=24h para prever próxima hora)
def create_sequences(data, lookback=24, forecast_horizon=1):
    """Cria sequências X, y para treinamento"""
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i-lookback:i])
        y.append(data[i:i+forecast_horizon])
    return np.array(X), np.array(y)

LOOKBACK = 24  # Usar últimas 24 horas
FORECAST_HORIZON = 1  # Prever próxima 1 hora

X, y = create_sequences(consumption_scaled, LOOKBACK, FORECAST_HORIZON)

print(f"  Sequências criadas: {X.shape}")
print(f"  X shape: {X.shape} (samples, timesteps)")
print(f"  y shape: {y.shape} (samples, forecast_horizon)")

# Reshape para LSTM [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split treino/validação/teste (60/20/20)
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\n  Treino: {X_train.shape[0]} amostras")
print(f"  Validação: {X_val.shape[0]} amostras")
print(f"  Teste: {X_test.shape[0]} amostras")

# ─── 3. CONSTRUIR MODELO LSTM ───
print("\n🔨 Construindo modelo LSTM...")

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(FORECAST_HORIZON)
], name='LSTM_Energy')

model_lstm.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print(f"  Parâmetros: {model_lstm.count_params():,}")
model_lstm.summary()

# ─── 4. TREINAR MODELO ───
print("\n🚀 Treinando modelo LSTM...")

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

history = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# ─── 5. AVALIAR NO TESTE ───
print("\n📊 Avaliando no conjunto de teste...")

# Predições
y_pred = model_lstm.predict(X_test, verbose=0)

# Desnormalizar
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# Métricas
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2 = r2_score(y_test_orig, y_pred_orig)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

print(f"\n  MAE: {mae:.3f} kWh")
print(f"  RMSE: {rmse:.3f} kWh")
print(f"  R²: {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# ─── 6. VISUALIZAR RESULTADOS ───
print("\n📈 Gerando visualizações...")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 1. Histórico de treinamento
axes[0].plot(history.history['loss'], label='Treino Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validação Loss', linewidth=2)
axes[0].set_title('Histórico de Treinamento', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Predições vs Real (primeiras 200 horas do teste)
n_display = 200
axes[1].plot(y_test_orig[:n_display], label='Real', linewidth=2, alpha=0.8)
axes[1].plot(y_pred_orig[:n_display], label='Previsto', linewidth=2, alpha=0.8)
axes[1].set_title(f'Previsões vs Real (Primeiras {n_display} horas)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hora')
axes[1].set_ylabel('Consumo (kWh)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Scatter plot
axes[2].scatter(y_test_orig, y_pred_orig, alpha=0.5, s=20)
axes[2].plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 
             'r--', linewidth=2, label='Perfeito')
axes[2].set_title('Predito vs Real', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Real (kWh)')
axes[2].set_ylabel('Previsto (kWh)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('energy_lstm_results.png', dpi=150)
print("  ✓ Resultados salvos: energy_lstm_results.png")

# ─── 7. PREVER PRÓXIMOS 7 DIAS (RECURSIVO) ───
print("\n🔮 Prevendo próximos 7 dias (168 horas) recursivamente...")

# Última sequência conhecida
last_sequence = consumption_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
future_predictions = []

for i in range(24 * 7):  # 168 horas
    # Prever próxima hora
    next_pred = model_lstm.predict(last_sequence, verbose=0)[0]
    future_predictions.append(next_pred[0])

    # Atualizar sequência (remover primeiro, adicionar novo no final)
    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

# Desnormalizar previsões futuras
future_predictions_orig = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# Visualizar previsão futura
plt.figure(figsize=(15, 6))

# Últimas 7 dias reais + próximos 7 dias previstos
last_week_real = df['consumption_kwh'].values[-24*7:]
future_dates = pd.date_range(start=df['timestamp'].iloc[-1] + pd.Timedelta(hours=1), 
                              periods=24*7, freq='H')

plt.plot(df['timestamp'].iloc[-24*7:], last_week_real, 
         label='Últimos 7 dias (Real)', linewidth=2, color='blue')
plt.plot(future_dates, future_predictions_orig, 
         label='Próximos 7 dias (Previsto)', linewidth=2, color='red', linestyle='--')
plt.axvline(x=df['timestamp'].iloc[-1], color='green', linestyle=':', 
            linewidth=2, label='Presente')
plt.title('Previsão de Consumo - Próximos 7 Dias', fontsize=14, fontweight='bold')
plt.xlabel('Data/Hora')
plt.ylabel('Consumo (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('energy_future_forecast.png', dpi=150)
print("  ✓ Previsão futura salva: energy_future_forecast.png")

# ─── 8. ANÁLISE DE RESÍDUOS ───
print("\n🔬 Análise de resíduos...")

residuals = y_test_orig - y_pred_orig

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma dos resíduos
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Resíduo (kWh)')
axes[0].set_ylabel('Frequência')
axes[0].grid(True, alpha=0.3)

# Resíduos ao longo do tempo
axes[1].scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Resíduos ao Longo do Tempo', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Amostra')
axes[1].set_ylabel('Resíduo (kWh)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('energy_residuals.png', dpi=150)
print("  ✓ Análise de resíduos salva: energy_residuals.png")

# Estatísticas dos resíduos
print(f"\n  Média dos resíduos: {residuals.mean():.4f} kWh")
print(f"  Desvio padrão: {residuals.std():.4f} kWh")
print(f"  Min/Max: {residuals.min():.4f} / {residuals.max():.4f} kWh")

# ─── 9. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ PREVISÃO DE CONSUMO DE ENERGIA CONCLUÍDA!")
print("="*70)

print("\n📊 Resumo do Modelo:")
print(f"  Arquitetura: LSTM(64) → Dropout → LSTM(32) → Dropout → Dense")
print(f"  Parâmetros: {model_lstm.count_params():,}")
print(f"  Lookback: {LOOKBACK} horas")
print(f"  Forecast horizon: {FORECAST_HORIZON} hora(s)")

print("\n📈 Performance:")
print(f"  MAE: {mae:.3f} kWh (~{mape:.1f}% de erro)")
print(f"  RMSE: {rmse:.3f} kWh")
print(f"  R²: {r2:.4f} ({'Excelente' if r2 > 0.9 else 'Bom' if r2 > 0.8 else 'Razoável'})")

print("\n📁 Arquivos gerados:")
print("  • energy_timeseries.png - Série temporal original")
print("  • energy_lstm_results.png - Avaliação do modelo")
print("  • energy_future_forecast.png - Previsão futura (7 dias)")
print("  • energy_residuals.png - Análise de erros")

print("\n💡 Insights:")
print("  • Modelo captura bem padrões diários e semanais")
print("  • Resíduos aproximadamente normais indicam bom ajuste")
print("  • Previsão recursiva mantém tendências de curto prazo")

print("\n🔧 Melhorias possíveis:")
print("  • Adicionar features: temperatura, feriados, eventos")
print("  • Testar Bidirectional LSTM")
print("  • Implementar attention mechanism")
print("  • Ensemble com GRU e Simple RNN")

# ───────────────────────────────────────────────────────────────────
# ✅ CHECKPOINT FINAL - VALIDAÇÕES:
# ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("✅ CHECKPOINT FINAL - AVALIAÇÃO DA ATIVIDADE")
print("="*70)

print(f"\n📊 MÉTRICAS ALCANÇADAS:")
print(f"   MAE: {mae:.3f} kWh")
print(f"   RMSE: {rmse:.3f} kWh")
print(f"   R²: {r2:.4f}")
print(f"   MAPE: {mape:.2f}%")

print(f"\n🎯 NÍVEL ALCANÇADO:")
if mae < 3 and r2 > 0.90:
    print("   🌟 AVANÇADO - Excelente! MAE<3 e R²>0.90")
    print("   Sugestões:")
    print("   - Implemente desafios opcionais (Attention, Multi-step)")
    print("   - Teste com datasets reais (UCI, Kaggle)")
    print("   - Compare com modelos estatísticos (ARIMA, Prophet)")
elif mae < 5 and r2 > 0.85:
    print("   ✅ INTERMEDIÁRIO - Muito bom! MAE<5 e R²>0.85")
    print("   Sugestões de melhoria:")
    print("   - Adicione features exógenas (hora, dia da semana)")
    print("   - Teste Bidirectional LSTM")
    print("   - Aumente lookback para 48-72 horas")
elif mae < 8 and r2 > 0.75:
    print("   ✅ MÍNIMO - Objetivo alcançado! MAE<8 e R²>0.75")
    print("   Sugestões de melhoria:")
    print("   - Aumente número de neurônios LSTM (64→128)")
    print("   - Treine mais épocas (100→200)")
    print("   - Verifique se early stopping não parou muito cedo")
else:
    print("   ⚠️ ABAIXO DO MÍNIMO - Revisar implementação")
    print("   Checklist:")
    print("   ✓ Dados normalizados [0,1]?")
    print("   ✓ Shape correto (N, timesteps, features)?")
    print("   ✓ Modelo treinou convergindo (loss diminuiu)?")
    print("   ✓ Lookback adequado (>= 24 horas)?")

print(f"\n📚 CONCEITOS APLICADOS:")
print(f"   ✓ Série temporal sintética com múltiplos componentes")
print(f"   ✓ Sliding window (lookback) para sequências")
print(f"   ✓ Normalização MinMaxScaler [0,1]")
print(f"   ✓ LSTM stacked (múltiplas camadas)")
print(f"   ✓ Dropout para regularização")
print(f"   ✓ Early Stopping e ReduceLROnPlateau")
print(f"   ✓ Métricas de regressão (MAE, RMSE, R², MAPE)")
print(f"   ✓ Previsão recursiva para forecast futuro")
print(f"   ✓ Análise de resíduos (normalidade)")

print(f"\n🎓 PRÓXIMOS PASSOS:")
print(f"   1. Revisar Exercício Prático 2 (Text Classification)")
print(f"   2. Estudar Slide 15 (Bidirectional RNN/LSTM)")
print(f"   3. Avançar para Aula 16 (Transformers e Attention)")

print("\n🎉🎉🎉 ATIVIDADE COMPLETA! EXCELENTE TRABALHO! 🎉🎉🎉")
