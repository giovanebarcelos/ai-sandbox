# GO1413-NãoRequerInstalaçãoDeErro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PREVISÃO DE TEMPERATURA COM LSTM - Dados Reais")
print("="*80)

# ============================================================================
# 1. CARREGAR E EXPLORAR DADOS REAIS
# ============================================================================

print("\n📥 Carregando dados de temperatura...")

# Opção 1: Carregar dataset real do Kaggle ou OpenWeatherMap
# Para este exemplo, vamos usar um dataset público

# Dataset: Daily Temperature of Major Cities (Kaggle)
# URL: https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

# Vamos simular um carregamento de dados reais estruturados
# Em produção, você usaria: pd.read_csv('temperature_data.csv')

# Gerando dados sintéticos que simulam padrões reais de temperatura
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
n_days = len(dates)

# Simular temperatura com componentes sazonais + tendência + ruído
# Padrão realista: sazonalidade anual + variação diária + tendência de aquecimento
day_of_year = dates.dayofyear
year = dates.year - 2010

# Componente sazonal (anual)
seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365)

# Tendência de longo prazo (aquecimento global: +0.02°C/ano)
trend = 0.02 * (year * 365 + day_of_year)

# Temperatura base
base_temp = 20

# Ruído realista (variação diária)
noise = np.random.normal(0, 3, n_days)

# Temperatura final
temperature = base_temp + seasonal + trend + noise

# Criar DataFrame
df = pd.DataFrame({
    'date': dates,
    'temperature': temperature
})

# Adicionar features temporais
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_year'] = df['date'].dt.dayofyear
df['day_of_week'] = df['date'].dt.dayofweek

print(f"✅ Dados carregados: {len(df)} dias ({df['date'].min().date()} a {df['date'].max().date()})")
print(f"\nEstatísticas de Temperatura:")
print(df['temperature'].describe())

# Visualizar série temporal completa
plt.figure(figsize=(16, 4))
plt.plot(df['date'], df['temperature'], linewidth=0.8, alpha=0.7)
plt.title('Série Temporal de Temperatura Diária (2010-2023)')
plt.xlabel('Data')
plt.ylabel('Temperatura (°C)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aula14_temperature_series.png', dpi=300, bbox_inches='tight')
print("\n📊 Gráfico salvo: aula14_temperature_series.png")

# Análise sazonal
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
monthly_avg = df.groupby('month')['temperature'].mean()
plt.bar(range(1, 13), monthly_avg.values, alpha=0.7, edgecolor='black')
plt.xlabel('Mês')
plt.ylabel('Temperatura Média (°C)')
plt.title('Temperatura Média por Mês')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
yearly_avg = df.groupby('year')['temperature'].mean()
plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
plt.xlabel('Ano')
plt.ylabel('Temperatura Média (°C)')
plt.title('Temperatura Média por Ano (Tendência)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aula14_temperature_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Análise sazonal salva: aula14_temperature_analysis.png")

# ============================================================================
# 2. PREPARAR DADOS COM SLIDING WINDOW
# ============================================================================

print("\n" + "="*80)
print("PREPARANDO DADOS COM SLIDING WINDOW")
print("="*80)

def create_sequences(data, window_size, forecast_horizon=1):
    """
    Cria sequências para LSTM usando sliding window

    Args:
        data: Array de valores
        window_size: Número de dias passados para usar como input
        forecast_horizon: Número de dias futuros para prever

    Returns:
        X, y: Arrays de features e targets
    """
    X, y = [], []

    for i in range(len(data) - window_size - forecast_horizon + 1):
        # Input: últimos window_size dias
        X.append(data[i:i+window_size])
        # Output: próximos forecast_horizon dias
        y.append(data[i+window_size:i+window_size+forecast_horizon])

    return np.array(X), np.array(y)

# Parâmetros
WINDOW_SIZE = 30  # Usar últimos 30 dias
FORECAST_HORIZON = 7  # Prever próximos 7 dias
TRAIN_SIZE = 0.7  # 70% treino
VAL_SIZE = 0.15   # 15% validação (15% teste restante)

# Normalizar dados (importante para LSTM!)
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_scaled = scaler.fit_transform(df[['temperature']])

print(f"Configuração:")
print(f"  Window Size: {WINDOW_SIZE} dias")
print(f"  Forecast Horizon: {FORECAST_HORIZON} dias")
print(f"  Train/Val/Test: {TRAIN_SIZE:.0%}/{VAL_SIZE:.0%}/{1-TRAIN_SIZE-VAL_SIZE:.0%}")

# Criar sequências
X, y = create_sequences(temperature_scaled.flatten(), WINDOW_SIZE, FORECAST_HORIZON)

print(f"\nForma dos dados:")
print(f"  X: {X.shape} (samples, window_size)")
print(f"  y: {y.shape} (samples, forecast_horizon)")

# Reshape para LSTM: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"  X após reshape: {X.shape}")

# Dividir em treino, validação e teste
n_samples = len(X)
train_end = int(n_samples * TRAIN_SIZE)
val_end = int(n_samples * (TRAIN_SIZE + VAL_SIZE))

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(X_train)} amostras")
print(f"  Validação: {len(X_val)} amostras")
print(f"  Teste: {len(X_test)} amostras")

# ============================================================================
# 3. CONSTRUIR MODELO LSTM
# ============================================================================

print("\n" + "="*80)
print("CONSTRUINDO MODELO LSTM")
print("="*80)

model = Sequential([
    # Primeira camada LSTM com return_sequences=True
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),

    # Segunda camada LSTM
    LSTM(32, return_sequences=False),
    Dropout(0.2),

    # Camadas Dense para gerar previsões
    Dense(16, activation='relu'),
    Dense(FORECAST_HORIZON)  # Output: próximos FORECAST_HORIZON dias
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Arquitetura do modelo:")
model.summary()

# ============================================================================
# 4. TREINAR MODELO
# ============================================================================

print("\n" + "="*80)
print("TREINANDO MODELO")
print("="*80)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_temperature_lstm.h5', save_best_only=True, monitor='val_loss')
]

# Treinar
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# Plotar curvas de aprendizado
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.title('Curva de Aprendizado - Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Treino')
plt.plot(history.history['val_mae'], label='Validação')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.title('Curva de Aprendizado - MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aula14_training_curves.png', dpi=300, bbox_inches='tight')
print("📊 Curvas de aprendizado salvas: aula14_training_curves.png")

# ============================================================================
# 5. AVALIAR MODELO
# ============================================================================

print("\n" + "="*80)
print("AVALIANDO MODELO NO CONJUNTO DE TESTE")
print("="*80)

# Fazer previsões
y_pred = model.predict(X_test)

# Desnormalizar previsões e valores reais
y_test_denorm = scaler.inverse_transform(y_test)
y_pred_denorm = scaler.inverse_transform(y_pred)

# Calcular métricas para cada horizonte de previsão
print("\nMétricas por horizonte de previsão:")
print(f"{'Dia':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
print("-" * 45)

for day in range(FORECAST_HORIZON):
    mae = mean_absolute_error(y_test_denorm[:, day], y_pred_denorm[:, day])
    rmse = np.sqrt(mean_squared_error(y_test_denorm[:, day], y_pred_denorm[:, day]))
    mape = np.mean(np.abs((y_test_denorm[:, day] - y_pred_denorm[:, day]) / y_test_denorm[:, day])) * 100

    print(f"  Dia +{day+1:1d}   {mae:>10.3f} {rmse:>10.3f} {mape:>10.2f}%")

# Métricas gerais
mae_overall = mean_absolute_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
rmse_overall = np.sqrt(mean_squared_error(y_test_denorm.flatten(), y_pred_denorm.flatten()))
mape_overall = np.mean(np.abs((y_test_denorm - y_pred_denorm) / y_test_denorm)) * 100

print("\n" + "="*80)
print("MÉTRICAS GERAIS:")
print("="*80)
print(f"MAE:  {mae_overall:.3f}°C")
print(f"RMSE: {rmse_overall:.3f}°C")
print(f"MAPE: {mape_overall:.2f}%")

# ============================================================================
# 6. VISUALIZAR PREVISÕES
# ============================================================================

print("\n" + "="*80)
print("VISUALIZANDO PREVISÕES")
print("="*80)

# Selecionar algumas previsões para visualizar
n_examples = 5
sample_indices = np.linspace(0, len(y_test_denorm)-1, n_examples, dtype=int)

fig, axes = plt.subplots(n_examples, 1, figsize=(14, 12))

for idx, sample_idx in enumerate(sample_indices):
    ax = axes[idx]

    # Dias de input (histórico)
    input_days = range(-WINDOW_SIZE, 0)
    input_temps = scaler.inverse_transform(X_test[sample_idx].reshape(-1, 1)).flatten()

    # Dias de previsão
    forecast_days = range(0, FORECAST_HORIZON)
    actual_temps = y_test_denorm[sample_idx]
    predicted_temps = y_pred_denorm[sample_idx]

    # Plotar
    ax.plot(input_days, input_temps, 'o-', label='Histórico (input)', 
            color='gray', alpha=0.6, linewidth=2)
    ax.plot(forecast_days, actual_temps, 'o-', label='Real', 
            color='blue', linewidth=2, markersize=8)
    ax.plot(forecast_days, predicted_temps, 'o--', label='Previsto', 
            color='red', linewidth=2, markersize=8)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Dias (relativo ao ponto de previsão)')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_title(f'Exemplo {idx+1}: MAE = {mean_absolute_error(actual_temps, predicted_temps):.2f}°C')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aula14_forecast_examples.png', dpi=300, bbox_inches='tight')
print("📊 Exemplos de previsão salvos: aula14_forecast_examples.png")

# Visualizar período contínuo
print("\nVisualizando 90 dias de previsão contínua...")

n_days_viz = 90
start_idx = 100

# Dados reais
dates_test = df['date'].iloc[val_end + WINDOW_SIZE + start_idx:
                              val_end + WINDOW_SIZE + start_idx + n_days_viz]
temps_real = df['temperature'].iloc[val_end + WINDOW_SIZE + start_idx:
                                     val_end + WINDOW_SIZE + start_idx + n_days_viz].values

# Previsões (pegando apenas o primeiro dia de cada previsão para evitar sobreposição)
temps_pred = []
for i in range(start_idx, min(start_idx + n_days_viz, len(y_pred_denorm))):
    temps_pred.append(y_pred_denorm[i][0])  # Primeiro dia da previsão

# Ajustar comprimento
min_len = min(len(dates_test), len(temps_pred))
dates_test = dates_test[:min_len]
temps_real = temps_real[:min_len]
temps_pred = temps_pred[:min_len]

plt.figure(figsize=(16, 6))
plt.plot(dates_test, temps_real, 'o-', label='Temperatura Real', 
         color='blue', linewidth=2, markersize=4, alpha=0.7)
plt.plot(dates_test, temps_pred, 'o-', label='Previsão LSTM (D+1)', 
         color='red', linewidth=2, markersize=4, alpha=0.7)
plt.fill_between(dates_test, temps_real, temps_pred, alpha=0.2, color='gray')
plt.xlabel('Data')
plt.ylabel('Temperatura (°C)')
plt.title(f'Previsão de Temperatura - Período de {min_len} dias')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('aula14_continuous_forecast.png', dpi=300, bbox_inches='tight')
print("📊 Previsão contínua salva: aula14_continuous_forecast.png")

# ============================================================================
# 7. ANÁLISE DE ERROS
# ============================================================================

print("\n" + "="*80)
print("ANÁLISE DE ERROS")
print("="*80)

# Calcular erros
errors = y_pred_denorm[:, 0] - y_test_denorm[:, 0]  # Erros do primeiro dia

print(f"Erro médio: {np.mean(errors):.3f}°C")
print(f"Desvio padrão dos erros: {np.std(errors):.3f}°C")
print(f"Erro máximo: {np.max(np.abs(errors)):.3f}°C")

# Distribuição dos erros
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Erro de Previsão (°C)')
plt.ylabel('Frequência')
plt.title('Distribuição dos Erros de Previsão')
plt.axvline(x=0, color='red', linestyle='--', label='Erro Zero')
plt.axvline(x=np.mean(errors), color='green', linestyle='--', 
            label=f'Erro Médio: {np.mean(errors):.2f}°C')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
plt.scatter(y_test_denorm[:, 0], y_pred_denorm[:, 0], alpha=0.5, s=10)
plt.plot([y_test_denorm[:, 0].min(), y_test_denorm[:, 0].max()],
         [y_test_denorm[:, 0].min(), y_test_denorm[:, 0].max()],
         'r--', linewidth=2, label='Predição Perfeita')
plt.xlabel('Temperatura Real (°C)')
plt.ylabel('Temperatura Prevista (°C)')
plt.title('Real vs Previsto (Dia +1)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aula14_error_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Análise de erros salva: aula14_error_analysis.png")

# ============================================================================
# 8. FUNÇÃO DE PREVISÃO
# ============================================================================

def forecast_temperature(model, last_days, scaler, n_days=7):
    """
    Prevê temperatura para os próximos n dias

    Args:
        model: Modelo LSTM treinado
        last_days: Últimos WINDOW_SIZE dias de temperatura
        scaler: Scaler usado no treinamento
        n_days: Número de dias para prever

    Returns:
        Array com temperaturas previstas
    """
    # Normalizar input
    last_days_scaled = scaler.transform(last_days.reshape(-1, 1))

    # Reshape para LSTM
    X = last_days_scaled.reshape(1, len(last_days), 1)

    # Prever
    prediction_scaled = model.predict(X, verbose=0)

    # Desnormalizar
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction[0]

# Exemplo de uso
print("\n" + "="*80)
print("EXEMPLO DE USO: PREVENDO PRÓXIMOS 7 DIAS")
print("="*80)

# Pegar últimos 30 dias de temperatura
last_30_days = df['temperature'].iloc[-WINDOW_SIZE:].values

print(f"\nÚltimos {WINDOW_SIZE} dias de temperatura:")
print(last_30_days)

# Prever próximos 7 dias
forecast = forecast_temperature(model, last_30_days, scaler, n_days=FORECAST_HORIZON)

print(f"\nPrevisão para os próximos {FORECAST_HORIZON} dias:")
for day, temp in enumerate(forecast, 1):
    print(f"  Dia +{day}: {temp:.2f}°C")

print("\n✅ Sistema de previsão de temperatura implementado com sucesso!")
print("💡 Use: forecast_temperature(model, last_30_days, scaler) para prever")
