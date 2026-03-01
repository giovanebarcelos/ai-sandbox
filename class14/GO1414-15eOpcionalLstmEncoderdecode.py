# GO1414-15eOpcionalLstmEncoderdecode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

print("="*70)
print("LSTM ENCODER-DECODER - Previsão Multivariada")
print("="*70)

# 1. GERAR DADOS SINTÉTICOS MULTIVARIADOS
def generate_multivariate_data(n_samples=1000):
    """
    Gera dados sintéticos com múltiplas variáveis correlacionadas
    """
    np.random.seed(42)

    # Data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Tendência
    trend = np.linspace(20, 30, n_samples)

    # Sazonalidade anual (365 dias)
    seasonality_annual = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 365)

    # Sazonalidade semanal (7 dias)
    seasonality_weekly = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 7)

    # Temperatura (variável target)
    temperature = trend + seasonality_annual + seasonality_weekly + \
                 np.random.normal(0, 1, n_samples)

    # Umidade (correlacionada negativamente com temperatura)
    humidity = 70 - 0.5 * (temperature - 25) + np.random.normal(0, 3, n_samples)
    humidity = np.clip(humidity, 30, 100)

    # Pressão atmosférica
    pressure = 1013 + np.random.normal(0, 5, n_samples)

    # Velocidade do vento
    wind_speed = 10 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / 30) + \
                np.random.normal(0, 2, n_samples)
    wind_speed = np.clip(wind_speed, 0, 30)

    # DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed
    })

    return df

df = generate_multivariate_data(n_samples=2000)
print(f"\n📊 Dataset gerado: {df.shape}")
print(df.head())
print("\n📈 Estatísticas:")
print(df.describe())

# Visualizar dados
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

variables = ['temperature', 'humidity', 'pressure', 'wind_speed']
titles = ['Temperatura (°C)', 'Umidade (%)', 'Pressão (hPa)', 'Vento (km/h)']

for ax, var, title in zip(axes, variables, titles):
    ax.plot(df['date'], df[var], linewidth=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Data')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multivariate_data.png', dpi=150)
print("✅ Dados visualizados em multivariate_data.png")

# 2. PREPARAR DADOS
def create_sequences_multivariate(data, input_length, output_length):
    """
    Cria sequências para encoder-decoder multivariado

    Args:
        data: array (n_samples, n_features)
        input_length: comprimento da sequência de entrada
        output_length: comprimento da sequência de saída

    Returns:
        X: (n_samples, input_length, n_features)
        y: (n_samples, output_length, 1)  # apenas temperatura
    """
    X, y = [], []

    for i in range(len(data) - input_length - output_length + 1):
        # Entrada: todas as features
        X.append(data[i:i+input_length])

        # Saída: apenas temperatura
        y.append(data[i+input_length:i+input_length+output_length, 0])

    return np.array(X), np.array(y).reshape(-1, output_length, 1)

# Configuração
INPUT_LENGTH = 30   # 30 dias de histórico
OUTPUT_LENGTH = 7   # Prever próximos 7 dias
TRAIN_SPLIT = 0.8

# Separar features
feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
data = df[feature_cols].values

# Normalizar
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Criar sequências
X, y = create_sequences_multivariate(data_scaled, INPUT_LENGTH, OUTPUT_LENGTH)

print(f"\n🔧 Preparação dos Dados:")
print(f"  X shape: {X.shape}")  # (samples, input_length, n_features)
print(f"  y shape: {y.shape}")  # (samples, output_length, 1)

# Split train/test
split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# 3. CONSTRUIR MODELO ENCODER-DECODER
def build_encoder_decoder(input_length, output_length, n_features, 
                         encoder_units=128, decoder_units=128):
    """
    Constrói modelo Encoder-Decoder LSTM

    Arquitetura:
    - Encoder: processa sequência de entrada e gera contexto
    - Decoder: gera sequência de saída baseado no contexto
    """
    # ENCODER
    encoder_inputs = layers.Input(shape=(input_length, n_features), name='encoder_input')

    # LSTM Encoder (retorna estado final)
    encoder_lstm = layers.LSTM(encoder_units, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # Estados do encoder são usados como contexto
    encoder_states = [state_h, state_c]

    # DECODER
    decoder_inputs = layers.Input(shape=(output_length, 1), name='decoder_input')

    # LSTM Decoder (inicializado com estados do encoder)
    decoder_lstm = layers.LSTM(decoder_units, return_sequences=True, 
                              return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Camada Dense para output
    decoder_dense = layers.Dense(1, name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Modelo completo
    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs, 
                        name='encoder_decoder_lstm')

    return model

model = build_encoder_decoder(
    input_length=INPUT_LENGTH,
    output_length=OUTPUT_LENGTH,
    n_features=len(feature_cols),
    encoder_units=128,
    decoder_units=128
)

model.summary()

# 4. COMPILAR E TREINAR
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Preparar decoder inputs (sequência deslocada)
# No treinamento, usamos y deslocado como entrada do decoder (teacher forcing)
decoder_input_train = np.zeros((len(y_train), OUTPUT_LENGTH, 1))
decoder_input_train[:, 1:, :] = y_train[:, :-1, :]

decoder_input_test = np.zeros((len(y_test), OUTPUT_LENGTH, 1))
decoder_input_test[:, 1:, :] = y_test[:, :-1, :]

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

print("\n🏋️ Treinando modelo Encoder-Decoder...")
history = model.fit(
    [X_train, decoder_input_train], y_train,
    validation_data=([X_test, decoder_input_test], y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 5. AVALIAR MODELO
print("\n📊 Avaliando modelo...")
test_loss, test_mae = model.evaluate([X_test, decoder_input_test], y_test, verbose=0)

# Desnormalizar MAE
temp_scaler = MinMaxScaler()
temp_scaler.fit(df[['temperature']])
test_mae_real = test_mae * (temp_scaler.data_max_[0] - temp_scaler.data_min_[0])

print(f"  Test Loss (MSE): {test_loss:.4f}")
print(f"  Test MAE (scaled): {test_mae:.4f}")
print(f"  Test MAE (real): {test_mae_real:.2f}°C")

# 6. VISUALIZAR PREDIÇÕES
# Função para prever sem teacher forcing
def predict_sequence(model, encoder_input):
    """
    Predição autoregressiva (sem teacher forcing)
    """
    # Encoder states
    encoder_lstm = model.get_layer('encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_input)
    states = [state_h, state_c]

    # Decoder
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_dense = model.get_layer('decoder_output')

    # Inicializar decoder input
    decoder_input = np.zeros((1, 1, 1))

    # Gerar sequência
    predictions = []

    for _ in range(OUTPUT_LENGTH):
        # Prever próximo valor
        decoder_output, state_h, state_c = decoder_lstm(decoder_input, initial_state=states)
        output = decoder_dense(decoder_output)

        # Salvar predição
        predictions.append(output[0, 0, 0])

        # Usar predição como próximo input
        decoder_input = output
        states = [state_h, state_c]

    return np.array(predictions)

# Prever algumas sequências de teste
n_examples = 5
test_indices = np.random.choice(len(X_test), n_examples, replace=False)

fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples))

for i, idx in enumerate(test_indices):
    # Prever
    pred = predict_sequence(model, X_test[idx:idx+1])
    true = y_test[idx, :, 0]

    # Desnormalizar
    pred_real = temp_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true_real = temp_scaler.inverse_transform(true.reshape(-1, 1)).flatten()

    # Plotar
    days = np.arange(1, OUTPUT_LENGTH + 1)
    axes[i].plot(days, true_real, 'o-', label='Real', linewidth=2, markersize=8)
    axes[i].plot(days, pred_real, 's--', label='Predito', linewidth=2, markersize=6)
    axes[i].set_title(f'Exemplo {i+1} - MAE: {np.mean(np.abs(pred_real - true_real)):.2f}°C',
                     fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Dia')
    axes[i].set_ylabel('Temperatura (°C)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('encoder_decoder_predictions.png', dpi=150)
print("\n✅ Predições visualizadas em encoder_decoder_predictions.png")

# 7. PLOTAR HISTÓRICO
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss durante Treinamento', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Train')
axes[1].plot(history.history['val_mae'], label='Validation')
axes[1].set_title('MAE durante Treinamento', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('encoder_decoder_training.png', dpi=150)
print("✅ Histórico de treinamento salvo")

print("\n✅ ENCODER-DECODER LSTM COMPLETO!")
print("\n📊 Resumo:")
print(f"  Arquitetura: Encoder-Decoder com LSTM")
print(f"  Input: {INPUT_LENGTH} dias × {len(feature_cols)} features")
print(f"  Output: {OUTPUT_LENGTH} dias de temperatura")
print(f"  MAE: {test_mae_real:.2f}°C")
print(f"  Parâmetros: {model.count_params():,}")
