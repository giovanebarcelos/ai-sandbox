# GO1432-NãoRequerInstalaçãoDeErro
# ══════════════════════════════════════════════════════════════════
# MULTI-STEP FORECASTING - ENCODER-DECODER
# Prever múltiplos passos futuros
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

print("📈 MULTI-STEP FORECASTING")
print("=" * 70)

# ─── 1. GERAR SÉRIE TEMPORAL ───
print("\n📊 Gerando série temporal...")

np.random.seed(42)

def generate_series(n_samples=1000, length=100):
    series = []

    for _ in range(n_samples):
        # Tendência + sazonalidade + ruído
        t = np.linspace(0, 10, length)

        trend = 0.5 * t
        seasonal = 2 * np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 1.5 * t)
        noise = np.random.normal(0, 0.3, length)

        ts = trend + seasonal + noise
        series.append(ts)

    return np.array(series)

data = generate_series(1000, length=100)

print(f"  Series shape: {data.shape}")

# ─── 2. PREPARAR DADOS (MULTI-STEP) ───
print("\n⚙️ Preparando dados para multi-step forecasting...")

input_length = 30
output_length = 10  # Prever 10 passos futuros

def create_multistep_dataset(data, input_len, output_len):
    X, y = [], []

    for series in data:
        for i in range(len(series) - input_len - output_len + 1):
            X.append(series[i:i+input_len])
            y.append(series[i+input_len:i+input_len+output_len])

    return np.array(X), np.array(y)

X_data, y_data = create_multistep_dataset(data, input_length, output_length)

print(f"  X shape (input): {X_data.shape}")
print(f"  y shape (output): {y_data.shape}")

# Reshape
X_data = X_data.reshape(-1, input_length, 1)
y_data = y_data.reshape(-1, output_length, 1)

# Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 3. ENCODER-DECODER MODEL ───
print("\n🏗️ Construindo Encoder-Decoder...")

latent_dim = 64

# Encoder
encoder_inputs = Input(shape=(input_length, 1))
encoder = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = RepeatVector(output_length)(state_h)
decoder = LSTM(latent_dim, return_sequences=True)
decoder_outputs = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(1))
outputs = decoder_dense(decoder_outputs)

model = Model(encoder_inputs, outputs, name='Encoder_Decoder')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando...")

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    verbose=0
)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"  Test MAE: {test_mae:.4f}")

# ─── 5. PREVISÕES ───
print("\n🔮 Gerando previsões...")

predictions = model.predict(X_test[:10], verbose=0)

# ─── 6. VISUALIZAR ───
print("\n📊 Visualizando previsões multi-step...")

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

for i in range(6):
    ax = axes[i // 3, i % 3]

    # Input
    input_seq = X_test[i].squeeze()
    true_future = y_test[i].squeeze()
    pred_future = predictions[i].squeeze()

    # Plot
    time_input = np.arange(len(input_seq))
    time_future = np.arange(len(input_seq), len(input_seq) + len(true_future))

    ax.plot(time_input, input_seq, 'b-', label='Input', linewidth=2)
    ax.plot(time_future, true_future, 'g-', label='True Future', linewidth=2)
    ax.plot(time_future, pred_future, 'r--', label='Predicted', linewidth=2)

    ax.axvline(len(input_seq), color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'Sample {i+1}', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Multi-Step Forecasting: {input_length} → {output_length} steps', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('multistep_forecasting.png', dpi=150)
print("✅ Previsões salvas: multistep_forecasting.png")

# ─── 7. COMPARAR ESTRATÉGIAS ───
print("\n📊 Comparando estratégias de multi-step...")

# Estratégia 1: Direct (já implementado acima)
mae_direct = test_mae

# Estratégia 2: Recursive (1-step iterativo)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model_recursive = Sequential([
    LSTM(64, input_shape=(input_length, 1)),
    Dense(1)
], name='Recursive')

model_recursive.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar para 1-step
X_train_1step = X_train
y_train_1step = y_train[:, 0, :]  # Primeiro passo apenas

model_recursive.fit(X_train_1step, y_train_1step, epochs=20, batch_size=64, verbose=0)

# Prever recursivamente
def predict_recursive(model, input_seq, n_steps):
    predictions = []
    current_input = input_seq.copy()

    for _ in range(n_steps):
        pred = model.predict(current_input.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])

        # Update input: remove primeiro, adicionar predição
        current_input = np.append(current_input[1:], pred[0, 0])

    return np.array(predictions)

# Avaliar recursive
errors_recursive = []
for i in range(len(X_test)):
    pred = predict_recursive(model_recursive, X_test[i].squeeze(), output_length)
    error = np.abs(y_test[i].squeeze() - pred).mean()
    errors_recursive.append(error)

mae_recursive = np.mean(errors_recursive)

print(f"\n  Direct Strategy MAE: {mae_direct:.4f}")
print(f"  Recursive Strategy MAE: {mae_recursive:.4f}")

# Plot comparação
fig, ax = plt.subplots(figsize=(8, 6))

strategies = ['Direct\n(Encoder-Decoder)', 'Recursive\n(Iterative 1-step)']
maes = [mae_direct, mae_recursive]

ax.bar(strategies, maes, color=['green', 'orange'], alpha=0.7)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('Multi-Step Forecasting Strategies', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, mae in enumerate(maes):
    ax.text(i, mae + 0.01, f'{mae:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('multistep_strategies.png', dpi=150)
print("✅ Estratégias salvas: multistep_strategies.png")

print("\n💡 MULTI-STEP FORECASTING:")
print("  • Direct: Prever todos os passos de uma vez")
print("  • Recursive: Prever 1-step iterativamente")
print("  • Multiple Output: Múltiplos modelos independentes")
print("  • Encoder-Decoder: Melhor para longas sequências")

print("\n🎯 ESTRATÉGIAS:")
print("  • Direct: Rápido, não propaga erros")
print("  • Recursive: Erro acumula, mais lento")
print("  • Hybrid: Combinar ambas abordagens")
print("  • Attention: Focar em partes relevantes do input")

print("\n🏆 APLICAÇÕES:")
print("  • Energy: Previsão de demanda elétrica")
print("  • Finance: Prever preços de ações")
print("  • Weather: Previsão meteorológica")
print("  • Traffic: Fluxo de tráfego")

print("\n✅ MULTI-STEP FORECASTING COMPLETO!")
