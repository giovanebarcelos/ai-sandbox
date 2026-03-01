# GO1424-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# GRU vs LSTM - COMPARAÇÃO DETALHADA
# Comparar performance, velocidade e uso de memória
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding
import matplotlib.pyplot as plt
import time

print("⚔️ GRU vs LSTM - COMPARAÇÃO")
print("=" * 70)

# ─── 1. CRIAR DATASET ───
print("\n📦 Gerando dataset sintético (soma de sequências)...")

np.random.seed(42)

# Tarefa: somar elementos de uma sequência
num_samples = 5000
seq_length = 20
max_value = 10

X = np.random.randint(0, max_value, size=(num_samples, seq_length))
y = np.sum(X, axis=1)

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Exemplo: {X[0]} → {y[0]}")

# Normalizar
X = X.astype('float32') / max_value
y = y.astype('float32') / (seq_length * max_value)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. CONSTRUIR MODELOS ───
print("\n🏗️ Construindo modelos...")

# LSTM
model_lstm = Sequential([
    LSTM(128, input_shape=(seq_length, 1)),
    Dense(64, activation='relu'),
    Dense(1)
], name='LSTM_Model')

# GRU
model_gru = Sequential([
    GRU(128, input_shape=(seq_length, 1)),
    Dense(64, activation='relu'),
    Dense(1)
], name='GRU_Model')

# SimpleRNN (baseline)
model_rnn = Sequential([
    tf.keras.layers.SimpleRNN(128, input_shape=(seq_length, 1)),
    Dense(64, activation='relu'),
    Dense(1)
], name='SimpleRNN_Model')

models = {
    'LSTM': model_lstm,
    'GRU': model_gru,
    'SimpleRNN': model_rnn
}

for name, model in models.items():
    print(f"  {name}: {model.count_params():,} parâmetros")

# ─── 3. TREINAR E COMPARAR ───
print("\n🚀 Treinando modelos...")

X_train_reshaped = X_train.reshape(-1, seq_length, 1)
X_test_reshaped = X_test.reshape(-1, seq_length, 1)

results = {}

for name, model in models.items():
    print(f"\n  Treinando {name}...")

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Treinar e medir tempo
    start_time = time.time()

    history = model.fit(
        X_train_reshaped, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        verbose=0
    )

    train_time = time.time() - start_time

    # Avaliar
    test_loss, test_mae = model.evaluate(X_test_reshaped, y_test, verbose=0)

    # Inferência
    start_time = time.time()
    _ = model.predict(X_test_reshaped[:100], verbose=0)
    inference_time = (time.time() - start_time) / 100 * 1000  # ms por sample

    results[name] = {
        'history': history,
        'train_time': train_time,
        'test_mae': test_mae,
        'inference_time': inference_time,
        'params': model.count_params()
    }

    print(f"    Train time: {train_time:.2f}s")
    print(f"    Test MAE: {test_mae:.4f}")
    print(f"    Inference: {inference_time:.2f}ms/sample")

# ─── 4. ANÁLISE COMPARATIVA ───
print("\n" + "="*70)
print("📊 ANÁLISE COMPARATIVA")
print("="*70)

print("\n⏱️  TEMPO DE TREINO:")
for name in ['SimpleRNN', 'GRU', 'LSTM']:
    t = results[name]['train_time']
    print(f"  {name:10s}: {t:.2f}s")

baseline_time = results['SimpleRNN']['train_time']
print(f"\n  GRU vs SimpleRNN: {results['GRU']['train_time']/baseline_time:.2f}x")
print(f"  LSTM vs SimpleRNN: {results['LSTM']['train_time']/baseline_time:.2f}x")
print(f"  GRU vs LSTM: {results['GRU']['train_time']/results['LSTM']['train_time']:.2f}x mais rápido")

print("\n🎯 ACCURACY (MAE):")
for name in ['SimpleRNN', 'GRU', 'LSTM']:
    mae = results[name]['test_mae']
    print(f"  {name:10s}: {mae:.4f}")

print("\n💾 PARÂMETROS:")
for name in ['SimpleRNN', 'GRU', 'LSTM']:
    params = results[name]['params']
    print(f"  {name:10s}: {params:,}")

print("\n⚡ INFERÊNCIA:")
for name in ['SimpleRNN', 'GRU', 'LSTM']:
    inf_time = results[name]['inference_time']
    print(f"  {name:10s}: {inf_time:.2f}ms/sample")

# ─── 5. VISUALIZAR ───
print("\n📈 Visualizando comparação...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Training loss
for name, color in zip(['SimpleRNN', 'GRU', 'LSTM'], ['red', 'orange', 'steelblue']):
    axes[0, 0].plot(results[name]['history'].history['loss'], label=name, color=color, linewidth=2)
    axes[0, 0].plot(results[name]['history'].history['val_loss'], linestyle='--', color=color, alpha=0.6)

axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Training MAE
for name, color in zip(['SimpleRNN', 'GRU', 'LSTM'], ['red', 'orange', 'steelblue']):
    axes[0, 1].plot(results[name]['history'].history['mae'], label=name, color=color, linewidth=2)

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('Training MAE', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Test MAE comparison
names = ['SimpleRNN', 'GRU', 'LSTM']
maes = [results[name]['test_mae'] for name in names]
colors = ['red', 'orange', 'steelblue']

axes[0, 2].bar(names, maes, color=colors, alpha=0.7)
axes[0, 2].set_ylabel('Test MAE')
axes[0, 2].set_title('Final Performance', fontsize=12, fontweight='bold')
axes[0, 2].grid(axis='y', alpha=0.3)

# Training time
times = [results[name]['train_time'] for name in names]

axes[1, 0].bar(names, times, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Time (s)')
axes[1, 0].set_title('Training Time', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Parameters
params = [results[name]['params'] for name in names]

axes[1, 1].bar(names, params, color=colors, alpha=0.7)
axes[1, 1].set_ylabel('Parameters')
axes[1, 1].set_title('Model Size', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

# Inference time
inf_times = [results[name]['inference_time'] for name in names]

axes[1, 2].bar(names, inf_times, color=colors, alpha=0.7)
axes[1, 2].set_ylabel('Time (ms/sample)')
axes[1, 2].set_title('Inference Speed', fontsize=12, fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.suptitle('GRU vs LSTM vs SimpleRNN - Comprehensive Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('gru_lstm_comparison.png', dpi=150)
print("✅ Comparação salva: gru_lstm_comparison.png")

# ─── 6. ARQUITETURA INTERNA ───
print("\n" + "="*70)
print("🔬 ARQUITETURA INTERNA")
print("="*70)

print("\n📐 SimpleRNN:")
print("  • 1 gate: Apenas hidden state")
print("  • Fórmula: h_t = tanh(W_h * h_{t-1} + W_x * x_t)")
print("  • Problema: Vanishing gradient")

print("\n🔒 LSTM (3 gates):")
print("  • Forget gate: o que esquecer do cell state")
print("  • Input gate: o que adicionar ao cell state")
print("  • Output gate: o que expor como hidden state")
print("  • Cell state: memória de longo prazo")
print("  • Parâmetros: 4 * (hidden_size * (hidden_size + input_size))")

print("\n⚡ GRU (2 gates):")
print("  • Reset gate: quanto do passado ignorar")
print("  • Update gate: balanceia passado e presente")
print("  • Sem cell state separado (mais simples)")
print("  • Parâmetros: 3 * (hidden_size * (hidden_size + input_size))")
print("  • ~25% menos parâmetros que LSTM")

print("\n💡 QUANDO USAR:")
print("  • SimpleRNN: Baseline, sequências curtas")
print("  • GRU: Default, mais rápido, menos memória")
print("  • LSTM: Sequências longas, tarefas complexas")

print("\n📊 TRADE-OFFS:")
print("  GRU:")
print("    ✓ Mais rápido (~20-30%)")
print("    ✓ Menos parâmetros (~25%)")
print("    ✓ Mais fácil de treinar")
print("    ✗ Menos expressivo")

print("\n  LSTM:")
print("    ✓ Mais expressivo (cell state separado)")
print("    ✓ Melhor em sequências muito longas")
print("    ✗ Mais lento")
print("    ✗ Mais parâmetros")

print("\n🎯 RECOMENDAÇÃO:")
print("  1. Comece com GRU (mais rápido)")
print("  2. Se performance não for suficiente, tente LSTM")
print("  3. Se sequências muito longas (>100), prefira LSTM")
print("  4. Para produção, considere velocidade vs accuracy")

print("\n✅ COMPARAÇÃO COMPLETA!")
