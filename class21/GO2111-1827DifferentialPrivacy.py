# GO2111-1827DifferentialPrivacy
# ═══════════════════════════════════════════════════════════════════
# DIFFERENTIAL PRIVACY - TREINAR MODELO COM PRIVACIDADE
# TensorFlow Privacy (DP-SGD)
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# ─── 1. CONFIGURAÇÃO ───
print("⚙️ Configurando Differential Privacy...")

# Hiperparâmetros de privacidade
L2_NORM_CLIP = 1.0  # Clipping de gradientes
NOISE_MULTIPLIER = 0.8  # Quantidade de ruído
NUM_MICROBATCHES = 256  # Microbatches para DP-SGD
LEARNING_RATE = 0.15

# Training
EPOCHS = 15
BATCH_SIZE = 256

# ─── 2. DADOS ───
print("\n📦 Carregando MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten (DP-SGD funciona melhor com modelos simples)
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

print(f"  Treino: {x_train_flat.shape}")
print(f"  Teste: {x_test_flat.shape}")

# ─── 3. MODELO SIMPLES (MLP) ───
def create_mlp_model():
    """Modelo MLP simples para DP training"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# ─── 4. TREINAR SEM PRIVACIDADE (BASELINE) ───
print("\n🔨 Treinando modelo SEM privacidade (baseline)...")

model_baseline = create_mlp_model()
model_baseline.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_baseline = model_baseline.fit(
    x_train_flat, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_flat, y_test),
    verbose=1
)

baseline_acc = history_baseline.history['val_accuracy'][-1]
print(f"\n✅ Modelo BASELINE - Accuracy: {baseline_acc:.4f}")

# ─── 5. TREINAR COM DIFFERENTIAL PRIVACY ───
print("\n🔐 Treinando modelo COM privacidade (DP-SGD)...")

model_private = create_mlp_model()

# Optimizer com DP
dp_optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
    l2_norm_clip=L2_NORM_CLIP,
    noise_multiplier=NOISE_MULTIPLIER,
    num_microbatches=NUM_MICROBATCHES,
    learning_rate=LEARNING_RATE
)

model_private.compile(
    optimizer=dp_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_private = model_private.fit(
    x_train_flat, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_flat, y_test),
    verbose=1
)

private_acc = history_private.history['val_accuracy'][-1]
print(f"\n✅ Modelo COM DP - Accuracy: {private_acc:.4f}")

# ─── 6. CALCULAR GARANTIAS DE PRIVACIDADE (ε, δ) ───
print("\n📊 Calculando garantias de privacidade (ε, δ)...")

# Parâmetros
n = len(x_train_flat)
sampling_probability = BATCH_SIZE / n
steps = (n // BATCH_SIZE) * EPOCHS

# Calcular epsilon (privacidade loss)
epsilon, opt_order = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
    n=n,
    batch_size=BATCH_SIZE,
    noise_multiplier=NOISE_MULTIPLIER,
    epochs=EPOCHS,
    delta=1e-5  # δ padrão (recomendado < 1/n)
)

print(f"\n🔒 Garantias de Privacidade:")
print(f"  - Epsilon (ε): {epsilon:.2f}")
print(f"  - Delta (δ): 1e-5")
print(f"  - Optimal RDP order: {opt_order}")

print(f"\n💡 Interpretação:")
if epsilon < 1:
    print(f"  ✅ Privacidade FORTE (ε < 1)")
elif epsilon < 10:
    print(f"  ⚡ Privacidade MODERADA (1 ≤ ε < 10)")
else:
    print(f"  ⚠️ Privacidade FRACA (ε ≥ 10)")

print(f"\n  Com ε={epsilon:.2f}, a probabilidade de vazamento de informação")
print(f"  individual é limitada por e^{epsilon:.2f} ≈ {np.exp(epsilon):.2f}×")

# ─── 7. COMPARAR PERFORMANCE ───
print("\n📊 Comparando modelos...")

accuracy_drop = (baseline_acc - private_acc) * 100
print(f"\n  Baseline Accuracy: {baseline_acc:.4f}")
print(f"  DP Accuracy: {private_acc:.4f}")
print(f"  Queda: {accuracy_drop:.2f}% (trade-off privacidade)")

# Visualizar curvas de treinamento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
epochs_range = range(1, EPOCHS + 1)
axes[0].plot(epochs_range, history_baseline.history['val_accuracy'], 
             'o-', label='SEM Privacidade', linewidth=2, markersize=6, color='blue')
axes[0].plot(epochs_range, history_private.history['val_accuracy'], 
             's-', label='COM Privacidade (DP-SGD)', linewidth=2, markersize=6, color='green')
axes[0].set_title('Accuracy de Validação', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(epochs_range, history_baseline.history['loss'], 
             'o-', label='SEM Privacidade', linewidth=2, markersize=6, color='blue')
axes[1].plot(epochs_range, history_private.history['loss'], 
             's-', label='COM Privacidade (DP-SGD)', linewidth=2, markersize=6, color='green')
axes[1].set_title('Loss de Treinamento', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('differential_privacy_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: differential_privacy_comparison.png")

# ─── 8. ANÁLISE DE SENSIBILIDADE ───
print("\n🔬 Análise de sensibilidade ao ruído...")

# Testar diferentes noise_multipliers
noise_multipliers = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0]
results = []

print("\nTestando diferentes níveis de ruído:")
for nm in noise_multipliers:
    # Calcular epsilon para este noise_multiplier
    eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=n,
        batch_size=BATCH_SIZE,
        noise_multiplier=nm,
        epochs=EPOCHS,
        delta=1e-5
    )

    results.append({'noise_multiplier': nm, 'epsilon': eps})
    print(f"  Noise={nm:.1f}: ε={eps:.2f} ({'FORTE' if eps < 1 else 'MODERADO' if eps < 10 else 'FRACO'})")

# Visualizar trade-off privacidade-ruído
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(results_df['noise_multiplier'], results_df['epsilon'], 
         'o-', linewidth=2, markersize=8, color='purple')
plt.axhline(y=1, color='green', linestyle='--', label='ε=1 (privacidade forte)')
plt.axhline(y=10, color='orange', linestyle='--', label='ε=10 (privacidade fraca)')
plt.xlabel('Noise Multiplier', fontsize=12)
plt.ylabel('Epsilon (ε)', fontsize=12)
plt.title('Trade-off: Ruído vs Privacidade', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('privacy_noise_tradeoff.png', dpi=150)
print("\n  ✓ Trade-off salvo: privacy_noise_tradeoff.png")

# ─── 9. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ ANÁLISE DE DIFFERENTIAL PRIVACY CONCLUÍDA!")
print("="*70)

print("\n📊 Resumo:")
print(f"\nPRIVACIDADE:")
print(f"  - Epsilon (ε): {epsilon:.2f}")
print(f"  - Delta (δ): 1e-5")
print(f"  - Noise multiplier: {NOISE_MULTIPLIER}")
print(f"  - L2 norm clip: {L2_NORM_CLIP}")

print(f"\nPERFORMANCE:")
print(f"  - Baseline accuracy: {baseline_acc:.4f}")
print(f"  - DP accuracy: {private_acc:.4f}")
print(f"  - Trade-off: {accuracy_drop:.2f}% de queda")

print("\n📁 Arquivos gerados:")
print("  - differential_privacy_comparison.png")
print("  - privacy_noise_tradeoff.png")

print("\n💡 Boas Práticas:")
print("  - Epsilon < 1: privacidade forte (recomendado para dados sensíveis)")
print("  - Noise multiplier ↑ → privacidade ↑, accuracy ↓")
print("  - L2 clipping previne gradientes grandes de vazarem info")
print("  - Microbatches melhoram qualidade do ruído adicionado")

print("\n🔐 Aplicações:")
print("  - Dados médicos (HIPAA compliance)")
print("  - Dados financeiros (PCI-DSS)")
print("  - Dados pessoais (GDPR, LGPD)")
print("  - Federated learning (Google, Apple)")
