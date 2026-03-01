# GO1217-35oAdversarialRobustnessDefesaContraAtaques
# ══════════════════════════════════════════════════════════════════
# ADVERSARIAL ROBUSTNESS - ATAQUES E DEFESAS
# Gerar exemplos adversariais e defender modelos
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

print("⚔️ ADVERSARIAL ROBUSTNESS")
print("=" * 70)

# ─── 1. CARREGAR DADOS ───
print("\n📦 Carregando MNIST...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Subset
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. TREINAR MODELO BASE ───
print("\n🏗️ Treinando modelo base...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
], name='BaseModel')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0, validation_split=0.2)

base_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"  Base accuracy: {base_acc:.4f}")

# ─── 3. GERAR EXEMPLOS ADVERSARIAIS (FGSM) ───
print("\n💣 Gerando exemplos adversariais (FGSM)...")

def fgsm_attack(model, images, labels, epsilon=0.1):
    """
    Fast Gradient Sign Method
    """
    images_tensor = tf.convert_to_tensor(images)
    labels_tensor = tf.convert_to_tensor(labels)

    with tf.GradientTape() as tape:
        tape.watch(images_tensor)
        predictions = model(images_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tensor, predictions)

    # Gradiente em relação à imagem
    gradient = tape.gradient(loss, images_tensor)

    # Perturbação: sinal do gradiente * epsilon
    perturbation = epsilon * tf.sign(gradient)

    # Exemplo adversarial
    adversarial = images_tensor + perturbation
    adversarial = tf.clip_by_value(adversarial, 0, 1)

    return adversarial.numpy()

# Gerar exemplos adversariais
epsilon = 0.1
X_adv = fgsm_attack(model, X_test[:100], y_test[:100], epsilon)

print(f"  Epsilon: {epsilon}")
print(f"  Perturbação média: {np.abs(X_adv - X_test[:100]).mean():.6f}")

# Avaliar em adversariais
adv_acc = model.evaluate(X_adv, y_test[:100], verbose=0)[1]

print(f"  Base accuracy: {base_acc:.4f}")
print(f"  Adversarial accuracy: {adv_acc:.4f}")
print(f"  Drop: {(base_acc - adv_acc)*100:.1f}%")

# ─── 4. ADVERSARIAL TRAINING ───
print("\n🛡️ Treinando com adversarial training...")

model_robust = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
], name='RobustModel')

model_robust.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar com mix de exemplos normais e adversariais
for epoch in range(5):
    # Dados normais
    model_robust.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0)

    # Gerar adversariais do próprio modelo
    X_train_adv = fgsm_attack(model_robust, X_train[:1000], y_train[:1000], epsilon)
    model_robust.fit(X_train_adv, y_train[:1000], epochs=1, batch_size=128, verbose=0)

    if epoch % 2 == 0:
        print(f"  Epoch {epoch+1}/5 concluída")

print("  ✓ Adversarial training concluído")

# Avaliar modelo robusto
robust_base_acc = model_robust.evaluate(X_test[:100], y_test[:100], verbose=0)[1]
robust_adv_acc = model_robust.evaluate(X_adv, y_test[:100], verbose=0)[1]

print(f"\n  Modelo robusto - Base: {robust_base_acc:.4f}")
print(f"  Modelo robusto - Adversarial: {robust_adv_acc:.4f}")
print(f"  Melhoria: +{(robust_adv_acc - adv_acc)*100:.1f}%")

# ─── 5. VISUALIZAR ───
print("\n📊 Visualizando exemplos adversariais...")

fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for i in range(6):
    # Original
    axes[0, i].imshow(X_test[i].squeeze(), cmap='gray')
    pred_orig = model.predict(X_test[i:i+1], verbose=0).argmax()
    axes[0, i].set_title(f'Original: {pred_orig}', fontsize=10)
    axes[0, i].axis('off')

    # Adversarial
    axes[1, i].imshow(X_adv[i].squeeze(), cmap='gray')
    pred_adv = model.predict(X_adv[i:i+1], verbose=0).argmax()
    color = 'red' if pred_adv != y_test[i] else 'green'
    axes[1, i].set_title(f'Adversarial: {pred_adv}', fontsize=10, color=color)
    axes[1, i].axis('off')

    # Diferença (ampliada)
    diff = (X_adv[i] - X_test[i]).squeeze()
    axes[2, i].imshow(diff * 10, cmap='RdBu', vmin=-1, vmax=1)
    axes[2, i].set_title(f'Diff (x10)', fontsize=10)
    axes[2, i].axis('off')

plt.suptitle(f'Adversarial Examples (FGSM, epsilon={epsilon})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adversarial_examples.png', dpi=150)
print("✅ Exemplos salvos: adversarial_examples.png")

# Comparação
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Base Model', 'Robust Model']
clean_accs = [base_acc, robust_base_acc]
adv_accs = [adv_acc, robust_adv_acc]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, clean_accs, width, label='Clean', color='steelblue', alpha=0.7)
ax.bar(x + width/2, adv_accs, width, label='Adversarial', color='orange', alpha=0.7)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Robustness Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('adversarial_robustness_comparison.png', dpi=150)
print("✅ Comparação salva: adversarial_robustness_comparison.png")

print("\n💡 ATAQUES ADVERSARIAIS:")
print("  • FGSM: Fast Gradient Sign Method (rápido)")
print("  • PGD: Projected Gradient Descent (iterativo)")
print("  • C&W: Carlini & Wagner (otimização)")
print("  • DeepFool: Mínima perturbação")

print("\n🛡️ DEFESAS:")
print("  • Adversarial Training: Treinar com exemplos adversariais")
print("  • Defensive Distillation: Suavizar predições")
print("  • Input Transformation: Denoise, compress")
print("  • Certified Defense: Garantias matemáticas")

print("\n⚠️ IMPORTÂNCIA:")
print("  • Segurança: Modelos em produção são vulneráveis")
print("  • Críticos: Carros autônomos, reconhecimento facial")
print("  • Research: Entender limitações de DNNs")

print("\n✅ ADVERSARIAL ROBUSTNESS COMPLETO!")
