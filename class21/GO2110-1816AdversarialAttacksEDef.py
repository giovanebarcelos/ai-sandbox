# GO2110-1816AdversarialAttacksEDef
# ═══════════════════════════════════════════════════════════════════
# ADVERSARIAL ATTACKS E DEFESA
# FGSM (Fast Gradient Sign Method) e Adversarial Training
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ─── 1. CONFIGURAÇÃO E DADOS ───
print("📦 Carregando e preparando MNIST...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Adicionar dimensão de canal
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(f"  Treino: {x_train.shape}, Teste: {x_test.shape}")

# Criar datasets TF
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

# ─── 2. MODELO CNN SIMPLES ───
def create_model():
    """Modelo CNN para classificação MNIST"""
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

# ─── 3. TREINAR MODELO NORMAL ───
print("\n🔨 Treinando modelo BASE (sem defesa)...")

model_base = create_model()
model_base.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_base = model_base.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds,
    verbose=1
)

# Avaliar
loss_base, acc_base = model_base.evaluate(test_ds, verbose=0)
print(f"\n✅ Modelo BASE - Accuracy: {acc_base:.4f}")

# ─── 4. GERAR ADVERSARIAL EXAMPLES (FGSM) ───
def create_adversarial_pattern(model, images, labels):
    """
    FGSM (Fast Gradient Sign Method):
    Adiciona perturbação na direção do gradiente da loss
    """
    images = tf.cast(images, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    # Gradiente da loss em relação à imagem
    gradient = tape.gradient(loss, images)

    # Sinal do gradiente
    signed_grad = tf.sign(gradient)

    return signed_grad

def generate_adversarial_examples(model, images, labels, epsilon=0.1):
    """
    Gera exemplos adversariais usando FGSM

    Args:
        epsilon: magnitude da perturbação (0.0-1.0)
    """
    perturbations = create_adversarial_pattern(model, images, labels)
    adversarial_images = images + epsilon * perturbations
    adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
    return adversarial_images

print("\n⚔️ Gerando exemplos adversariais (FGSM)...")

# Selecionar batch de teste
test_batch = list(test_ds.take(1))[0]
clean_images, clean_labels = test_batch

# Gerar adversariais com diferentes epsilons
epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]
accuracies = []

print("\nTestando robustez com diferentes epsilons:")
for eps in epsilons:
    if eps == 0:
        adv_images = clean_images
    else:
        adv_images = generate_adversarial_examples(model_base, clean_images, clean_labels, eps)

    predictions = model_base.predict(adv_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == clean_labels.numpy())
    accuracies.append(accuracy)

    print(f"  Epsilon={eps:.2f}: Accuracy={accuracy:.4f} ({accuracy*100:.1f}%)")

# Visualizar impacto
plt.figure(figsize=(12, 5))

# Gráfico de accuracy vs epsilon
plt.subplot(1, 2, 1)
plt.plot(epsilons, accuracies, 'o-', linewidth=2, markersize=8, color='red')
plt.axhline(y=acc_base, color='green', linestyle='--', label='Baseline (clean)')
plt.xlabel('Epsilon (perturbação)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Robustez a Ataques Adversariais', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([0, 1])

# Exemplo visual
plt.subplot(1, 2, 2)
eps_visual = 0.3
adv_example = generate_adversarial_examples(model_base, clean_images[:1], clean_labels[:1], eps_visual)

# Original
orig_img = clean_images[0].numpy().squeeze()
adv_img = adv_example[0].numpy().squeeze()
perturbation = (adv_img - orig_img) * 10  # Amplificar para visualizar

# Predições
orig_pred = np.argmax(model_base.predict(clean_images[:1], verbose=0))
adv_pred = np.argmax(model_base.predict(adv_example, verbose=0))

# Criar composição
composite = np.hstack([orig_img, perturbation, adv_img])
plt.imshow(composite, cmap='gray')
plt.title(f'Original ({orig_pred}) | Perturbação (×10) | Adversarial ({adv_pred})', 
          fontsize=10, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('adversarial_attack_analysis.png', dpi=150)
print("\n  ✓ Análise salva: adversarial_attack_analysis.png")

# ─── 5. ADVERSARIAL TRAINING (DEFESA) ───
print("\n🛡️ Treinando modelo ROBUSTO com Adversarial Training...")

model_robust = create_model()
model_robust.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Função de treinamento adversarial
@tf.function
def train_step_adversarial(model, images, labels, epsilon=0.1):
    """
    Um step de treinamento com exemplos adversariais
    """
    # 1. Gerar adversariais
    adv_images = generate_adversarial_examples(model, images, labels, epsilon)

    # 2. Mix: 50% clean + 50% adversarial
    mixed_images = tf.concat([images, adv_images], axis=0)
    mixed_labels = tf.concat([labels, labels], axis=0)

    # 3. Treinar no mix
    with tf.GradientTape() as tape:
        predictions = model(mixed_images, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(mixed_labels, predictions)
        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Treinar com adversarial training
epochs_robust = 5
print(f"\nTreinando por {epochs_robust} épocas...")

for epoch in range(epochs_robust):
    print(f"\nÉpoca {epoch+1}/{epochs_robust}")
    epoch_losses = []

    for batch_idx, (images, labels) in enumerate(train_ds):
        loss = train_step_adversarial(model_robust, images, labels, epsilon=0.1)
        epoch_losses.append(loss.numpy())

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss={loss.numpy():.4f}")

    avg_loss = np.mean(epoch_losses)

    # Avaliar em dados limpos
    clean_acc = model_robust.evaluate(test_ds, verbose=0)[1]
    print(f"  Época {epoch+1}: Loss médio={avg_loss:.4f}, Accuracy (clean)={clean_acc:.4f}")

print("\n✅ Treinamento adversarial concluído!")

# ─── 6. COMPARAR MODELO BASE vs ROBUSTO ───
print("\n📊 Comparando modelos em dados adversariais...")

comparison_results = {
    'epsilon': [],
    'base_accuracy': [],
    'robust_accuracy': []
}

print("\nRobustez comparativa:")
for eps in epsilons:
    # Gerar adversariais
    if eps == 0:
        adv_test_images = clean_images
    else:
        adv_test_images = generate_adversarial_examples(model_base, clean_images, clean_labels, eps)

    # Avaliar ambos os modelos
    pred_base = model_base.predict(adv_test_images, verbose=0)
    pred_robust = model_robust.predict(adv_test_images, verbose=0)

    acc_base = np.mean(np.argmax(pred_base, axis=1) == clean_labels.numpy())
    acc_robust = np.mean(np.argmax(pred_robust, axis=1) == clean_labels.numpy())

    comparison_results['epsilon'].append(eps)
    comparison_results['base_accuracy'].append(acc_base)
    comparison_results['robust_accuracy'].append(acc_robust)

    improvement = (acc_robust - acc_base) * 100
    print(f"  ε={eps:.2f}: Base={acc_base:.4f}, Robusto={acc_robust:.4f} "
          f"(+{improvement:+.1f}%)")

# Visualizar comparação
plt.figure(figsize=(12, 6))

plt.plot(comparison_results['epsilon'], comparison_results['base_accuracy'], 
         'o-', label='Modelo BASE', linewidth=2, markersize=8, color='red')
plt.plot(comparison_results['epsilon'], comparison_results['robust_accuracy'], 
         's-', label='Modelo ROBUSTO', linewidth=2, markersize=8, color='green')

plt.xlabel('Epsilon (perturbação)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparação: Modelo Base vs Robusto', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('adversarial_defense_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: adversarial_defense_comparison.png")

# ─── 7. VISUALIZAR MÚLTIPLOS EXEMPLOS ───
print("\n🎨 Gerando galeria de exemplos adversariais...")

n_examples = 10
sample_indices = np.random.choice(len(clean_images), n_examples, replace=False)
eps_gallery = 0.2

fig, axes = plt.subplots(3, n_examples, figsize=(20, 6))

for i, idx in enumerate(sample_indices):
    img = clean_images[idx:idx+1]
    label = clean_labels[idx:idx+1]

    # Original
    orig_pred = np.argmax(model_base.predict(img, verbose=0))
    axes[0, i].imshow(img[0, :, :, 0], cmap='gray')
    axes[0, i].set_title(f'{label.numpy()[0]} → {orig_pred}', fontsize=9)
    axes[0, i].axis('off')

    # Adversarial
    adv_img = generate_adversarial_examples(model_base, img, label, eps_gallery)
    adv_pred = np.argmax(model_base.predict(adv_img, verbose=0))
    axes[1, i].imshow(adv_img[0, :, :, 0], cmap='gray')
    color = 'red' if adv_pred != label.numpy()[0] else 'green'
    axes[1, i].set_title(f'{label.numpy()[0]} → {adv_pred}', fontsize=9, color=color)
    axes[1, i].axis('off')

    # Perturbação (amplificada)
    perturbation = (adv_img[0, :, :, 0] - img[0, :, :, 0]) * 10
    axes[2, i].imshow(perturbation, cmap='bwr', vmin=-1, vmax=1)
    axes[2, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Adversarial', fontsize=11, fontweight='bold')
axes[2, 0].set_ylabel('Perturbação (×10)', fontsize=11, fontweight='bold')

plt.suptitle(f'Galeria de Ataques Adversariais (ε={eps_gallery})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adversarial_gallery.png', dpi=150)
print("  ✓ Galeria salva: adversarial_gallery.png")

# ─── 8. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ ANÁLISE DE SEGURANÇA ADVERSARIAL CONCLUÍDA!")
print("="*70)

print("\n📊 Estatísticas Finais:")
print(f"\nModelo BASE:")
print(f"  - Accuracy (clean): {acc_base:.4f}")
print(f"  - Accuracy (ε=0.3): {comparison_results['base_accuracy'][-1]:.4f}")
print(f"  - Queda: {(acc_base - comparison_results['base_accuracy'][-1])*100:.1f}%")

robust_clean_acc = comparison_results['robust_accuracy'][0]
robust_adv_acc = comparison_results['robust_accuracy'][-1]
print(f"\nModelo ROBUSTO:")
print(f"  - Accuracy (clean): {robust_clean_acc:.4f}")
print(f"  - Accuracy (ε=0.3): {robust_adv_acc:.4f}")
print(f"  - Queda: {(robust_clean_acc - robust_adv_acc)*100:.1f}%")

improvement = (robust_adv_acc - comparison_results['base_accuracy'][-1]) * 100
print(f"\n💡 Melhoria com Adversarial Training: +{improvement:.1f}% em ε=0.3")

print("\n📁 Arquivos gerados:")
print("  - adversarial_attack_analysis.png - Impacto do epsilon")
print("  - adversarial_defense_comparison.png - Base vs Robusto")
print("  - adversarial_gallery.png - Galeria de exemplos")

print("\n⚠️ Lições Aprendidas:")
print("  - Modelos DNNs são vulneráveis a perturbações imperceptíveis")
print("  - Adversarial training melhora robustez significativamente")
print("  - Trade-off: robustez pode reduzir accuracy em dados limpos")
print("  - Defesa em profundidade: combine múltiplas técnicas")
