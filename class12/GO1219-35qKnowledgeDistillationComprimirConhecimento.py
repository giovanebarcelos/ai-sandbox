# GO1219-35qKnowledgeDistillationComprimirConhecimento
# ══════════════════════════════════════════════════════════════════
# KNOWLEDGE DISTILLATION - COMPRIMIR MODELOS
# Transferir conhecimento de modelo grande (teacher) para pequeno (student)
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Lambda
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

print("🎯 KNOWLEDGE DISTILLATION")
print("=" * 70)

# ─── 1. CARREGAR DADOS ───
print("\n📦 Carregando MNIST...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Subset
X_train = X_train[:10000]
y_train = y_train[:10000]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. TREINAR TEACHER (GRANDE) ───
print("\n🏫 Treinando Teacher Model (grande)...")

teacher = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
], name='Teacher')

teacher.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

teacher.fit(X_train, y_train, epochs=15, batch_size=128, verbose=0, validation_split=0.2)

teacher_acc = teacher.evaluate(X_test, y_test, verbose=0)[1]

print(f"  Parâmetros: {teacher.count_params():,}")
print(f"  Accuracy: {teacher_acc:.4f}")

# ─── 3. STUDENT (PEQUENO) - SEM DISTILLATION ───
print("\n🎓 Treinando Student Model baseline (sem distillation)...")

student_baseline = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
], name='Student_Baseline')

student_baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

student_baseline.fit(X_train, y_train, epochs=15, batch_size=128, verbose=0, validation_split=0.2)

student_baseline_acc = student_baseline.evaluate(X_test, y_test, verbose=0)[1]

print(f"  Parâmetros: {student_baseline.count_params():,}")
print(f"  Accuracy: {student_baseline_acc:.4f}")

# ─── 4. DISTILLATION LOSS ───
print("\n🔥 Implementando Distillation Loss...")

def distillation_loss(y_true, y_pred, teacher_pred, temperature=3.0, alpha=0.1):
    """
    Loss combinado:
    - Hard targets: Cross-entropy com labels verdadeiros
    - Soft targets: KL divergence com predições do teacher
    """
    # Hard loss (com labels)
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Soft loss (com teacher)
    # Aplicar temperatura para suavizar distribuições
    teacher_soft = tf.nn.softmax(teacher_pred / temperature)
    student_soft = tf.nn.softmax(y_pred / temperature)

    soft_loss = tf.keras.losses.kullback_leibler_divergence(teacher_soft, student_soft)

    # Combinar
    return alpha * hard_loss + (1 - alpha) * (temperature**2) * soft_loss

print("  ✓ Distillation loss definido (temperature=3.0, alpha=0.1)")

# ─── 5. TREINAR STUDENT COM DISTILLATION ───
print("\n🎓🔥 Treinando Student com Knowledge Distillation...")

# Criar student
student_distilled = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='linear')  # Logits (antes de softmax)
], name='Student_Distilled')

# Gerar soft labels do teacher
teacher_logits = teacher.predict(X_train, verbose=0)

# Custom training loop
optimizer = tf.keras.optimizers.Adam()
loss_metric = tf.keras.metrics.Mean()

temperature = 3.0
alpha = 0.1

for epoch in range(15):
    for i in range(0, len(X_train), 128):
        batch_x = X_train[i:i+128]
        batch_y = y_train[i:i+128]
        batch_teacher = teacher_logits[i:i+128]

        with tf.GradientTape() as tape:
            student_logits = student_distilled(batch_x, training=True)

            # Hard loss
            student_probs = tf.nn.softmax(student_logits)
            hard_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, student_probs)

            # Soft loss
            teacher_soft = tf.nn.softmax(batch_teacher / temperature)
            student_soft = tf.nn.softmax(student_logits / temperature)
            soft_loss = tf.keras.losses.kullback_leibler_divergence(teacher_soft, student_soft)

            # Total loss
            loss = alpha * hard_loss + (1 - alpha) * (temperature**2) * soft_loss
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, student_distilled.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_distilled.trainable_variables))
        loss_metric.update_state(loss)

    if epoch % 5 == 0:
        print(f"  Epoch {epoch+1}/15 - Loss: {loss_metric.result():.4f}")
    loss_metric.reset_states()

print("  ✓ Distillation concluído")

# Avaliar
student_distilled_logits = student_distilled.predict(X_test, verbose=0)
student_distilled_probs = tf.nn.softmax(student_distilled_logits).numpy()
student_distilled_acc = np.mean(student_distilled_probs.argmax(axis=1) == y_test)

print(f"  Accuracy: {student_distilled_acc:.4f}")

# ─── 6. COMPARAR ───
print("\n" + "="*70)
print("📊 COMPARAÇÃO DE MODELOS")
print("="*70)

print(f"\n🏫 TEACHER (grande):")
print(f"  Parâmetros: {teacher.count_params():,}")
print(f"  Accuracy: {teacher_acc:.4f}")

print(f"\n🎓 STUDENT (pequeno) - Baseline:")
print(f"  Parâmetros: {student_baseline.count_params():,}")
print(f"  Accuracy: {student_baseline_acc:.4f}")
print(f"  Gap vs Teacher: {(teacher_acc - student_baseline_acc)*100:.1f}%")

print(f"\n🎓🔥 STUDENT (pequeno) - Distilled:")
print(f"  Parâmetros: {student_distilled.count_params():,}")
print(f"  Accuracy: {student_distilled_acc:.4f}")
print(f"  Gap vs Teacher: {(teacher_acc - student_distilled_acc)*100:.1f}%")
print(f"  Melhoria vs Baseline: +{(student_distilled_acc - student_baseline_acc)*100:.1f}%")

# Compressão
compression = teacher.count_params() / student_distilled.count_params()
print(f"\n📊 COMPRESSÃO: {compression:.1f}x menor")

# ─── 7. VISUALIZAR ───
print("\n📈 Visualizando comparação...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
models = ['Teacher', 'Student\nBaseline', 'Student\nDistilled']
accs = [teacher_acc, student_baseline_acc, student_distilled_acc]
colors = ['blue', 'orange', 'green']

axes[0].bar(models, accs, color=colors, alpha=0.7)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Performance', fontsize=12, fontweight='bold')
axes[0].set_ylim(0.9, 1.0)
axes[0].grid(axis='y', alpha=0.3)

for i, acc in enumerate(accs):
    axes[0].text(i, acc + 0.002, f'{acc:.3f}', ha='center', fontweight='bold')

# Parâmetros
params = [teacher.count_params(), student_baseline.count_params(), student_distilled.count_params()]

axes[1].bar(models, params, color=colors, alpha=0.7)
axes[1].set_ylabel('Parameters', fontsize=12)
axes[1].set_title('Model Size', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, p in enumerate(params):
    axes[1].text(i, p + 5000, f'{p:,}', ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Knowledge Distillation Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('knowledge_distillation.png', dpi=150)
print("✅ Comparação salva: knowledge_distillation.png")

print("\n💡 KNOWLEDGE DISTILLATION:")
print("  • Teacher: Modelo grande e preciso")
print("  • Student: Modelo pequeno e rápido")
print("  • Soft Targets: Distribuições suaves do teacher")
print("  • Temperature: Suaviza predições (típico: 2-5)")

print("\n🎯 BENEFÍCIOS:")
print("  • Student aprende melhor que treino direto")
print("  • Compressão: 10-100x menos parâmetros")
print("  • Deploy: Mobile, edge devices")
print("  • Latência: Inferência mais rápida")

print("\n📚 VARIAÇÕES:")
print("  • Self-Distillation: Teacher = Student")
print("  • Multi-Teacher: Múltiplos teachers")
print("  • Feature Distillation: Match intermediate layers")
print("  • Online Distillation: Treinar simultaneamente")

print("\n✅ KNOWLEDGE DISTILLATION COMPLETO!")
