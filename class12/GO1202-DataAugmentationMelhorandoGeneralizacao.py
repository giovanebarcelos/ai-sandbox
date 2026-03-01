# GO1202-DataAugmentationMelhorandoGeneralização
# ═══════════════════════════════════════════════════════════════════
# DATA AUGMENTATION - MELHORANDO GENERALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ─── 1. CARREGAR DADOS ───
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# ─── 2. CRIAR ImageDataGenerator ───
datagen = ImageDataGenerator(
    rotation_range=10,           # Rotação ±10 graus
    width_shift_range=0.1,       # Deslocamento horizontal 10%
    height_shift_range=0.1,      # Deslocamento vertical 10%
    zoom_range=0.1,              # Zoom 90-110%
    shear_range=0.1,             # Cisalhamento
    fill_mode='nearest'          # Preencher pixels vazios
)

# Fit no train set
datagen.fit(x_train)

# ─── 3. VISUALIZAR AUGMENTATIONS ───
# Pegar uma imagem e gerar variações
sample_img = x_train[0:1]  # Shape: (1, 28, 28, 1)

fig, axes = plt.subplots(3, 5, figsize=(12, 7))
axes = axes.ravel()

for i in range(15):
    # Gerar variação augmentada
    augmented = datagen.flow(sample_img, batch_size=1)[0][0]
    axes[i].imshow(augmented.reshape(28, 28), cmap='gray')
    axes[i].axis('off')

plt.suptitle('Data Augmentation - Variações da Mesma Imagem', fontsize=14)
plt.tight_layout()
plt.show()

# ─── 4. TREINAR COM AUGMENTATION ───
# Modelo simples
model_aug = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='CNN_with_Augmentation')

model_aug.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Treinando COM Data Augmentation...")

# Treinar usando generator
history_aug = model_aug.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    steps_per_epoch=len(x_train) // 128,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

# Avaliar
test_loss_aug, test_acc_aug = model_aug.evaluate(x_test, y_test, verbose=0)
print(f"\nCOM Augmentation - Test Accuracy: {test_acc_aug:.4f}")

# ─── 5. COMPARAÇÃO: COM vs SEM AUGMENTATION ───
print("\nTreinando SEM Data Augmentation (baseline)...")

model_baseline = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='CNN_baseline')

model_baseline.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_baseline = model_baseline.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

test_loss_base, test_acc_base = model_baseline.evaluate(x_test, y_test, verbose=0)
print(f"SEM Augmentation - Test Accuracy: {test_acc_base:.4f}")

# Comparar
print("\n" + "="*50)
print("COMPARAÇÃO")
print("="*50)
print(f"Sem Augmentation: {test_acc_base:.4f}")
print(f"Com Augmentation: {test_acc_aug:.4f}")
print(f"Melhoria: {(test_acc_aug - test_acc_base)*100:.2f}%")

# Plotar comparação
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['loss'], label='Train (sem aug)', linestyle='-')
plt.plot(history_baseline.history['val_loss'], label='Val (sem aug)', linestyle='-')
plt.plot(history_aug.history['loss'], label='Train (com aug)', linestyle='--')
plt.plot(history_aug.history['val_loss'], label='Val (com aug)', linestyle='--')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['accuracy'], label='Train (sem aug)', linestyle='-')
plt.plot(history_baseline.history['val_accuracy'], label='Val (sem aug)', linestyle='-')
plt.plot(history_aug.history['accuracy'], label='Train (com aug)', linestyle='--')
plt.plot(history_aug.history['val_accuracy'], label='Val (com aug)', linestyle='--')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
