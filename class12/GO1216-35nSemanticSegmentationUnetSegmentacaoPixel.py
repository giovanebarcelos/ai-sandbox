# GO1216-35nSemanticSegmentationUnetSegmentaçãoPixel
# ══════════════════════════════════════════════════════════════════
# SEMANTIC SEGMENTATION COM U-NET
# Classificar cada pixel da imagem
# ══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
import matplotlib.pyplot as plt

print("🎨 SEMANTIC SEGMENTATION COM U-NET")
print("=" * 70)

# ─── 1. GERAR DATASET SINTÉTICO ───
print("\n📦 Gerando imagens e máscaras de segmentação...")

def generate_segmentation_data(num_samples=200, img_size=128):
    X = []
    y = []

    for _ in range(num_samples):
        # Imagem com 3 classes: background, círculo, quadrado
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Classe 1: Círculo
        center = (np.random.randint(30, img_size-30), np.random.randint(30, img_size-30))
        radius = np.random.randint(15, 25)
        cv2.circle(img, center, radius, (255, 0, 0), -1)
        cv2.circle(mask, center, radius, 1, -1)

        # Classe 2: Quadrado
        x = np.random.randint(20, img_size-40)
        y = np.random.randint(20, img_size-40)
        size = np.random.randint(20, 35)
        cv2.rectangle(img, (x, y), (x+size, y+size), (0, 255, 0), -1)
        cv2.rectangle(mask, (x, y), (x+size, y+size), 2, -1)

        X.append(img.astype('float32') / 255.0)
        y.append(mask)

    return np.array(X), np.array(y)

X_data, y_data = generate_segmentation_data(500, img_size=128)

print(f"  Images: {X_data.shape}")
print(f"  Masks: {y_data.shape}")
print(f"  Classes: 0=Background, 1=Circle, 2=Square")

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. CONSTRUIR U-NET ───
print("\n🏗️ Construindo U-Net...")

def unet_model(input_shape=(128, 128, 3), num_classes=3):
    inputs = Input(input_shape)

    # Encoder (contracting path)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder (expansive path) com skip connections
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    return Model(inputs=[inputs], outputs=[outputs], name='U-Net')

model = unet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 3. TREINAR ───
print("\n🚀 Treinando U-Net...")

history = model.fit(
    X_train,
    y_train.reshape(-1, 128, 128, 1),
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    verbose=0
)

test_loss, test_acc = model.evaluate(X_test, y_test.reshape(-1, 128, 128, 1), verbose=0)

print(f"  Test Accuracy (pixel-wise): {test_acc:.4f}")

# ─── 4. PREDIÇÕES ───
print("\n🔮 Gerando segmentações...")

predictions = model.predict(X_test[:6], verbose=0)
pred_masks = predictions.argmax(axis=-1)

# ─── 5. VISUALIZAR ───
print("\n📊 Visualizando resultados...")

fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for i in range(6):
    # Imagem original
    axes[0, i].imshow(X_test[i])
    axes[0, i].set_title('Original', fontsize=10)
    axes[0, i].axis('off')

    # Ground truth
    axes[1, i].imshow(y_test[i], cmap='tab10', vmin=0, vmax=2)
    axes[1, i].set_title('Ground Truth', fontsize=10)
    axes[1, i].axis('off')

    # Predição
    axes[2, i].imshow(pred_masks[i], cmap='tab10', vmin=0, vmax=2)
    axes[2, i].set_title('Prediction', fontsize=10, color='green', fontweight='bold')
    axes[2, i].axis('off')

plt.suptitle('U-Net Semantic Segmentation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('unet_segmentation.png', dpi=150)
print("✅ Segmentação salva: unet_segmentation.png")

# ─── 6. MÉTRICAS POR CLASSE ───
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true_flat = y_test.flatten()
y_pred_flat = pred_masks.flatten()

cm = confusion_matrix(y_true_flat, y_pred_flat)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Background', 'Circle', 'Square'],
            yticklabels=['Background', 'Circle', 'Square'])
plt.title('Pixel-wise Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('unet_confusion_matrix.png', dpi=150)
print("✅ Confusion matrix salva: unet_confusion_matrix.png")

print("\n💡 U-NET ARCHITECTURE:")
print("  • Encoder-Decoder simétrico")
print("  • Skip connections: Preservam detalhes espaciais")
print("  • Contracting path: Captura contexto")
print("  • Expansive path: Localização precisa")

print("\n🎯 APLICAÇÕES:")
print("  • Medical: Segmentação de tumores, órgãos")
print("  • Autonomous Driving: Road segmentation")
print("  • Satellite: Land use classification")
print("  • Agriculture: Crop segmentation")

print("\n📊 VARIAÇÕES:")
print("  • U-Net++: Nested skip connections")
print("  • Attention U-Net: Attention gates")
print("  • ResU-Net: Residual connections")
print("  • DeepLab: Atrous convolutions")

print("\n✅ SEMANTIC SEGMENTATION COMPLETO!")
