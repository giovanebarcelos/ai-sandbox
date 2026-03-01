# GO1222-35t3dCnnVideoClassification
# ══════════════════════════════════════════════════════════════════
# 3D CNN - VIDEO CLASSIFICATION
# Convolução espaço-temporal para vídeos
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

print("🎬 3D CNN - VIDEO CLASSIFICATION")
print("=" * 70)

# ─── 1. GERAR DADOS DE VÍDEO SINTÉTICOS ───
print("\n📦 Gerando vídeos sintéticos...")

def generate_video_data(num_samples=200, frames=16, height=32, width=32):
    """
    Gerar vídeos sintéticos com 2 classes:
    - Classe 0: Movimento horizontal
    - Classe 1: Movimento vertical
    """
    videos = []
    labels = []

    for _ in range(num_samples):
        video = np.zeros((frames, height, width, 1), dtype=np.float32)
        label = np.random.randint(0, 2)

        # Objeto se movendo
        obj_size = 5

        if label == 0:  # Movimento horizontal
            y = height // 2
            for t in range(frames):
                x = int((t / frames) * (width - obj_size))
                video[t, y:y+obj_size, x:x+obj_size, 0] = 1.0
        else:  # Movimento vertical
            x = width // 2
            for t in range(frames):
                y = int((t / frames) * (height - obj_size))
                video[t, y:y+obj_size, x:x+obj_size, 0] = 1.0

        videos.append(video)
        labels.append(label)

    return np.array(videos), np.array(labels)

X_data, y_data = generate_video_data(500, frames=16, height=32, width=32)

print(f"  Videos: {X_data.shape}")
print(f"  Shape: (samples, frames, height, width, channels)")
print(f"  Classes: 0=Horizontal, 1=Vertical")

# Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. CONSTRUIR 3D CNN ───
print("\n🏗️ Construindo 3D CNN...")

model = Sequential([
    # Primeira camada: convolução 3D
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', 
           input_shape=(16, 32, 32, 1)),
    MaxPooling3D(pool_size=(2, 2, 2)),

    # Segunda camada
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    # Flatten e dense
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='3D_CNN')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")
print("\n  Arquitetura 3D CNN:")
for layer in model.layers:
    if 'conv3d' in layer.name or 'max_pooling3d' in layer.name:
        print(f"    {layer.name}: {layer.output_shape}")

# ─── 3. TREINAR ───
print("\n🚀 Treinando 3D CNN...")

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    verbose=0
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"  Test accuracy: {test_acc:.4f}")

# ─── 4. PREDIÇÕES ───
print("\n🔮 Testando predições...")

predictions = model.predict(X_test[:10], verbose=0)
pred_classes = (predictions > 0.5).astype(int).flatten()

for i in range(10):
    true_class = 'Horizontal' if y_test[i] == 0 else 'Vertical'
    pred_class = 'Horizontal' if pred_classes[i] == 0 else 'Vertical'
    status = '✅' if pred_classes[i] == y_test[i] else '❌'

    print(f"  {status} Video {i+1}: True={true_class}, Pred={pred_class}")

# ─── 5. VISUALIZAR VÍDEO ───
print("\n🎥 Visualizando frames de vídeo...")

fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Video horizontal
video_h = X_test[y_test == 0][0]
for i in range(8):
    frame_idx = int(i * 16 / 8)
    axes[0, i].imshow(video_h[frame_idx, :, :, 0], cmap='gray')
    axes[0, i].set_title(f't={frame_idx}', fontsize=8)
    axes[0, i].axis('off')

axes[0, 0].text(-5, 16, 'Horizontal →', fontsize=10, fontweight='bold', 
                rotation=90, va='center')

# Video vertical
video_v = X_test[y_test == 1][0]
for i in range(8):
    frame_idx = int(i * 16 / 8)
    axes[1, i].imshow(video_v[frame_idx, :, :, 0], cmap='gray')
    axes[1, i].set_title(f't={frame_idx}', fontsize=8)
    axes[1, i].axis('off')

axes[1, 0].text(-5, 16, 'Vertical ↓', fontsize=10, fontweight='bold', 
                rotation=90, va='center')

plt.suptitle('3D CNN: Video Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('3d_cnn_video.png', dpi=150)
print("✅ Video frames salvos: 3d_cnn_video.png")

# ─── 6. TRAINING HISTORY ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('3D CNN Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('3d_cnn_training.png', dpi=150)
print("✅ Training salvo: 3d_cnn_training.png")

print("\n💡 3D CNN:")
print("  • Conv3D: Kernel (tempo, altura, largura)")
print("  • Captura movimento temporal")
print("  • Aprendizagem espaço-temporal")
print("  • Input: (batch, frames, height, width, channels)")

print("\n🎯 COMPARAÇÃO:")
print("  • 2D CNN: Frame-by-frame (sem temporal)")
print("  • 2D CNN + LSTM: Processar frames sequencialmente")
print("  • 3D CNN: Convolução direta no espaço-tempo")
print("  • (2+1)D CNN: Decomposição espacial e temporal")

print("\n🏆 ARQUITETURAS:")
print("  • C3D: 3D ConvNets (Tran et al., 2015)")
print("  • I3D: Inflated 3D ConvNet (Carreira, 2017)")
print("  • SlowFast: Two-pathway network (Facebook, 2019)")
print("  • X3D: Efficient 3D networks")

print("\n🎬 APLICAÇÕES:")
print("  • Action Recognition: UCF-101, Kinetics")
print("  • Gesture Recognition: Controle por gestos")
print("  • Sports Analysis: Táticas, movimentos")
print("  • Video Surveillance: Detecção de anomalias")

print("\n📊 DATASETS:")
print("  • UCF-101: 101 action classes")
print("  • Kinetics-700: 700 classes, 650k videos")
print("  • Something-Something: Interações com objetos")
print("  • ActivityNet: 200 classes, 20k videos")

print("\n✅ 3D CNN VIDEO CLASSIFICATION COMPLETO!")
