# GO1213-35kSingleShotDetectorSsdObject
# ══════════════════════════════════════════════════════════════════
# SINGLE SHOT DETECTOR (SSD) - OBJECT DETECTION
# Detectar múltiplos objetos em uma única passagem
# ══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate, Reshape
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("🎯 SINGLE SHOT DETECTOR (SSD)")
print("=" * 70)

# ─── 1. GERAR DATASET SINTÉTICO ───
print("\n📦 Gerando dataset de objetos...")

def generate_object_image(img_size=128, num_objects=3):
    """
    Gera imagem com múltiplos objetos e suas bounding boxes
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
    boxes = []
    labels = []

    for _ in range(num_objects):
        # Gerar objeto aleatório
        obj_type = np.random.randint(0, 3)  # 0: quadrado, 1: círculo, 2: triângulo

        size = np.random.randint(15, 30)
        x = np.random.randint(size, img_size - size)
        y = np.random.randint(size, img_size - size)

        if obj_type == 0:  # Quadrado
            color = (255, 0, 0)
            cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
        elif obj_type == 1:  # Círculo
            color = (0, 255, 0)
            cv2.circle(img, (x, y), size//2, color, -1)
        else:  # Triângulo
            color = (0, 0, 255)
            pts = np.array([[x, y-size//2], [x-size//2, y+size//2], [x+size//2, y+size//2]])
            cv2.fillPoly(img, [pts], color)

        # Bounding box (normalizado)
        x_min = max(0, (x - size//2)) / img_size
        y_min = max(0, (y - size//2)) / img_size
        x_max = min(img_size, (x + size//2)) / img_size
        y_max = min(img_size, (y + size//2)) / img_size

        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(obj_type)

    return img.astype('float32') / 255.0, np.array(boxes), np.array(labels)

# Gerar dataset
num_samples = 500
X_images = []
y_boxes = []
y_labels = []

for _ in range(num_samples):
    img, boxes, labels = generate_object_image(img_size=128, num_objects=2)
    X_images.append(img)

    # Preencher com zeros se menos de 3 objetos
    padded_boxes = np.zeros((3, 4))
    padded_labels = np.zeros(3)

    n = len(boxes)
    padded_boxes[:n] = boxes
    padded_labels[:n] = labels

    y_boxes.append(padded_boxes)
    y_labels.append(padded_labels)

X_images = np.array(X_images)
y_boxes = np.array(y_boxes)
y_labels = np.array(y_labels)

print(f"  Images: {X_images.shape}")
print(f"  Boxes: {y_boxes.shape}")
print(f"  Labels: {y_labels.shape}")

# ─── 2. CONSTRUIR MODELO SSD ───
print("\n🏗️ Construindo SSD model...")

input_img = Input(shape=(128, 128, 3))

# Backbone (feature extraction)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Detection heads (3 objetos * 5 outputs cada: 4 coords + 1 classe)
box_output = Dense(12, activation='sigmoid', name='boxes')(x)  # 3 boxes * 4 coords
label_output = Dense(3, activation='softmax', name='labels')(x)  # 3 labels

model = Model(inputs=input_img, outputs=[box_output, label_output], name='SSD_Simple')

model.compile(
    optimizer='adam',
    loss={
        'boxes': 'mse',
        'labels': 'sparse_categorical_crossentropy'
    },
    loss_weights={'boxes': 1.0, 'labels': 0.5},
    metrics={'labels': 'accuracy'}
)

print(f"  Parâmetros: {model.count_params():,}")

# ─── 3. TREINAR ───
print("\n🚀 Treinando SSD...")

# Reshape boxes para (samples, 12)
y_boxes_flat = y_boxes.reshape(num_samples, -1)

history = model.fit(
    X_images,
    {'boxes': y_boxes_flat, 'labels': y_labels[:, 0]},  # Usar apenas primeiro objeto
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"  Final loss: {history.history['loss'][-1]:.4f}")
print(f"  Label accuracy: {history.history['labels_accuracy'][-1]:.4f}")

# ─── 4. TESTAR DETECÇÃO ───
print("\n🔍 Testando detecção...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

class_names = ['Square', 'Circle', 'Triangle']
colors = ['red', 'green', 'blue']

for idx in range(6):
    ax = axes[idx // 3, idx % 3]

    # Pegar imagem de teste
    test_img = X_images[idx]
    gt_boxes = y_boxes[idx]
    gt_labels = y_labels[idx]

    # Predição
    pred_boxes, pred_labels = model.predict(test_img[np.newaxis, ...], verbose=0)
    pred_boxes = pred_boxes[0].reshape(3, 4)
    pred_label = np.argmax(pred_labels[0])

    # Mostrar imagem
    ax.imshow(test_img)

    # Ground truth boxes (verde)
    for i in range(len(gt_boxes)):
        if gt_boxes[i].sum() > 0:
            x_min, y_min, x_max, y_max = gt_boxes[i]
            rect = patches.Rectangle(
                (x_min*128, y_min*128),
                (x_max - x_min)*128,
                (y_max - y_min)*128,
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)

    # Predicted box (vermelho)
    x_min, y_min, x_max, y_max = pred_boxes[0]  # Primeira predição
    rect = patches.Rectangle(
        (x_min*128, y_min*128),
        (x_max - x_min)*128,
        (y_max - y_min)*128,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    ax.set_title(f'Pred: {class_names[pred_label]}', fontsize=10, fontweight='bold')
    ax.axis('off')

plt.suptitle('SSD Object Detection (Green=GT, Red=Pred)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ssd_detection.png', dpi=150)
print("✅ Detecção salva: ssd_detection.png")

print("\n💡 SSD vs OUTROS DETECTORES:")
print("  • R-CNN: 2-stage (Region Proposal + Classification)")
print("  • Fast R-CNN: ROI Pooling, mais rápido")
print("  • Faster R-CNN: RPN (Region Proposal Network)")
print("  • SSD: Single-shot, múltiplas escalas, real-time")
print("  • YOLO: Grid-based, extremamente rápido")
print("  • RetinaNet: Focal Loss, melhor accuracy")

print("\n🎯 SSD CARACTERÍSTICAS:")
print("  • Multi-scale feature maps: Detecta objetos de diferentes tamanhos")
print("  • Anchor boxes: Default boxes em múltiplas escalas/aspect ratios")
print("  • Single-shot: Uma única forward pass")
print("  • Speed: ~30-60 FPS (depende da resolução)")

print("\n📊 MÉTRICAS:")
print("  • IoU (Intersection over Union): Overlap entre boxes")
print("  • mAP (mean Average Precision): Métrica principal")
print("  • NMS (Non-Maximum Suppression): Remover duplicatas")

print("\n✅ SSD OBJECT DETECTION COMPLETO!")
