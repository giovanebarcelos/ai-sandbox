# GO1221-35sMaskRcnnInstanceSegmentation
# ══════════════════════════════════════════════════════════════════
# MASK R-CNN - INSTANCE SEGMENTATION
# Detectar objetos + segmentar cada instância
# ══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print("🎭 MASK R-CNN - INSTANCE SEGMENTATION")
print("=" * 70)

# ─── 1. GERAR DADOS SINTÉTICOS ───
print("\n📦 Gerando imagens com múltiplas instâncias...")

def generate_instance_data(num_samples=100, img_size=128):
    images = []
    boxes = []  # Bounding boxes
    masks = []  # Máscaras de segmentação

    for _ in range(num_samples):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        instance_masks = []
        instance_boxes = []

        # Adicionar 2-4 objetos
        num_objects = np.random.randint(2, 5)

        for obj_id in range(num_objects):
            # Círculo ou retângulo
            obj_type = np.random.choice(['circle', 'rectangle'])

            mask = np.zeros((img_size, img_size), dtype=np.uint8)

            if obj_type == 'circle':
                center = (np.random.randint(20, img_size-20), 
                         np.random.randint(20, img_size-20))
                radius = np.random.randint(10, 20)
                color = tuple(np.random.randint(0, 255, 3).tolist())

                cv2.circle(img, center, radius, color, -1)
                cv2.circle(mask, center, radius, 255, -1)

                # Bounding box
                x1, y1 = max(0, center[0]-radius), max(0, center[1]-radius)
                x2, y2 = min(img_size, center[0]+radius), min(img_size, center[1]+radius)

            else:  # rectangle
                x = np.random.randint(10, img_size-40)
                y = np.random.randint(10, img_size-40)
                w = np.random.randint(20, 30)
                h = np.random.randint(20, 30)
                color = tuple(np.random.randint(0, 255, 3).tolist())

                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

                x1, y1, x2, y2 = x, y, x+w, y+h

            instance_boxes.append([x1, y1, x2, y2])
            instance_masks.append(mask)

        images.append(img.astype('float32') / 255.0)
        boxes.append(instance_boxes)
        masks.append(instance_masks)

    return np.array(images), boxes, masks

X_data, boxes_data, masks_data = generate_instance_data(200)

print(f"  Images: {X_data.shape}")
print(f"  Objetos por imagem: 2-4")

# ─── 2. MODELO SIMPLIFICADO (CONCEITUAL) ───
print("\n🏗️ Mask R-CNN (versão simplificada)...")

# Nota: Mask R-CNN real é muito complexo
# Esta é uma demonstração conceitual dos componentes

print("\n  COMPONENTES DO MASK R-CNN:")
print("  1. Backbone: ResNet/FPN (feature extraction)")
print("  2. RPN: Region Proposal Network (propostas de objetos)")
print("  3. ROI Align: Extrair features das regiões")
print("  4. Box Head: Classificação + bounding box")
print("  5. Mask Head: Segmentação de cada instância")

# Modelo de detecção simples (conceito)
def simple_detection_model(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    # Feature extraction (backbone simplificado)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Mask branch (segmentação)
    mask = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    mask = UpSampling2D((2, 2))(mask)
    mask = Conv2D(32, (3, 3), activation='relu', padding='same')(mask)
    mask = UpSampling2D((2, 2))(mask)
    mask_output = Conv2D(1, (1, 1), activation='sigmoid')(mask)

    model = Model(inputs=inputs, outputs=mask_output)
    return model

model = simple_detection_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(f"\n  Parâmetros: {model.count_params():,}")

# ─── 3. VISUALIZAR CONCEITO ───
print("\n📊 Visualizando instance segmentation...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # Imagem original
    axes[0, i].imshow(X_data[i])
    axes[0, i].set_title('Original', fontsize=10)
    axes[0, i].axis('off')

    # Máscaras + bounding boxes
    axes[1, i].imshow(X_data[i])

    # Combinar todas as máscaras
    combined_mask = np.zeros((128, 128))
    for mask in masks_data[i]:
        combined_mask = np.maximum(combined_mask, mask)

    # Overlay mask
    axes[1, i].imshow(combined_mask, alpha=0.4, cmap='jet')

    # Desenhar bounding boxes
    for box in boxes_data[i]:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                         fill=False, edgecolor='red', linewidth=2)
        axes[1, i].add_patch(rect)

    axes[1, i].set_title(f'Instances: {len(masks_data[i])}', fontsize=10, color='green')
    axes[1, i].axis('off')

plt.suptitle('Mask R-CNN: Instance Segmentation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mask_rcnn_instances.png', dpi=150)
print("✅ Instances salvas: mask_rcnn_instances.png")

print("\n💡 MASK R-CNN:")
print("  • Extension do Faster R-CNN")
print("  • Adiciona branch de segmentação")
print("  • Detecta + classifica + segmenta cada instância")
print("  • ROI Align: Preserva alinhamento espacial")

print("\n🎯 DIFERENÇAS:")
print("  • Semantic Segmentation: Pixel-wise classes")
print("  • Instance Segmentation: Separar cada objeto")
print("  • Panoptic Segmentation: Semantic + Instance")

print("\n🏆 APLICAÇÕES:")
print("  • Autonomous Driving: Detectar pedestres, carros")
print("  • Medical: Segmentar células, tumores")
print("  • Robotics: Manipulação de objetos")
print("  • Agriculture: Contar frutas individuais")

print("\n📊 MÉTRICAS:")
print("  • mAP: Mean Average Precision")
print("  • IoU: Intersection over Union")
print("  • Mask AP: AP específico para máscaras")

print("\n🔥 DATASETS:")
print("  • COCO: 330k images, 80 classes, instance masks")
print("  • Cityscapes: Urban scenes")
print("  • ADE20K: Scene parsing")

print("\n✅ MASK R-CNN COMPLETO!")
