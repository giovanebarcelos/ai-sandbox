# GO1237-Tensorflow
from tensorflow.keras.applications import VGG16
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# Carregar VGG sem camadas FC (feature extractor)
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))

# Congelar camadas pre-treinadas
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas específicas do domínio
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)

# ─── VISUALIZAÇÃO: HIERARQUIA DE FEATURES POR BLOCO DO VGG ───
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Diagrama de hierarquia de features
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Hierarquia de Features Aprendidas pelo VGG', fontsize=12, fontweight='bold')

blocos = [
    ('Bloco 1\n64 filtros',   '#4e79a7', 'Bordas e linhas\n(vertical, horizontal, diagonal)'),
    ('Bloco 2\n128 filtros',  '#f28e2b', 'Texturas simples\n(listras, grades, manchas)'),
    ('Bloco 3\n256 filtros',  '#e15759', 'Padrões complexos\n(olhos, rodas, janelas)'),
    ('Bloco 4\n512 filtros',  '#76b7b2', 'Partes de objetos\n(faces, portas, patas)'),
    ('Bloco 5\n512 filtros',  '#59a14f', 'Objetos completos\n(carros, cães, pessoas)'),
]

for i, (nome, cor, descricao) in enumerate(blocos):
    y = 6 - i * 1.1
    # Caixa do bloco
    rect = mpatches.FancyBboxPatch((0.3, y - 0.35), 2.5, 0.65,
                                    boxstyle='round,pad=0.05', linewidth=1.5,
                                    edgecolor='black', facecolor=cor, alpha=0.85)
    ax.add_patch(rect)
    ax.text(1.55, y, nome, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # Descrição
    ax.text(3.2, y, descricao, ha='left', va='center', fontsize=8.5, color='black')
    if i < len(blocos) - 1:
        ax.annotate('', xy=(1.55, 6 - (i+1)*1.1 + 0.32), xytext=(1.55, y - 0.35),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3))

# Camadas customizadas
for j, (nome, cor) in enumerate([('GAP + Dense(256)', '#edc948'), ('Dropout(0.5)', '#ff9da7'), (f'Dense(num_classes)', '#b07aa1')]):
    y = 0.9 - j * 0.0  # all at bottom
    pass  # omitted for diagram clarity
rect_new = mpatches.FancyBboxPatch((0.3, 0.1), 2.5, 0.55,
                                    boxstyle='round,pad=0.05', linewidth=2,
                                    edgecolor='gold', facecolor='#edc948', alpha=0.9)
ax.add_patch(rect_new)
ax.text(1.55, 0.38, 'Classificador Customizado\nGAP → Dense → Softmax', ha='center', va='center',
        fontsize=8, fontweight='bold')
ax.text(3.2, 0.38, '← Únicas camadas treináveis!\n   (transfer learning)', ha='left', va='center', fontsize=8.5)

# Camadas congeladas vs treináveis
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 6)
axes[1].axis('off')
axes[1].set_title('Transfer Learning: Congelado vs Treinável', fontsize=12, fontweight='bold')

stages = [
    ('Feature Extraction', 'Bloco 1–5 congelados\n(pesos ImageNet preservados)', '#aaaaaa', '❌ Congelado'),
    ('Fine-Tuning leve',   'Últimos 2 blocos descongelados\n(lr=1e-5, ajuste sutil)', '#76b7b2', '⚠️ Parcial'),
    ('Fine-Tuning total',  'Todos os pesos treináveis\n(lr muito baixo, risco de overfit)', '#e15759', '✅ Treinável'),
]
acc_expected = [0.72, 0.82, 0.88]
for i, (stage, desc, cor, status) in enumerate(stages):
    y = 5 - i * 1.8
    rect = mpatches.FancyBboxPatch((0.2, y - 0.5), 4.5, 0.9, boxstyle='round,pad=0.1',
                                    linewidth=1.5, edgecolor='black', facecolor=cor, alpha=0.6)
    axes[1].add_patch(rect)
    axes[1].text(2.45, y, f'{stage}\n{desc}', ha='center', va='center', fontsize=9)
    axes[1].text(5, y + 0.1, status, ha='left', va='center', fontsize=10, fontweight='bold')
    axes[1].text(5, y - 0.25, f'Acc esperada: ~{acc_expected[i]:.0%}', ha='left', va='center', fontsize=8.5, color='navy')

plt.suptitle('VGG16 como Feature Extractor / Transfer Learning', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
