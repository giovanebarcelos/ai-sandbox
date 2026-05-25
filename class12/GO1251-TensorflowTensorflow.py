# GO1251-TensorflowTensorflow
# Transfer Learning com VGG16 em 2 FASES:
# FASE 1 — Feature Extraction: congela VGG16, treina apenas o topo (rápido, 10 épocas)
# FASE 2 — Fine-tuning: descongela as últimas 4 camadas, refina com LR muito baixo
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# ─── FASE 1: Feature Extraction ───
# include_top=False: remove as 3 camadas FC do VGG16 original (ImageNet, 1000 classes)
# weights='imagenet': carrega pesos treinados em 1.2M imagens — bagagem de conhecimento
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar: layer.trainable=False → pesos não serão alterados no treino
# Razão: queremos usar as features aprendidas no ImageNet sem destruí-las
for layer in base_model.layers:
    layer.trainable = False

# Adicionar cabeçalho de classificação para NOSSO problema (10 classes)
# GlobalAveragePooling2D: reduz cada feature map para 1 valor (sem params extras)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)  # camada de adaptação
x = layers.Dropout(0.5)(x)                   # regularização para dados pequenos
predictions = layers.Dense(10, activation='softmax')(x)  # saída para 10 classes

# Model funcional: permite conectar base_model ao nosso topo
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',         # LR padrão (0.001) — ok pois só o topo treina
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar apenas as novas camadas (rápido pois só ~131k params são treináveis)
model.fit(x_train, y_train, epochs=10)

# ─── FASE 2: Fine-tuning (opcional) ───
# Depois que o topo convergiu, podemos refinar as últimas camadas do VGG16
# As últimas camadas têm features mais específicas (as primeiras têm features genéricas)
for layer in base_model.layers[-4:]:
    layer.trainable = True  # descongela os últimos 4 layers (bloco 5 do VGG16)

# LR muito baixo (1e-5 vs padrão 1e-3): evita destruir os pesos pré-treinados
# Fine-tuning com LR alto = catastrofic forgetting
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 100× menor que o padrão!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)  # poucas épocas: ajuste fino, não retreinar

# ─── VISUALIZAÇÃO: CONCEITO DE TRANSFER LEARNING ───
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Diagrama: Feature Extraction vs Fine-Tuning
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('VGG16: Feature Extraction vs Fine-Tuning', fontsize=12, fontweight='bold')

blocos_vgg = [
    ('Bloco 1\n64 filtros', '#4e79a7'), ('Bloco 2\n128 filtros', '#4e79a7'),
    ('Bloco 3\n256 filtros', '#4e79a7'), ('Bloco 4\n512 filtros', '#4e79a7'),
    ('Bloco 5\n512 filtros', '#76b7b2'),
]
y_positions = [10.5, 9.0, 7.5, 6.0, 4.5]
for (nome, cor), y in zip(blocos_vgg, y_positions):
    rect = mpatches.FancyBboxPatch((0.5, y - 0.5), 3.5, 0.9,
                                    boxstyle='round,pad=0.1', linewidth=1.5,
                                    edgecolor='black', facecolor=cor, alpha=0.8)
    ax.add_patch(rect)
    ax.text(2.25, y, nome, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Novas camadas
novas = [('GAP + Dense(256)', '#59a14f'), ('Dropout(0.5)', '#f28e2b'), ('Dense(10)', '#e15759')]
for i, (nome, cor) in enumerate(novas):
    y = 3.0 - i * 1.2
    rect = mpatches.FancyBboxPatch((0.5, y - 0.5), 3.5, 0.9,
                                    boxstyle='round,pad=0.1', linewidth=1.5,
                                    edgecolor='black', facecolor=cor, alpha=0.8)
    ax.add_patch(rect)
    ax.text(2.25, y, nome, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Legenda congelado/treinável
ax.add_patch(mpatches.Patch(color='#4e79a7', label='Congelado (ImageNet)'))
ax.add_patch(mpatches.Patch(color='#76b7b2', label='Fine-tuning (últimas camadas)'))
ax.add_patch(mpatches.Patch(color='#59a14f', label='Novas camadas (treináveis)'))
ax.legend(loc='lower right', fontsize=8)

# Anotações
ax.text(5, 10.5, '❄️ CONGELADO\n(extração de features)', ha='left', va='center',
        fontsize=9, color='#4e79a7', fontweight='bold')
ax.text(5, 4.5, '🔓 FINE-TUNING\n(ajuste fino)', ha='left', va='center',
        fontsize=9, color='#76b7b2', fontweight='bold')
ax.text(5, 2.5, '✅ TREINÁVEL\n(classificador customizado)', ha='left', va='center',
        fontsize=9, color='#59a14f', fontweight='bold')

# Comparação: treinar do zero vs transfer learning
ax2 = axes[1]
fases = ['Feature\nExtraction\n(10 épocas)', 'Fine-Tuning\n(5 épocas)']
accs_transfer = [0.72, 0.83]
accs_scratch = [0.45, 0.65]
x = np.arange(len(fases))
bars1 = ax2.bar(x - 0.2, accs_transfer, 0.35, label='Transfer Learning (VGG16)',
                color='#4e79a7', edgecolor='black')
bars2 = ax2.bar(x + 0.2, accs_scratch, 0.35, label='Treinar do zero (CNN simples)',
                color='#e15759', edgecolor='black', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(fases, fontsize=10)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('Acurácia')
ax2.set_title('Transfer Learning vs Treinar do Zero', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(list(bars1) + list(bars2), accs_transfer + accs_scratch):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('VGG16 Transfer Learning', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
