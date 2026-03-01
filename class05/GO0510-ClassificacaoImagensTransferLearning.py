# GO0510-ClassificaçãoImagensTransferLearning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow e Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ═══════════════════════════════════════════════════════════════════
# 1. CONFIGURAÇÃO E DATASET
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("CLASSIFICAÇÃO DE IMAGENS - TRANSFER LEARNING")
print("="*70)

# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3  # Exemplo: Gato, Cachorro, Pássaro

print(f"\n⚙️ Configurações:")
print(f"   • Tamanho da imagem: {IMG_SIZE}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Épocas: {EPOCHS}")
print(f"   • Número de classes: {NUM_CLASSES}")

# ═══════════════════════════════════════════════════════════════════
# 2. CRIAR DATASET SINTÉTICO (SIMULAÇÃO)
# ═══════════════════════════════════════════════════════════════════

print("\n📊 Criando dataset sintético para demonstração...")

# Simular imagens (em produção, usar tf.keras.utils.image_dataset_from_directory)
def criar_imagens_sinteticas(n_samples, img_size, num_classes):
    """Criar imagens sintéticas para demonstração"""
    images = []
    labels = []

    for _ in range(n_samples):
        # Criar imagem aleatória
        img = np.random.rand(*img_size, 3).astype(np.float32)

        # Label aleatório
        label = np.random.randint(0, num_classes)

        # Adicionar padrões diferentes por classe (simplificado)
        if label == 0:  # Classe 0: tons avermelhados
            img[:, :, 0] += 0.3
        elif label == 1:  # Classe 1: tons esverdeados
            img[:, :, 1] += 0.3
        else:  # Classe 2: tons azulados
            img[:, :, 2] += 0.3

        img = np.clip(img, 0, 1)

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Criar dados de treino e teste
X_train, y_train = criar_imagens_sinteticas(300, IMG_SIZE, NUM_CLASSES)
X_val, y_val = criar_imagens_sinteticas(60, IMG_SIZE, NUM_CLASSES)
X_test, y_test = criar_imagens_sinteticas(60, IMG_SIZE, NUM_CLASSES)

print(f"\n✅ Dataset criado:")
print(f"   • Treino: {len(X_train)} imagens")
print(f"   • Validação: {len(X_val)} imagens")
print(f"   • Teste: {len(X_test)} imagens")

# ═══════════════════════════════════════════════════════════════════
# 3. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("DATA AUGMENTATION")
print("="*70)

# Criar gerador com augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

print("\n✅ Transformações configuradas:")
print("   • Flip horizontal aleatório")
print("   • Rotação aleatória (±10%)")
print("   • Zoom aleatório (±10%)")
print("   • Contraste aleatório (±10%)")

# Visualizar exemplos de augmentation
print("\n🎨 Gerando exemplos de augmentation...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Exemplos de Data Augmentation', fontsize=16, fontweight='bold')

# Pegar uma imagem
sample_img = X_train[0:1]

for i in range(10):
    ax = axes[i // 5, i % 5]
    augmented = data_augmentation(sample_img, training=True)
    ax.imshow(augmented[0])
    ax.axis('off')
    ax.set_title(f'Aug {i+1}')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 4. MODELO BASE - MOBILENETV2 PRÉ-TREINADA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TRANSFER LEARNING - MOBILENETV2")
print("="*70)

print("\n📥 Carregando MobileNetV2 pré-treinada (ImageNet)...")

# Carregar modelo base sem as camadas de classificação
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,  # Remover camadas de classificação
    weights='imagenet'   # Pesos pré-treinados do ImageNet
)

# Congelar camadas base (não treinar inicialmente)
base_model.trainable = False

print(f"✅ MobileNetV2 carregada:")
print(f"   • Total de camadas: {len(base_model.layers)}")
print(f"   • Parâmetros: {base_model.count_params():,}")
print(f"   • Treinável: {base_model.trainable}")

# ═══════════════════════════════════════════════════════════════════
# 5. CONSTRUIR MODELO COMPLETO
# ═══════════════════════════════════════════════════════════════════

print("\n🏗️ Construindo modelo completo...")

# Criar modelo sequencial
model = keras.Sequential([
    # Data augmentation
    data_augmentation,

    # Pré-processamento (normalização para MobileNet)
    layers.Rescaling(1./127.5, offset=-1),

    # Base model (congelada)
    base_model,

    # Camadas de classificação customizadas
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilar
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Modelo construído:")
model.summary()

# ═══════════════════════════════════════════════════════════════════
# 6. CALLBACKS
# ═══════════════════════════════════════════════════════════════════

print("\n⚙️ Configurando callbacks...")

callbacks = [
    # Early stopping (parar se não melhorar)
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduzir learning rate se estagnado
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("✅ Callbacks configurados:")
print("   • Early Stopping (patience=5)")
print("   • Reduce LR on Plateau (factor=0.5)")

# ═══════════════════════════════════════════════════════════════════
# 7. FASE 1: TREINAR APENAS CAMADAS SUPERIORES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 1: TREINAMENTO (CAMADAS SUPERIORES)")
print("="*70)

print("\n🚀 Iniciando treinamento da fase 1...")

history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Fase 1 concluída!")

# ═══════════════════════════════════════════════════════════════════
# 8. FASE 2: FINE-TUNING (DESCONGELAR CAMADAS BASE)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 2: FINE-TUNING")
print("="*70)

print("\n🔓 Descongelando camadas da base model...")

# Descongelar últimas 30 camadas
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"✅ Camadas descongeladas:")
trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"   • Treináveis: {trainable_layers}/{len(base_model.layers)}")

# Recompilar com learning rate menor
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # LR menor!
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Iniciando fine-tuning...")

history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Fine-tuning concluído!")

# ═══════════════════════════════════════════════════════════════════
# 9. AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("AVALIAÇÃO DO MODELO")
print("="*70)

# Avaliar no conjunto de teste
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n📊 Resultados no conjunto de teste:")
print(f"   • Loss: {test_loss:.4f}")
print(f"   • Acurácia: {test_acc:.4f}")

# Predições
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Relatório de classificação
class_names = [f'Classe {i}' for i in range(NUM_CLASSES)]
print("\n📈 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ═══════════════════════════════════════════════════════════════════
# 10. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n🎨 Gerando visualizações...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 10.1 Histórico de Treinamento - Acurácia
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history_phase1.history['accuracy'], label='Fase 1 - Treino', linewidth=2)
ax1.plot(history_phase1.history['val_accuracy'], label='Fase 1 - Val', linewidth=2)

phase2_offset = len(history_phase1.history['accuracy'])
phase2_epochs = range(phase2_offset, phase2_offset + len(history_phase2.history['accuracy']))
ax1.plot(phase2_epochs, history_phase2.history['accuracy'], 
        label='Fase 2 - Treino', linewidth=2, linestyle='--')
ax1.plot(phase2_epochs, history_phase2.history['val_accuracy'], 
        label='Fase 2 - Val', linewidth=2, linestyle='--')

ax1.axvline(x=phase2_offset, color='r', linestyle=':', label='Início Fine-Tuning')
ax1.set_xlabel('Época')
ax1.set_ylabel('Acurácia')
ax1.set_title('Evolução da Acurácia')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 10.2 Histórico de Treinamento - Loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history_phase1.history['loss'], label='Fase 1 - Treino', linewidth=2)
ax2.plot(history_phase1.history['val_loss'], label='Fase 1 - Val', linewidth=2)
ax2.plot(phase2_epochs, history_phase2.history['loss'], 
        label='Fase 2 - Treino', linewidth=2, linestyle='--')
ax2.plot(phase2_epochs, history_phase2.history['val_loss'], 
        label='Fase 2 - Val', linewidth=2, linestyle='--')
ax2.axvline(x=phase2_offset, color='r', linestyle=':', label='Início Fine-Tuning')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.set_title('Evolução do Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 10.3 Matriz de Confusão
ax3 = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=class_names,
            yticklabels=class_names)
ax3.set_title('Matriz de Confusão')
ax3.set_ylabel('Real')
ax3.set_xlabel('Predito')

# 10.4 Distribuição de Confiança
ax4 = fig.add_subplot(gs[1, :])
confidences = np.max(y_pred_proba, axis=1)
correct = (y_pred == y_test)

ax4.hist(confidences[correct], bins=20, alpha=0.7, label='Correto', color='green')
ax4.hist(confidences[~correct], bins=20, alpha=0.7, label='Incorreto', color='red')
ax4.set_xlabel('Confiança da Predição')
ax4.set_ylabel('Frequência')
ax4.set_title('Distribuição de Confiança das Predições')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 10.5 Exemplos de Predições
for i in range(6):
    ax = fig.add_subplot(gs[2, i//2 if i < 3 else (i-3)//2])

    idx = np.random.randint(0, len(X_test))
    img = X_test[idx]
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = y_pred_proba[idx][pred_label]

    ax.imshow(img)
    ax.axis('off')

    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'Real: {class_names[true_label]}\n'
                f'Pred: {class_names[pred_label]} ({confidence:.2%})',
                color=color, fontsize=9)

fig.suptitle('Análise Completa - Transfer Learning com MobileNetV2', 
            fontsize=16, fontweight='bold')

plt.show()

# ═══════════════════════════════════════════════════════════════════
# 11. TESTAR COM NOVA IMAGEM
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTE COM NOVA IMAGEM")
print("="*70)

def prever_imagem(img, model, class_names):
    """Prever classe de uma nova imagem"""
    # Adicionar dimensão de batch
    img_batch = np.expand_dims(img, axis=0)

    # Predição
    pred_proba = model.predict(img_batch, verbose=0)[0]
    pred_class = np.argmax(pred_proba)

    return pred_class, pred_proba

# Testar com imagem aleatória
nova_img = X_test[0]
pred_class, pred_proba = prever_imagem(nova_img, model, class_names)

print(f"\n🔍 Predição para nova imagem:")
print(f"   • Classe predita: {class_names[pred_class]}")
print(f"   • Confiança: {pred_proba[pred_class]:.2%}")
print(f"\n   📊 Probabilidades por classe:")
for i, prob in enumerate(pred_proba):
    print(f"      {class_names[i]}: {prob:.2%}")

# ═══════════════════════════════════════════════════════════════════
# 12. CONCLUSÕES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONCLUSÕES E RECOMENDAÇÕES")
print("="*70)

print(f"\n🏆 RESULTADOS FINAIS:")
print(f"   • Acurácia em teste: {test_acc:.2%}")
print(f"   • Parâmetros totais: {model.count_params():,}")
print(f"   • Parâmetros treináveis: {sum([np.prod(p.shape) for p in model.trainable_weights]):,}")

print("\n💡 VANTAGENS DO TRANSFER LEARNING:")
print("   1. Treina com poucos dados (datasets pequenos)")
print("   2. Converge rapidamente (menos épocas)")
print("   3. Melhor generalização (features pré-aprendidas)")
print("   4. Menor custo computacional")
print("   5. Estado da arte em muitos problemas")

print("\n📊 ESTRATÉGIA DE TREINAMENTO:")
print("   Fase 1: Treinar apenas camadas superiores")
print("           → Adaptar features à nova tarefa")
print("   Fase 2: Fine-tuning de camadas profundas")
print("           → Refinar features específicas")

print("\n🔧 MELHORIAS POSSÍVEIS:")
print("   • Usar modelos maiores (ResNet, EfficientNet)")
print("   • Mais data augmentation (MixUp, CutMix)")
print("   • Ensemble de múltiplos modelos")
print("   • Test-Time Augmentation (TTA)")
print("   • Otimizar hiperparâmetros (learning rate, dropout)")

print("\n🎯 APLICAÇÕES PRÁTICAS:")
print("   • Classificação de produtos (e-commerce)")
print("   • Diagnóstico médico (raio-X, tomografia)")
print("   • Controle de qualidade (defeitos)")
print("   • Reconhecimento facial")
print("   • Análise de imagens satélite")

print("\n📚 PRÓXIMOS PASSOS:")
print("   • Estudar outras arquiteturas (Vision Transformers)")
print("   • Implementar Object Detection (YOLO, Faster R-CNN)")
print("   • Explorar Segmentação Semântica (U-Net)")
print("   • Deploy em produção (TensorFlow Lite, ONNX)")

print("\n" + "="*70)
print("FIM DO EXERCÍCIO 3")
print("="*70)
