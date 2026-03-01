# GO1411-351Opcional5VideoActionR
# ═══════════════════════════════════════════════════════════════════
# RECONHECIMENTO DE AÇÕES EM VÍDEO COM LSTM + CNN
# Arquitetura: CNN (features espaciais) → LSTM (features temporais)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, LSTM, Dense, 
                                      Dropout, TimeDistributed, Input)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── 1. GERAR DADOS SINTÉTICOS DE VÍDEO ───
print("🎥 Gerando dados sintéticos de vídeos (ações)...")

np.random.seed(42)

# Parâmetros
n_samples = 1000
n_frames = 10  # 10 frames por vídeo
img_height, img_width = 64, 64
n_channels = 3
n_classes = 4

action_labels = ['walking', 'running', 'jumping', 'sitting']

def generate_action_video(action, n_frames=10):
    """Gera vídeo sintético simulando ação"""
    frames = []

    for t in range(n_frames):
        # Frame base
        frame = np.random.rand(img_height, img_width, n_channels) * 0.2

        if action == 'walking':
            # Movimento horizontal lento
            center_x = int(img_width * (0.3 + 0.4 * t / n_frames))
            center_y = img_height // 2
            frame[center_y-10:center_y+10, center_x-5:center_x+5, :] = 0.8

        elif action == 'running':
            # Movimento horizontal rápido
            center_x = int(img_width * (0.2 + 0.6 * t / n_frames))
            center_y = img_height // 2
            frame[center_y-8:center_y+8, center_x-4:center_x+4, :] = 0.9

        elif action == 'jumping':
            # Movimento vertical
            center_x = img_width // 2
            center_y = int(img_height * (0.7 - 0.4 * np.sin(2 * np.pi * t / n_frames)))
            frame[center_y-12:center_y+12, center_x-6:center_x+6, :] = 1.0

        elif action == 'sitting':
            # Estático embaixo
            center_x = img_width // 2
            center_y = int(img_height * 0.8)
            frame[center_y-6:center_y+6, center_x-8:center_x+8, :] = 0.7

        frames.append(frame)

    return np.array(frames)

# Gerar vídeos
videos = []
labels = []

for i in range(n_samples):
    action_idx = i % n_classes
    action = action_labels[action_idx]

    video = generate_action_video(action, n_frames)
    videos.append(video)
    labels.append(action_idx)

videos = np.array(videos)
labels = np.array(labels)

print(f"  Shape dos vídeos: {videos.shape} (samples, frames, H, W, C)")
print(f"  Labels: {labels.shape}")
print(f"  Classes: {action_labels}")

# Visualizar exemplos
fig, axes = plt.subplots(4, 10, figsize=(20, 8))

for action_idx in range(4):
    sample_video = videos[labels == action_idx][0]

    for frame_idx in range(10):
        axes[action_idx, frame_idx].imshow(sample_video[frame_idx])
        axes[action_idx, frame_idx].axis('off')

        if frame_idx == 0:
            axes[action_idx, frame_idx].set_ylabel(action_labels[action_idx], 
                                                     fontsize=12, fontweight='bold')
        if action_idx == 0:
            axes[action_idx, frame_idx].set_title(f'Frame {frame_idx+1}', fontsize=10)

plt.suptitle('Exemplos de Vídeos por Ação', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('video_action_examples.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Exemplos salvos: video_action_examples.png")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados...")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    videos, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"  Treino: {X_train.shape[0]} vídeos")
print(f"  Teste: {X_test.shape[0]} vídeos")

# ─── 3. MODELO 1: CNN-LSTM SIMPLES ───
print("\n🔨 Modelo 1: CNN-LSTM Simples (TimeDistributed)...")

model_cnn_lstm = Sequential([
    # CNN para cada frame (TimeDistributed aplica CNN a cada timestep)
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), 
                    input_shape=(n_frames, img_height, img_width, n_channels)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),

    # LSTM para modelar sequência temporal
    LSTM(64, return_sequences=False),
    Dropout(0.5),

    # Classificação
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
], name='CNN_LSTM')

model_cnn_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model_cnn_lstm.count_params():,}")

# Treinar
history_cnn_lstm = model_cnn_lstm.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    verbose=0
)

acc_cnn_lstm = model_cnn_lstm.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_cnn_lstm:.4f}")

# ─── 4. MODELO 2: CNN-LSTM BIDIRECIONAL ───
print("\n🔨 Modelo 2: CNN-LSTM Bidirecional...")

from tensorflow.keras.layers import Bidirectional

model_bidir = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), 
                    input_shape=(n_frames, img_height, img_width, n_channels)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),

    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),

    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
], name='CNN_BiLSTM')

model_bidir.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model_bidir.count_params():,}")

history_bidir = model_bidir.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    verbose=0
)

acc_bidir = model_bidir.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_bidir:.4f}")

# ─── 5. MODELO 3: TRANSFER LEARNING (MobileNetV2 + LSTM) ───
print("\n🔨 Modelo 3: Transfer Learning (MobileNetV2 + LSTM)...")

# Carregar MobileNetV2 pré-treinado (sem topo)
base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, n_channels),
    pooling='avg'
)
base_model.trainable = False  # Congelar pesos

# Construir modelo
inputs = Input(shape=(n_frames, img_height, img_width, n_channels))
x = TimeDistributed(base_model)(inputs)  # Aplicar MobileNetV2 a cada frame
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(n_classes, activation='softmax')(x)

model_transfer = Model(inputs, outputs, name='MobileNetV2_LSTM')

model_transfer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model_transfer.count_params():,}")

history_transfer = model_transfer.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=8,
    verbose=0
)

acc_transfer = model_transfer.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_transfer:.4f}")

# ─── 6. COMPARAR MODELOS ───
print("\n📊 Comparando modelos...")

import pandas as pd

comparison = pd.DataFrame({
    'Modelo': ['CNN-LSTM', 'CNN-BiLSTM', 'MobileNetV2-LSTM'],
    'Parâmetros': [
        model_cnn_lstm.count_params(),
        model_bidir.count_params(),
        model_transfer.count_params()
    ],
    'Accuracy': [acc_cnn_lstm, acc_bidir, acc_transfer]
})

print("\n" + comparison.to_string(index=False))

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
comparison.plot(x='Modelo', y='Accuracy', kind='bar', ax=axes[0], legend=False, color='skyblue')
axes[0].set_title('Accuracy por Modelo', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticklabels(comparison['Modelo'], rotation=45, ha='right')

# Parâmetros (escala log)
axes[1].bar(comparison['Modelo'], comparison['Parâmetros'], color='coral')
axes[1].set_title('Número de Parâmetros (log scale)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Parâmetros')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticklabels(comparison['Modelo'], rotation=45, ha='right')

# Históricos de treinamento (melhor modelo)
best_idx = comparison['Accuracy'].idxmax()
best_model_name = comparison.iloc[best_idx]['Modelo']

if best_model_name == 'CNN-LSTM':
    history = history_cnn_lstm
elif best_model_name == 'CNN-BiLSTM':
    history = history_bidir
else:
    history = history_transfer

axes[2].plot(history.history['accuracy'], label='Treino', linewidth=2)
axes[2].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
axes[2].set_title(f'Treinamento - {best_model_name}', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Época')
axes[2].set_ylabel('Accuracy')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('video_action_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: video_action_comparison.png")

# ─── 7. MATRIZ DE CONFUSÃO (MELHOR MODELO) ───
print("\n📊 Matriz de confusão do melhor modelo...")

best_model = model_transfer if acc_transfer >= max(acc_cnn_lstm, acc_bidir) else \
             (model_bidir if acc_bidir >= acc_cnn_lstm else model_cnn_lstm)

y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=action_labels, yticklabels=action_labels)
plt.title(f'Matriz de Confusão - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('video_action_confusion.png', dpi=150)
print("  ✓ Matriz salva: video_action_confusion.png")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=action_labels))

# ─── 8. PREDIÇÃO EM NOVO VÍDEO ───
print("\n🎬 Testando em novo vídeo...")

# Gerar novo vídeo de teste
test_action = 'jumping'
test_video = generate_action_video(test_action, n_frames).reshape(1, n_frames, img_height, img_width, n_channels)

# Prever
pred_probs = best_model.predict(test_video, verbose=0)[0]
pred_class = np.argmax(pred_probs)

print(f"\n  Ação real: {test_action}")
print(f"  Ação prevista: {action_labels[pred_class]}")
print(f"\n  Probabilidades:")
for i, action in enumerate(action_labels):
    print(f"    {action}: {pred_probs[i]*100:.2f}%")

# ─── 9. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ RECONHECIMENTO DE AÇÕES EM VÍDEO CONCLUÍDO!")
print("="*70)

print(f"\n📊 Resultados:")
print(f"  • CNN-LSTM: {acc_cnn_lstm:.4f}")
print(f"  • CNN-BiLSTM: {acc_bidir:.4f}")
print(f"  • MobileNetV2-LSTM: {acc_transfer:.4f}")
print(f"  • Melhor: {best_model_name} ({max(acc_cnn_lstm, acc_bidir, acc_transfer):.4f})")

print("\n📁 Arquivos gerados:")
print("  • video_action_examples.png - Exemplos de vídeos")
print("  • video_action_comparison.png - Comparação de modelos")
print("  • video_action_confusion.png - Matriz de confusão")

print("\n💡 Arquitetura:")
print("  • TimeDistributed CNN: Extrai features espaciais de cada frame")
print("  • LSTM: Modela relações temporais entre frames")
print("  • Transfer Learning: Usa MobileNetV2 pré-treinado")

print("\n🔧 Aplicações reais:")
print("  • Vigilância e segurança")
print("  • Análise esportiva")
print("  • Interfaces gestuais")
print("  • Análise médica (movimentos)")
