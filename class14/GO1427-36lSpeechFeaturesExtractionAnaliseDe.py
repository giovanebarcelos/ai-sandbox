# GO1427-36lSpeechFeaturesExtractionAnáliseDe
# ══════════════════════════════════════════════════════════════════
# SPEECH FEATURES EXTRACTION COM LSTM
# Extrair features de áudio e classificar (simulado)
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt

print("🎤 SPEECH FEATURES EXTRACTION COM LSTM")
print("=" * 70)

# ─── 1. GERAR SINAIS DE ÁUDIO SINTÉTICOS ───
print("\n🔊 Gerando sinais de áudio sintéticos...")

def generate_speech_features(num_samples=500, seq_length=50, num_features=13):
    """
    Simula MFCCs (Mel-Frequency Cepstral Coefficients)
    """
    X = []
    y = []

    for _ in range(num_samples):
        # Classe 0: Padrão baixa frequência
        # Classe 1: Padrão média frequência  
        # Classe 2: Padrão alta frequência

        cls = np.random.randint(0, 3)

        if cls == 0:  # Baixa frequência
            signal = np.random.randn(seq_length, num_features) * 0.5
            signal[:, :5] += np.sin(np.linspace(0, 4*np.pi, seq_length))[:, None] * 2

        elif cls == 1:  # Média frequência
            signal = np.random.randn(seq_length, num_features) * 0.5
            signal[:, 5:10] += np.sin(np.linspace(0, 8*np.pi, seq_length))[:, None] * 2

        else:  # Alta frequência
            signal = np.random.randn(seq_length, num_features) * 0.5
            signal[:, 10:] += np.sin(np.linspace(0, 16*np.pi, seq_length))[:, None] * 2

        X.append(signal)
        y.append(cls)

    return np.array(X), np.array(y)

X_data, y_data = generate_speech_features(num_samples=1000)

print(f"  Data shape: {X_data.shape}")  # (samples, timesteps, features)
print(f"  Labels shape: {y_data.shape}")
print(f"  Classes: 0=Low freq, 1=Mid freq, 2=High freq")

# ─── 2. SPLIT DATA ───
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 3. CONSTRUIR MODELO ───
print("\n🏗️ Construindo modelo LSTM para speech...")

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(50, 13)),
    Dropout(0.5),

    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(3, activation='softmax')
], name='Speech_LSTM')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando modelo...")

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=0
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"  Test Accuracy: {test_acc:.4f}")

# ─── 5. CONFUSION MATRIX ───
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)

class_names = ['Low Freq', 'Mid Freq', 'High Freq']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Speech Classification - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('speech_confusion_matrix.png', dpi=150)
print("\n✅ Confusion matrix salva: speech_confusion_matrix.png")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ─── 6. VISUALIZAR FEATURES ───
print("\n📊 Visualizando speech features...")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for cls in range(3):
    # Pegar 3 exemplos de cada classe
    indices = np.where(y_test == cls)[0][:3]

    for i, idx in enumerate(indices):
        features = X_test[idx]
        pred = y_pred[idx]

        # Heatmap das features
        axes[cls, i].imshow(features.T, aspect='auto', cmap='viridis', origin='lower')
        axes[cls, i].set_xlabel('Time', fontsize=10)

        if i == 0:
            axes[cls, i].set_ylabel(f'{class_names[cls]}\nFeature', fontsize=10)

        color = 'green' if pred == cls else 'red'
        axes[cls, i].set_title(f'Pred: {class_names[pred]}', 
                               fontsize=10, color=color, fontweight='bold')

plt.suptitle('Speech Features Visualization (MFCCs)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('speech_features.png', dpi=150)
print("✅ Features salvas: speech_features.png")

# ─── 7. ANÁLISE TEMPORAL ───
print("\n📈 Analisando padrões temporais...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for cls, ax in enumerate(axes):
    # Média das features ao longo do tempo para cada classe
    cls_indices = np.where(y_train == cls)[0]
    cls_samples = X_train[cls_indices]

    mean_features = cls_samples.mean(axis=0)  # (timesteps, features)

    # Plot de algumas features
    for feat_idx in [0, 5, 10]:
        ax.plot(mean_features[:, feat_idx], label=f'Feature {feat_idx}', linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'{class_names[cls]} - Temporal Pattern', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Temporal Patterns by Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('speech_temporal_patterns.png', dpi=150)
print("✅ Padrões temporais salvos: speech_temporal_patterns.png")

print("\n💡 SPEECH PROCESSING:")
print("  • MFCCs: Mel-Frequency Cepstral Coefficients (13-40 coefs)")
print("  • Spectrogram: Representação tempo-frequência")
print("  • Mel-Spectrogram: Escala perceptual (mel)")
print("  • Pitch: Frequência fundamental (F0)")

print("\n🎯 FEATURES COMUNS:")
print("  • MFCCs: 13 coeficientes + deltas + delta-deltas")
print("  • Energy: Energia do sinal")
print("  • Zero Crossing Rate: Taxa de cruzamento por zero")
print("  • Spectral Centroid: Centro de massa do espectro")

print("\n📊 APLICAÇÕES:")
print("  • ASR: Automatic Speech Recognition")
print("  • Speaker Recognition: Identificar pessoa")
print("  • Emotion Recognition: Detectar emoção")
print("  • Language Identification: Identificar idioma")

print("\n🏆 DATASETS:")
print("  • LibriSpeech: 1000h de audiobooks")
print("  • Common Voice: Mozilla, multilíngue")
print("  • TIMIT: Benchmark clássico")
print("  • VoxCeleb: Speaker recognition")

print("\n✅ SPEECH FEATURES COMPLETO!")
