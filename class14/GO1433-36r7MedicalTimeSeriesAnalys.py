# GO1433-36r7MedicalTimeSeriesAnalys
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DE SÉRIES TEMPORAIS MÉDICAS - ECG COM LSTM
# Aplicação: Detecção de arritmia cardíaca
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns

# ─── 1. GERAR DADOS DE ECG SINTÉTICOS ───
print("💓 Gerando dados sintéticos de ECG...")

np.random.seed(42)

n_samples = 2000
n_timesteps = 200
sampling_rate = 360  # Hz

def generate_normal_ecg(n_points=200):
    """Gera ECG normal"""
    t = np.linspace(0, 1, n_points)

    # P wave
    p_wave = 0.2 * np.exp(-((t - 0.15) ** 2) / 0.001)

    # QRS complex
    q_wave = -0.1 * np.exp(-((t - 0.35) ** 2) / 0.0001)
    r_wave = 1.0 * np.exp(-((t - 0.4) ** 2) / 0.0001)
    s_wave = -0.2 * np.exp(-((t - 0.45) ** 2) / 0.0001)

    # T wave
    t_wave = 0.3 * np.exp(-((t - 0.7) ** 2) / 0.003)

    # Combinar
    ecg = p_wave + q_wave + r_wave + s_wave + t_wave

    # Adicionar ruído fisiológico
    ecg += np.random.normal(0, 0.02, n_points)

    return ecg

def generate_abnormal_ecg(n_points=200, anomaly_type='tachycardia'):
    """Gera ECG com anomalia"""
    ecg = generate_normal_ecg(n_points)

    if anomaly_type == 'tachycardia':
        # Frequência cardíaca elevada (batimentos mais próximos)
        t = np.linspace(0, 2, n_points)  # Dois ciclos
        ecg = generate_normal_ecg(n_points // 2)
        ecg = np.tile(ecg, 2)[:n_points]

    elif anomaly_type == 'bradycardia':
        # Frequência cardíaca reduzida (batimentos mais espaçados)
        ecg_stretched = generate_normal_ecg(n_points // 2)
        ecg = np.interp(np.linspace(0, len(ecg_stretched), n_points), 
                        np.arange(len(ecg_stretched)), ecg_stretched)

    elif anomaly_type == 'premature_beat':
        # Batimento prematuro
        ecg_normal = generate_normal_ecg(n_points)
        # Inserir batimento extra no meio
        insert_pos = n_points // 2
        extra_beat = generate_normal_ecg(40)
        ecg = np.concatenate([ecg_normal[:insert_pos], extra_beat, 
                              ecg_normal[insert_pos+40:]])[:n_points]

    elif anomaly_type == 'atrial_fibrillation':
        # Fibrilação atrial (ritmo irregular)
        ecg += np.random.normal(0, 0.1, n_points)
        # Remover P wave
        t = np.linspace(0, 1, n_points)
        p_wave = 0.2 * np.exp(-((t - 0.15) ** 2) / 0.001)
        ecg -= p_wave

    return ecg

# Gerar dataset balanceado
n_per_class = n_samples // 5

ecg_signals = []
labels = []
label_names = []

# Classe 0: Normal
for i in range(n_per_class):
    ecg = generate_normal_ecg(n_timesteps)
    ecg_signals.append(ecg)
    labels.append(0)
    label_names.append('Normal')

# Classe 1: Tachycardia
for i in range(n_per_class):
    ecg = generate_abnormal_ecg(n_timesteps, 'tachycardia')
    ecg_signals.append(ecg)
    labels.append(1)
    label_names.append('Tachycardia')

# Classe 2: Bradycardia
for i in range(n_per_class):
    ecg = generate_abnormal_ecg(n_timesteps, 'bradycardia')
    ecg_signals.append(ecg)
    labels.append(2)
    label_names.append('Bradycardia')

# Classe 3: Premature Beat
for i in range(n_per_class):
    ecg = generate_abnormal_ecg(n_timesteps, 'premature_beat')
    ecg_signals.append(ecg)
    labels.append(3)
    label_names.append('Premature Beat')

# Classe 4: Atrial Fibrillation
for i in range(n_per_class):
    ecg = generate_abnormal_ecg(n_timesteps, 'atrial_fibrillation')
    ecg_signals.append(ecg)
    labels.append(4)
    label_names.append('Atrial Fibrillation')

ecg_signals = np.array(ecg_signals)
labels = np.array(labels)

print(f"  Shape dos sinais: {ecg_signals.shape} (samples, timesteps)")
print(f"  Labels: {labels.shape}")
print(f"  Classes: Normal, Tachycardia, Bradycardia, Premature Beat, Atrial Fibrillation")

# Visualizar exemplos
class_labels = ['Normal', 'Tachycardia', 'Bradycardia', 'Premature Beat', 'Atrial Fibrillation']
colors = ['green', 'red', 'orange', 'purple', 'brown']

fig, axes = plt.subplots(5, 1, figsize=(15, 12))

for i, (class_label, color) in enumerate(zip(class_labels, colors)):
    sample_idx = np.where(labels == i)[0][0]
    axes[i].plot(ecg_signals[sample_idx], color=color, linewidth=2)
    axes[i].set_title(f'Classe {i}: {class_label}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Amplitude (mV)')
    axes[i].grid(True, alpha=0.3)

    if i == 4:
        axes[i].set_xlabel('Tempo (amostras)')

plt.tight_layout()
plt.savefig('ecg_examples.png', dpi=150)
print("\n  ✓ Exemplos salvos: ecg_examples.png")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados...")

# Normalizar
scaler = StandardScaler()
ecg_scaled = scaler.fit_transform(ecg_signals)

# Reshape para LSTM [samples, timesteps, features]
X = ecg_scaled.reshape(ecg_scaled.shape[0], ecg_scaled.shape[1], 1)
y = labels

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"  Treino: {X_train.shape[0]}")
print(f"  Validação: {X_val.shape[0]}")
print(f"  Teste: {X_test.shape[0]}")

# ─── 3. MODELO 1: LSTM SIMPLES ───
print("\n🔨 Modelo 1: LSTM Simples...")

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(n_timesteps, 1)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')
], name='LSTM')

model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model_lstm.count_params():,}")

history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

acc_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_lstm:.4f}")

# ─── 4. MODELO 2: CNN-LSTM HÍBRIDO ───
print("\n🔨 Modelo 2: CNN-LSTM Híbrido...")

model_cnn_lstm = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(n_timesteps, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')
], name='CNN_LSTM')

model_cnn_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model_cnn_lstm.count_params():,}")

history_cnn_lstm = model_cnn_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

acc_cnn_lstm = model_cnn_lstm.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_cnn_lstm:.4f}")

# ─── 5. MODELO 3: BIDIRECTIONAL LSTM ───
print("\n🔨 Modelo 3: Bidirectional LSTM...")

model_bilstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(n_timesteps, 1)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')
], name='BiLSTM')

model_bilstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model_bilstm.count_params():,}")

history_bilstm = model_bilstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

acc_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_bilstm:.4f}")

# ─── 6. COMPARAR MODELOS ───
print("\n📊 Comparando modelos...")

comparison = pd.DataFrame({
    'Modelo': ['LSTM', 'CNN-LSTM', 'BiLSTM'],
    'Parâmetros': [
        model_lstm.count_params(),
        model_cnn_lstm.count_params(),
        model_bilstm.count_params()
    ],
    'Accuracy': [acc_lstm, acc_cnn_lstm, acc_bilstm]
})

print("\n" + comparison.to_string(index=False))

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
comparison.plot(x='Modelo', y='Accuracy', kind='bar', ax=axes[0], legend=False, color='skyblue')
axes[0].set_title('Accuracy por Modelo', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticklabels(comparison['Modelo'], rotation=0)

# Histórico do melhor modelo
best_idx = comparison['Accuracy'].idxmax()
best_name = comparison.iloc[best_idx]['Modelo']

if best_name == 'LSTM':
    history = history_lstm
elif best_name == 'CNN-LSTM':
    history = history_cnn_lstm
else:
    history = history_bilstm

axes[1].plot(history.history['accuracy'], label='Treino', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
axes[1].set_title(f'Treinamento - {best_name}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_model_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: ecg_model_comparison.png")

# ─── 7. ANÁLISE DETALHADA DO MELHOR MODELO ───
best_model = model_bilstm if acc_bilstm >= max(acc_lstm, acc_cnn_lstm) else \
             (model_cnn_lstm if acc_cnn_lstm >= acc_lstm else model_lstm)

print(f"\n🏆 Melhor modelo: {best_name}")

# Predições
y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Matriz de Confusão - {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('ecg_confusion_matrix.png', dpi=150)
print("  ✓ Matriz de confusão salva: ecg_confusion_matrix.png")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_labels))

# ─── 8. ANÁLISE ROC POR CLASSE ───
print("\n📊 Curvas ROC por classe...")

from sklearn.preprocessing import label_binarize

# Binarizar labels
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

# Calcular ROC para cada classe
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)

    axes[i].plot(fpr, tpr, color='darkorange', linewidth=2, 
                 label=f'ROC (AUC = {roc_auc:.3f})')
    axes[i].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    axes[i].set_title(f'{class_labels[i]}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('FPR')
    axes[i].set_ylabel('TPR')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Remover último subplot vazio
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('ecg_roc_curves.png', dpi=150)
print("  ✓ Curvas ROC salvas: ecg_roc_curves.png")

# ─── 9. PREDIÇÃO EM NOVO SINAL ───
print("\n💓 Testando em novo sinal ECG...")

# Gerar novo sinal
test_signal = generate_abnormal_ecg(n_timesteps, 'atrial_fibrillation')
test_signal_scaled = scaler.transform(test_signal.reshape(1, -1))
test_signal_reshaped = test_signal_scaled.reshape(1, n_timesteps, 1)

# Prever
pred_probs = best_model.predict(test_signal_reshaped, verbose=0)[0]
pred_class = np.argmax(pred_probs)

# Visualizar
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Sinal
axes[0].plot(test_signal, color='blue', linewidth=2)
axes[0].set_title(f'Sinal ECG de Teste - Previsto: {class_labels[pred_class]}', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Amplitude (mV)')
axes[0].set_xlabel('Tempo (amostras)')
axes[0].grid(True, alpha=0.3)

# Probabilidades
axes[1].bar(class_labels, pred_probs, color=['green' if i == pred_class else 'gray' 
                                               for i in range(5)])
axes[1].set_title('Probabilidades por Classe', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Probabilidade')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ecg_prediction_example.png', dpi=150)
print("  ✓ Exemplo de predição salvo: ecg_prediction_example.png")

print(f"\n  Classe prevista: {class_labels[pred_class]}")
print(f"  Probabilidades:")
for i, label in enumerate(class_labels):
    print(f"    {label}: {pred_probs[i]*100:.2f}%")

# ─── 10. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ ANÁLISE DE ECG COM LSTM CONCLUÍDA!")
print("="*70)

print(f"\n📊 Resultados:")
print(f"  • LSTM: {acc_lstm:.4f}")
print(f"  • CNN-LSTM: {acc_cnn_lstm:.4f}")
print(f"  • BiLSTM: {acc_bilstm:.4f}")
print(f"  • Melhor: {best_name} ({max(acc_lstm, acc_cnn_lstm, acc_bilstm):.4f})")

print("\n📁 Arquivos gerados:")
print("  • ecg_examples.png - Exemplos de sinais ECG")
print("  • ecg_model_comparison.png - Comparação de modelos")
print("  • ecg_confusion_matrix.png - Matriz de confusão")
print("  • ecg_roc_curves.png - Curvas ROC por classe")
print("  • ecg_prediction_example.png - Exemplo de predição")

print("\n💡 Aplicações médicas:")
print("  • Monitoramento cardíaco contínuo")
print("  • Detecção precoce de arritmias")
print("  • Triagem automática em emergências")
print("  • Telemedicina e dispositivos wearables")

print("\n🔧 Próximos passos:")
print("  • Usar dados reais (MIT-BIH Arrhythmia Database)")
print("  • Implementar detecção em tempo real")
print("  • Adicionar explicabilidade (SHAP, Grad-CAM)")
print("  • Transfer learning com modelos pré-treinados")
