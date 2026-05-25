# GO1252-Sklearn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# Classification Report
# Métricas por classe:
#   Precision = TP / (TP + FP)  → "quando digo que é X, acerto quanto?"
#   Recall    = TP / (TP + FN)  → "de todos os X reais, encontro quantos?"
#   F1-score  = 2 × (P × R) / (P + R)  → média harmônica entre precision e recall


if __name__ == "__main__":
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix: matriz N×N onde cm[i,j] = qtd de amostras da classe i
    # classificadas como classe j
    # Diagonal principal: acertos (TP de cada classe)
    # Fora da diagonal: erros (FP para coluna, FN para linha)
    cm = confusion_matrix(y_true, y_pred)

    # ─── VISUALIZAÇÃO: CONFUSION MATRIX HEATMAP ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Contagens absolutas
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Matriz de Confusão (Contagens)', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    tick_marks = np.arange(len(class_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(class_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=8,
                         color='white' if cm[i, j] > thresh else 'black')
    axes[0].set_xlabel('Predito')
    axes[0].set_ylabel('Real')

    # Normalizada (percentual)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Matriz de Confusão (Normalizada %)', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(class_names, fontsize=8)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center', fontsize=7,
                         color='white' if cm_norm[i, j] > 0.5 else 'black')
    axes[1].set_xlabel('Predito')
    axes[1].set_ylabel('Real')

    # Acurácia por classe na diagonal
    per_class_acc = np.diag(cm_norm)
    overall_acc = np.trace(cm) / np.sum(cm)
    plt.suptitle(f'Avaliação do Modelo | Acurácia Geral: {overall_acc:.4f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Barras de acurácia por classe
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['#59a14f' if a >= 0.9 else '#f28e2b' if a >= 0.7 else '#e15759'
              for a in per_class_acc]
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor='black')
    ax.axhline(y=overall_acc, color='navy', linestyle='--', linewidth=2,
               label=f'Média: {overall_acc:.4f}')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Acurácia')
    ax.set_title('Acurácia por Classe', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()
