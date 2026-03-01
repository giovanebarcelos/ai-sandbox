# GO0505-RocCurveEAuc
# ═══════════════════════════════════════════════════════════════════
# ROC CURVE e AUC
# ═══════════════════════════════════════════════════════════════════

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# ───────────────────────────────────────────────────────────────────
# PREPARAR DADOS
# ───────────────────────────────────────────────────────────────────

# Carregar dados


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ───────────────────────────────────────────────────────────────────
    # TREINAR MODELO
    # ───────────────────────────────────────────────────────────────────

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # ───────────────────────────────────────────────────────────────────
    # PREPARAR PARA ROC (PROBLEMA BINÁRIO)
    # ───────────────────────────────────────────────────────────────────

    # Para problema binário (simplificando Iris para 2 classes)
    y_binary = (y_test == 2).astype(int)  # Classe 2 vs resto
    y_scores = model.predict_proba(X_test)[:, 2]  # Probabilidade classe 2

    # ───────────────────────────────────────────────────────────────────
    # CALCULAR ROC
    # ───────────────────────────────────────────────────────────────────

    fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    # ───────────────────────────────────────────────────────────────────
    # PLOTAR
    # ───────────────────────────────────────────────────────────────────

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.50)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall/Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    print("="*60)
    print("ROC-AUC ANALYSIS")
    print("="*60)
    print(f"AUC Score: {roc_auc:.3f}")
    print("\nInterpretação:")
    print("  AUC = 1.0:   Classificador perfeito")
    print("  AUC = 0.9-1: Excelente")
    print("  AUC = 0.8-0.9: Bom")
    print("  AUC = 0.7-0.8: Razoável")
    print("  AUC = 0.5-0.7: Ruim")
    print("  AUC = 0.5:   Random (linha diagonal)")
    print("  AUC < 0.5:   Pior que random (inverta predições!)")

    # ───────────────────────────────────────────────────────────────────
    # ENCONTRAR THRESHOLD ÓTIMO
    # ───────────────────────────────────────────────────────────────────

    # Threshold que maximiza (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nThreshold ótimo: {optimal_threshold:.3f}")
    print(f"  TPR (Recall): {tpr[optimal_idx]:.3f}")
    print(f"  FPR: {fpr[optimal_idx]:.3f}")
