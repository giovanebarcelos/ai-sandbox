# GO0404-LearningCurvesDiagnósticoDeOverfittingunderfitting
# ═══════════════════════════════════════════════════════════════════
# LEARNING CURVES - DIAGNÓSTICO DE OVERFITTING/UNDERFITTING
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits

# Carregar dados


if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target

    # ───────────────────────────────────────────────────────────────────
    # FUNÇÃO PARA PLOTAR LEARNING CURVES
    # ───────────────────────────────────────────────────────────────────

    def plot_learning_curve(estimator, title, X, y, cv=5):
        """
        Plota learning curves para diagnóstico
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        # Calcular médias e desvios
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plotar
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Treino')
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validação')

        # Área de desvio padrão
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, val_mean - val_std,
                         val_mean + val_std, alpha=0.1, color='g')

        plt.xlabel('Tamanho do Conjunto de Treino')
        plt.ylabel('Acurácia')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        y_min = max(0.0, min(train_mean.min(), val_mean.min()) - 0.1)
        plt.ylim([y_min, 1.05])
        plt.show()

    # ───────────────────────────────────────────────────────────────────
    # COMPARAR 3 CENÁRIOS
    # ───────────────────────────────────────────────────────────────────

    # 1. Underfitting (árvore muito simples)
    print("1. UNDERFITTING (max_depth=1)")
    underfit_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    plot_learning_curve(underfit_model, 'Underfitting (Árvore Rasa)', X, y)

    # 2. Bom ajuste
    print("\n2. BOM AJUSTE (max_depth=5)")
    good_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    plot_learning_curve(good_model, 'Bom Ajuste (Balanceado)', X, y)

    # 3. Overfitting (árvore muito profunda)
    print("\n3. OVERFITTING (max_depth=None)")
    overfit_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    plot_learning_curve(overfit_model, 'Overfitting (Árvore Profunda)', X, y)

    print("\n" + "="*60)
    print("INTERPRETAÇÃO:")
    print("="*60)
    print("Underfitting: Ambas curvas baixas, próximas")
    print("Bom ajuste: Ambas curvas altas, gap pequeno")
    print("Overfitting: Treino alto, validação baixo, GAP GRANDE")
