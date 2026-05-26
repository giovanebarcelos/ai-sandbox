# GO0927-Sklearn
# ═══════════════════════════════════════════════════════════════════
# SCIKIT-LEARN — MLP CLASSIFIER
# Slide 28: MLP com sklearn.MLPClassifier
# ═══════════════════════════════════════════════════════════════════
"""
sklearn.neural_network.MLPClassifier: MLP pronto para usar.
Ótimo para referência e benchmark antes de implementar do zero.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


if __name__ == "__main__":
    # Dataset make_circles
    X, y = make_circles(n_samples=600, noise=0.1, factor=0.5, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    print("=" * 55)
    print("MLPClassifier — sklearn (make_circles)")
    print("=" * 55)

    # Modelo
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),   # 2 camadas ocultas
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        verbose=False,
    )
    mlp.fit(X_tr, y_tr)

    y_pred = mlp.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\nTest Accuracy: {acc*100:.1f}%")
    print(f"Épocas treinadas: {mlp.n_iter_}")
    print("\n" + classification_report(y_te, y_pred))

    # Curva de loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(mlp.loss_curve_, color="blue",   label="Treino")
    if mlp.validation_scores_ is not None:
        axes[0].plot([1 - s for s in mlp.validation_scores_],
                     color="orange", linestyle="--", label="1 - Val Score")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Curva de Loss — MLPClassifier"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Fronteira de decisão
    h = 0.02
    x_min, x_max = X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5
    y_min, y_max = X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    axes[1].scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=plt.cm.RdYlBu,
                    edgecolors="k", s=20, label="Treino")
    axes[1].scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap=plt.cm.RdYlBu,
                    edgecolors="k", s=20, marker="^", label="Teste")
    axes[1].set_title(f"Fronteira de Decisão (acc={acc*100:.1f}%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("GO0927_mlp_sklearn.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0927_mlp_sklearn.png")
