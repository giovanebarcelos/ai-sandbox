# GO0529-Sklearn
# Calcular pesos automaticamente
from sklearn.utils.class_weight import compute_class_weight


if __name__ == "__main__":
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )

    # Para ratio 19:1:
    # Classe 0 (majoritária): peso = 0.526
    # Classe 1 (minoritária): peso = 10.0  ← 19x maior!
