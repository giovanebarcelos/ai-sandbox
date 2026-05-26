# GO0911-Templates
# ═══════════════════════════════════════════════════════════════════
# TEMPLATE KERAS — MLP PARA CLASSIFICAÇÃO
# Slide 22: Arquitetura com Keras Sequential
# ═══════════════════════════════════════════════════════════════════
"""
Template de referência para construir um MLP com Keras/TensorFlow.
Usa dados sintéticos (make_moons) para rodar sem datasets externos.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TF_OK = True
except ImportError:
    TF_OK = False
    print("TensorFlow não instalado — exibindo template comentado.")


TEMPLATE_KERAS = """
# ── Template Keras MLP ──────────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.3),
    Dense(64,  activation='relu'),
    Dropout(0.2),
    Dense(10,  activation='softmax'),       # 10 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',        # multiclasse one-hot
    metrics=['accuracy'],
)

history = model.fit(
    X_train, y_train_oh,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"Test acc: {acc*100:.2f}%")
# ─────────────────────────────────────────────────────────
"""


if __name__ == "__main__":
    print("=" * 60)
    print("TEMPLATE KERAS — MLP CLASSIFICAÇÃO")
    print("=" * 60)
    print(TEMPLATE_KERAS)

    if TF_OK:
        print("\nTensorFlow disponível! Treinando exemplo real com make_moons...")
        X, y = make_moons(n_samples=800, noise=0.25, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        y_tr_oh = tf.keras.utils.to_categorical(y_tr, 2)
        y_te_oh = tf.keras.utils.to_categorical(y_te, 2)

        model = Sequential([
            Dense(128, activation="relu", input_shape=(2,)),
            Dropout(0.3),
            Dense(64,  activation="relu"),
            Dense(2,   activation="softmax"),
        ])
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        history = model.fit(X_tr, y_tr_oh, epochs=40, batch_size=32,
                            validation_split=0.2, verbose=0)
        _, acc = model.evaluate(X_te, y_te_oh, verbose=0)
        print(f"Test acc: {acc*100:.1f}%")

        plt.figure(figsize=(8, 4))
        plt.plot(history.history["accuracy"],     label="Treino")
        plt.plot(history.history["val_accuracy"], label="Validação")
        plt.xlabel("Época"); plt.ylabel("Acurácia")
        plt.title("Template Keras MLP — make_moons")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("GO0911_template_keras.png", dpi=100, bbox_inches="tight")
        plt.show()
        print("Salvo: GO0911_template_keras.png")
    else:
        print("Instale TensorFlow com: pip install tensorflow")
