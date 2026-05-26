# GO0931-Model
# ═══════════════════════════════════════════════════════════════════
# MODELO KERAS — MLP MNIST
# Slide 22: Arquitetura com Keras/TensorFlow
# ═══════════════════════════════════════════════════════════════════
"""
Modelo Sequential do Keras: Dense(128) → Dense(64) → Dense(10).
Compilado com adam + categorical_crossentropy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TF_OK = True
except ImportError:
    TF_OK = False


MODEL_CODE = """
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64,  activation='relu'),
    Dense(10,  activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
"""

if __name__ == "__main__":
    print("=" * 60)
    print("MODELO KERAS — MLP PARA MNIST")
    print("=" * 60)
    print(MODEL_CODE)

    if TF_OK:
        # Dados sintéticos para demo sem download
        np.random.seed(42)
        N_tr, N_te = 2000, 400
        X_train = np.random.randn(N_tr, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, N_tr)
        X_test  = np.random.randn(N_te, 784).astype(np.float32)
        y_test  = np.random.randint(0, 10, N_te)
        y_tr_oh = tf.keras.utils.to_categorical(y_train, 10)
        y_te_oh = tf.keras.utils.to_categorical(y_test,  10)

        model = Sequential([
            Dense(128, activation="relu", input_shape=(784,)),
            Dense(64,  activation="relu"),
            Dense(10,  activation="softmax"),
        ])
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()

        history = model.fit(X_train, y_tr_oh, epochs=10, batch_size=64,
                            validation_split=0.2, verbose=1)
        _, acc = model.evaluate(X_test, y_te_oh, verbose=0)
        print(f"\nTest accuracy (dados sintéticos): {acc*100:.1f}%")

        plt.figure(figsize=(8, 3))
        plt.plot(history.history["accuracy"],     label="Treino")
        plt.plot(history.history["val_accuracy"], label="Validação")
        plt.xlabel("Época"); plt.ylabel("Acurácia")
        plt.title("Keras MLP — Dense(128)→Dense(64)→Dense(10)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("GO0931_model_keras.png", dpi=100, bbox_inches="tight")
        plt.show(); print("Salvo: GO0931_model_keras.png")
    else:
        print("TensorFlow não encontrado.")
        print("Instale com: pip install tensorflow")
        print("\n📌 O modelo acima é o padrão para classificar MNIST com Keras.")
