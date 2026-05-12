# GO1064-BatchNormalization
# Modelo MLP com BatchNormalization: estabiliza distribuição das ativações entre
# camadas, acelera convergência e permite learning rates maiores no MNIST.
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# 1) Dados
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

# 2) Modelo COM BatchNormalization
#    Posicionamento mais comum: Dense (linear) → BatchNorm → Ativação
model = keras.Sequential([
    layers.Dense(256, input_shape=(784,)),        # sem ativação aqui
    layers.BatchNormalization(),                   # normaliza antes de ativar
    layers.Activation("relu"),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

# 3) Compilação — BatchNorm permite LR maior (0.01 em vez do padrão 0.001)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4) Treino
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

# 5) Avaliação
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia em teste (com BatchNorm): {test_acc:.4f}")

# 6) Curvas de convergência
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Treino")
axes[0].plot(history.history["val_accuracy"], label="Validação")
axes[0].set_title("Accuracy (com BatchNorm)")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Treino")
axes[1].plot(history.history["val_loss"], label="Validação")
axes[1].set_title("Loss (com BatchNorm)")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
