# GO1065-BatchNormComparacaoSem
# Modelo MLP idêntico ao GO1064, mas SEM BatchNormalization: permite comparar
# velocidade de convergência, estabilidade e acurácia final entre os dois.
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

# 1) Dados (idêntico ao GO1064 para comparação justa)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

# 2) Modelo SEM BatchNormalization — mesma arquitetura base do GO1064
#    Usa LR padrão 0.001; com LR=0.01 (como no GO1064) a convergência seria
#    instável sem BatchNorm.
model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

# 3) Compilação com LR padrão (0.001) — sem BatchNorm, LR maior tende a divergir
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
print(f"Acurácia em teste (sem BatchNorm): {test_acc:.4f}")

# 6) Curvas de convergência
# Observar: curva de loss tende a ser mais ruidosa e convergência mais lenta
# do que no GO1064 com BatchNorm
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Treino")
axes[0].plot(history.history["val_accuracy"], label="Validação")
axes[0].set_title("Accuracy (sem BatchNorm)")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Treino")
axes[1].plot(history.history["val_loss"], label="Validação")
axes[1].set_title("Loss (sem BatchNorm)")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
