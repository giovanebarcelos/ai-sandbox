# GO1016-Sklearn
# Gera predições no conjunto de teste, calcula a matriz de confusão com sklearn
# e a visualiza como heatmap com seaborn para analisar erros por classe.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

if __name__ == "__main__":
    # Treinar modelo MNIST para gerar predições reais
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=128,
              validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - MNIST')
    plt.tight_layout()
    plt.show()
    print(classification_report(y_test, y_pred_classes))
