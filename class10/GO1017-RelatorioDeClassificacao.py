# GO1017-RelatórioDeClassificação
# Exibe o relatório completo de classificação com precision, recall e F1-score
# por classe, complementando a matriz de confusão com métricas textuais.
# Relatório de classificação
# print(classification_report(y_test, y_pred_classes))

if __name__ == "__main__":
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
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from sklearn.metrics import classification_report

    # Constrói e treina modelo MNIST, gera predições para o relatório de classificação
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=0)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    print(classification_report(y_test, y_pred_classes))

    # Extrai métricas do classification_report e plota heatmap de precision/recall/f1 por classe
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_classes, labels=list(range(10)))
    metrics_matrix = np.array([precision, recall, f1])

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='YlGn',
                xticklabels=[str(i) for i in range(10)],
                yticklabels=['Precision', 'Recall', 'F1-Score'],
                ax=ax)
    ax.set_title('Classification Report — Precision / Recall / F1 por Classe')
    ax.set_xlabel('Classe (dígito)')
    plt.tight_layout()
    plt.show()
