# GO1020-Ver
# Dicas de depuração: inspecionar labels, verificar o range dos dados e confirmar
# a função de perda correta para evitar erros silenciosos de configuração.
# print(y_train[:10])                    # Ver se labels fazem sentido
# print(X_train.min(), X_train.max())    # Range dos dados
# model.compile(loss='sparse_categorical_crossentropy', ...)  # Loss correta

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente
    from tensorflow import keras

    # Carrega MNIST e inspeciona os labels e o range dos dados
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    print(y_train[:10])
    print(X_train.min(), X_train.max())

    # Gráfico de histograma mostrando a distribuição dos labels em y_train
    plt.figure(figsize=(8, 4))
    plt.hist(y_train, bins=10, rwidth=0.8, color='#3498DB', edgecolor='white')
    plt.xticks(range(10))
    plt.title('Distribuição dos Labels em y_train (MNIST)')
    plt.xlabel('Classe (dígito)')
    plt.ylabel('Quantidade de amostras')
    plt.tight_layout()
    plt.show()
