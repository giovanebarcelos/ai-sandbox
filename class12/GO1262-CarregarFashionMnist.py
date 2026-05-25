# GO1262-CarregarFashionMnist
# Carregar Fashion MNIST
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == "__main__":
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = \
        keras.datasets.fashion_mnist.load_data()

    # Classes
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker',
                   'Bag', 'Ankle boot']

    # ─── VISUALIZAÇÃO: AMOSTRAS DO FASHION MNIST ───
    # Grid com 5 exemplos de cada classe
    fig, axes = plt.subplots(10, 6, figsize=(12, 20))
    for cls_idx in range(10):
        # Encontrar exemplos dessa classe
        indices = np.where(y_train == cls_idx)[0][:5]
        # Label na primeira coluna
        axes[cls_idx, 0].text(0.5, 0.5, f'{cls_idx}\n{class_names[cls_idx]}',
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               transform=axes[cls_idx, 0].transAxes)
        axes[cls_idx, 0].axis('off')
        axes[cls_idx, 0].set_facecolor('#f0f0f0')
        # Mostrar 5 exemplos
        for j, idx in enumerate(indices):
            axes[cls_idx, j + 1].imshow(x_train[idx], cmap='gray')
            axes[cls_idx, j + 1].axis('off')

    plt.suptitle('Fashion MNIST - 5 Exemplos por Classe (28×28, escala cinza)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    # ─── VISUALIZAÇÃO: DISTRIBUIÇÃO DAS CLASSES ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribuição no treino
    counts_train = [np.sum(y_train == i) for i in range(10)]
    ax1.bar(class_names, counts_train, color=plt.cm.tab10(np.linspace(0, 1, 10)),
            edgecolor='black')
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Número de Amostras')
    ax1.set_title(f'Distribuição no Treino ({len(x_train):,} amostras)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, cnt in enumerate(counts_train):
        ax1.text(i, cnt + 50, str(cnt), ha='center', va='bottom', fontsize=8)

    # Distribuição no teste
    counts_test = [np.sum(y_test == i) for i in range(10)]
    ax2.bar(class_names, counts_test, color=plt.cm.tab10(np.linspace(0, 1, 10)),
            edgecolor='black')
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Número de Amostras')
    ax2.set_title(f'Distribuição no Teste ({len(x_test):,} amostras)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, cnt in enumerate(counts_test):
        ax2.text(i, cnt + 5, str(cnt), ha='center', va='bottom', fontsize=8)

    print(f'\nDataset Fashion MNIST:')
    print(f'  Treino: {x_train.shape} | Labels: {y_train.shape}')
    print(f'  Teste:  {x_test.shape} | Labels: {y_test.shape}')
    print(f'  Range de pixels: [{x_train.min()}, {x_train.max()}]')

    plt.suptitle('Fashion MNIST - Distribuição das Classes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
