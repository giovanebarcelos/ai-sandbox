# GO1004-TensorflowTensorflow
# Mostra as duas formas equivalentes de importar a API Keras a partir do TensorFlow 2.x.
from tensorflow import keras
# ou
import tensorflow.keras as keras

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

    # Imprime a versão do Keras disponível via TensorFlow
    print(keras.__version__)

    # Gráfico de anotação mostrando a hierarquia tf -> keras -> Sequential/layers
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.text(5, 5, 'tensorflow (tf)', ha='center', va='center',
            fontsize=14, bbox=dict(boxstyle='round', facecolor='#AED6F1', edgecolor='navy'))
    ax.text(5, 3.2, 'tf.keras', ha='center', va='center',
            fontsize=13, bbox=dict(boxstyle='round', facecolor='#A9DFBF', edgecolor='green'))
    ax.text(2.5, 1.4, 'Sequential', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='#FAD7A0', edgecolor='orange'))
    ax.text(7.5, 1.4, 'layers', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='#FAD7A0', edgecolor='orange'))
    ax.annotate('', xy=(5, 3.7), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    ax.annotate('', xy=(2.5, 1.9), xytext=(4.2, 2.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(7.5, 1.9), xytext=(5.8, 2.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.set_title('Hierarquia: tf → keras → Sequential / layers')
    plt.tight_layout()
    plt.show()
