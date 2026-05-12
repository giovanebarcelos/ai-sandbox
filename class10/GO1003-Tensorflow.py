# GO1003-Tensorflow
# Demonstra a execução eager do TensorFlow 2.x: operações em tensores são executadas
# imediatamente, sem sessões ou grafos explícitos.
import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b
print(c)  # Sem sessões! Sem placeholders! Só código normal!

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    import numpy as np

    # Operações sobre tensores a=[1,2,3], b=[4,5,6] e sua soma
    a_vals = a.numpy()
    b_vals = b.numpy()
    c_vals = c.numpy()

    # Gráfico de barras agrupadas mostrando os valores de a, b e a+b lado a lado
    x = np.arange(len(a_vals))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, a_vals, width, label='a = [1,2,3]')
    ax.bar(x,         b_vals, width, label='b = [4,5,6]')
    ax.bar(x + width, c_vals, width, label='a+b = [5,7,9]')
    ax.set_xticks(x)
    ax.set_xticklabels(['índice 0', 'índice 1', 'índice 2'])
    ax.set_title('Operações com Tensores (TF Eager)')
    ax.set_ylabel('Valor')
    ax.legend()
    plt.tight_layout()
    plt.show()
