# GO1005-Tensorflow
# Verifica a versão instalada do TensorFlow e lista os dispositivos GPU disponíveis,
# útil para confirmar o ambiente de execução antes de treinar modelos.
import tensorflow as tf
print(tf.__version__)  # 2.x.x
print(tf.config.list_physical_devices('GPU'))

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":

    # Imprime versão e dispositivos disponíveis no ambiente
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CPUs: {len(cpus)}  |  GPUs: {len(gpus)}")

    # Gráfico de pizza mostrando a proporção de CPU e GPU disponíveis
    labels = ['CPU disponível', f'GPU disponível ({len(gpus)})']
    sizes  = [len(cpus), max(len(gpus), 0)]
    if sum(sizes) == 0:
        sizes = [1, 0]
    colors = ['#5DADE2', '#F39C12']
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f'{int(round(p*sum(sizes)/100))}' if p > 0 else '0',
        startangle=90
    )
    ax.set_title(f'Dispositivos disponíveis\n(TF {tf.__version__})')
    plt.tight_layout()
    plt.show()
