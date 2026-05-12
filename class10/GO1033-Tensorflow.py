# GO1033-Tensorflow
# Exibe o resumo textual da arquitetura do modelo e gera um diagrama visual em PNG
# com formatos de camadas, útil para inspecionar e documentar a rede.
# Resumo do modelo
# model.summary()

# Gerar imagem do modelo
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True,
#            show_layer_names=True, rankdir='TB')

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

    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.utils import plot_model

    # Constrói modelo, chama summary e gera diagrama visual com plot_model
    inputs  = Input(shape=(784,), name='input')
    x       = Dense(128, activation='relu', name='dense_1')(inputs)
    x       = Dense(64, activation='relu', name='dense_2')(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model   = Model(inputs=inputs, outputs=outputs, name='modelo_exemplo')

    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True,
               show_layer_names=True, rankdir='TB')

    # Gráfico de barras mostrando o número de parâmetros por camada do modelo
    layer_names  = [layer.name for layer in model.layers if hasattr(layer, 'count_params')]
    param_counts = [layer.count_params() for layer in model.layers if hasattr(layer, 'count_params')]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(layer_names, param_counts, color='#2980B9')
    for bar, val in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha='center', va='bottom', fontsize=9)
    ax.set_title('Número de Parâmetros por Camada')
    ax.set_xlabel('Camada')
    ax.set_ylabel('Parâmetros')
    plt.tight_layout()
    plt.show()
