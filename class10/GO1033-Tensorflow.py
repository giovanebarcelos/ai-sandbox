# GO1033-Tensorflow
# Exibe o resumo textual da arquitetura do modelo e gera um diagrama visual em PNG
# com formatos de camadas, útil para inspecionar e documentar a rede.
# Resumo do modelo
model.summary()

# Gerar imagem do modelo
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True,
           show_layer_names=True, rankdir='TB')
