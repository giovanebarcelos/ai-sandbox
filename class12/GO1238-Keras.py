# GO1238-Keras
from keras.models import Model

# Extrair saídas de cada camada


if __name__ == "__main__":
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Visualizar o que cada camada "vê"
    activations = activation_model.predict(img)

    # BLOCO 1 (conv1): Detecta bordas, cores básicas
    # BLOCO 2 (conv2): Texturas simples (listras, grades)
    # BLOCO 3 (conv3): Padrões complexos (olhos, rodas)
    # BLOCO 4 (conv4): Partes de objetos (faces, portas)
    # BLOCO 5 (conv5): Objetos completos (carros, cães)
