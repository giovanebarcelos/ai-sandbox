# GO1254-ExtrairFiltrosDa
# Extrair filtros da primeira camada


if __name__ == "__main__":
    filters, biases = model.layers[0].get_weights()
    print(f"Shape: {filters.shape}")  # (3, 3, 1, 32)

    # Extrair feature maps
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    feature_model = Model(inputs=model.input, outputs=layer_outputs)
    feature_maps = feature_model.predict(image)
