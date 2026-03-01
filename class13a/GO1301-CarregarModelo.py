# GO1301-CarregarModelo
# Carregar modelo pré-treinado


if __name__ == "__main__":
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Congelar todas as camadas
    base_model.trainable = False

    # Adicionar novas camadas para seu problema
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
