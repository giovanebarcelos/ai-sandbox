# GO1012-Callbacks
# Configura três callbacks essenciais para treino robusto: EarlyStopping para parar
# sem melhora, ModelCheckpoint para salvar o melhor modelo e ReduceLROnPlateau
# para reduzir o learning rate automaticamente quando a val_loss estacionar.
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'mnist_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
