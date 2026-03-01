# GO1255-ReduzirLrQuando
# Reduzir LR quando val_loss estagnar


if __name__ == "__main__":
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Multiplicar LR por 0.5
        patience=3,        # Aguardar 3 épocas
        min_lr=1e-7
    )

    # Decay exponencial
    ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
