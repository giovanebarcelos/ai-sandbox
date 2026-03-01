# GO1233-Durante
# Durante TREINO:


if __name__ == "__main__":
    x = Dense(4096)(x)
    x = Dropout(0.5)(x)  # 50% neurônios = 0
    # Cada batch treina uma "sub-rede" diferente

    # Durante TESTE:
    x = Dense(4096)(x)
    # Usa TODOS neurônios (scaled)
