# GO1258-Codigo


if __name__ == "__main__":
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
