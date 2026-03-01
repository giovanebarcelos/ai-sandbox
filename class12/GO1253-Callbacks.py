# GO1253-Callbacks


if __name__ == "__main__":
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint('best.h5', save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]
