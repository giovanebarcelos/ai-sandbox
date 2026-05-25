# GO1250-TensorflowTensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dados


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Modelo mais profundo
    model = keras.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        # Bloco 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        # Bloco 3
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        # Classificador
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compilar e treinar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=50,
        validation_data=(x_test, y_test)
    )
    # Resultado esperado: ~85-90% accuracy

    # ─── VISUALIZAÇÃO: CURVAS DE TREINAMENTO ───
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#4e79a7')
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#e15759', linestyle='--')
    axes[0].set_title('Loss por Época - CIFAR-10', fontsize=13)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#4e79a7')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#e15759', linestyle='--')
    axes[1].axhline(y=0.85, color='green', linestyle=':', linewidth=1.5, label='Meta 85%')
    axes[1].set_title('Acurácia por Época - CIFAR-10', fontsize=13)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    plt.suptitle(f'CNN CIFAR-10 com Data Augmentation — Acurácia Final: {test_acc:.4f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ─── VISUALIZAÇÃO: AMOSTRAS DO CIFAR-10 ───
    x_test_display = (x_test * 255).astype('uint8')
    preds = model.predict(x_test[:25], verbose=0)
    pred_labels = preds.argmax(axis=1)
    true_labels = y_test[:25].argmax(axis=1)

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test_display[i])
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        ax.set_title(f'P:{class_names[pred_labels[i]][:5]}\nR:{class_names[true_labels[i]][:5]}',
                     fontsize=8, color=color)
        ax.axis('off')
    plt.suptitle('CIFAR-10: Predições (verde=correto, vermelho=erro)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
