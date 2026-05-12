# GO1027-Codigo
# Realiza busca manual de hiperparâmetros por força bruta (grid search): testa todas
# as combinações de learning rate, unidades e dropout, registrando a melhor val_accuracy
# de cada configuração e exibindo as melhores combinações ao final.
learning_rates = [0.001, 0.0001]
hidden_units = [128, 256, 512]
dropout_rates = [0.2, 0.3, 0.5]
results = []

# for lr in learning_rates:
#     for units in hidden_units:
#         for dropout in dropout_rates:
#             print(f"Testing: LR={lr}, Units={units}, Drop={dropout}")
#
#             model = Sequential([
#                 Flatten(input_shape=(28, 28)),
#                 Dense(units, 'relu'),
#                 Dropout(dropout),
#                 Dense(10, 'softmax')
#             ])
#
#             model.compile(
#                 optimizer=Adam(learning_rate=lr),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy']
#             )
#
#             history = model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=20,
#                 callbacks=[EarlyStopping(patience=5)],
#                 verbose=0
#             )
#
#             val_acc = max(history.history['val_accuracy'])
#             results.append({
#                 'lr': lr, 'units': units, 'dropout': dropout,
#                 'val_acc': val_acc
#             })

# # Encontrar melhor combinação
# import pandas as pd
# df = pd.DataFrame(results).sort_values('val_acc', ascending=False)
# print(df.head())

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # Carrega 200 amostras do MNIST para grid search rápido (5 épocas por trial)
    (X_full, y_full), _ = keras.datasets.mnist.load_data()
    X_full = X_full / 255.0
    X_train_gs = X_full[:160]
    y_train_gs = y_full[:160]
    X_val_gs   = X_full[160:200]
    y_val_gs   = y_full[160:200]

    results = []
    for lr in learning_rates:
        for units in hidden_units:
            for dropout in dropout_rates:
                print(f"Testing: LR={lr}, Units={units}, Drop={dropout}")
                m = Sequential([
                    Flatten(input_shape=(28, 28)),
                    Dense(units, 'relu'),
                    Dropout(dropout),
                    Dense(10, 'softmax')
                ])
                m.compile(optimizer=Adam(learning_rate=lr),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
                hist = m.fit(X_train_gs, y_train_gs,
                             validation_data=(X_val_gs, y_val_gs),
                             epochs=5,
                             callbacks=[EarlyStopping(patience=3)],
                             verbose=0)
                val_acc = max(hist.history['val_accuracy'])
                results.append({'lr': lr, 'units': units,
                                'dropout': dropout, 'val_acc': val_acc})

    df = pd.DataFrame(results).sort_values('val_acc', ascending=False)
    print(df.head())

    # Cria pivot table e plota heatmap de val_accuracy por units × dropout para lr=0.001
    df_lr = df[df['lr'] == 0.001]
    pivot = df_lr.pivot(index='units', columns='dropout', values='val_acc')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Grid Search — Val Accuracy (lr=0.001)')
    ax.set_xlabel('Dropout')
    ax.set_ylabel('Units')
    plt.tight_layout()
    plt.show()
