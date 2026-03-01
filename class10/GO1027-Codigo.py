# GO1027-Codigo
learning_rates = [0.001, 0.0001]
hidden_units = [128, 256, 512]
dropout_rates = [0.2, 0.3, 0.5]
results = []

for lr in learning_rates:
    for units in hidden_units:
        for dropout in dropout_rates:
            print(f"Testing: LR={lr}, Units={units}, Drop={dropout}")

            model = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(units, 'relu'),
                Dropout(dropout),
                Dense(10, 'softmax')
            ])

            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                callbacks=[EarlyStopping(patience=5)],
                verbose=0
            )

            val_acc = max(history.history['val_accuracy'])
            results.append({
                'lr': lr, 'units': units, 'dropout': dropout,
                'val_acc': val_acc
            })

# Encontrar melhor combinação
import pandas as pd
df = pd.DataFrame(results).sort_values('val_acc', ascending=False)
print(df.head())
