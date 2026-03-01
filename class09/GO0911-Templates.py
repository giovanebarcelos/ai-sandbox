# GO0911-Templates


if __name__ == "__main__":
    model = Sequential([Dense(128, 'relu'), Dense(10, 'softmax')])
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.fit(X, y, epochs=10, validation_split=0.2)
