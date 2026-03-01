# GO1002-Model
model = Sequential([
    Dense(128, 'relu'),
    Dense(10, 'softmax')
])
model.compile('adam', 'categorical_crossentropy')
model.fit(X, y, epochs=10)
