# GO1009-ClassificaçãoBinária
# Classificação binária
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Classificação multiclasse (MNIST)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Regressão
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Com learning rate customizado
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
