# GO1401-5SimpleRnnEmKeras
from tensorflow.keras.layers import SimpleRNN


if __name__ == "__main__":
    model = Sequential([
        SimpleRNN(64, return_sequences=True,
                  input_shape=(timesteps, features)),
        SimpleRNN(32),  # return_sequences=False (default)
        Dense(10, activation='softmax')
    ])
