# GO1402-12LstmEmKeras
from tensorflow.keras.layers import LSTM


if __name__ == "__main__":
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        LSTM(128, return_sequences=True, dropout=0.2),
        LSTM(64, dropout=0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
