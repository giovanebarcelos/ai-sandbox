# GO1525-20TextClassificationComDeepLearning
  from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
  model = Sequential([
      Embedding(vocab_size, 128, input_length=max_len),
      Conv1D(128, 5, activation='relu'),
      GlobalMaxPooling1D(),
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(1, activation='sigmoid')
  ])
