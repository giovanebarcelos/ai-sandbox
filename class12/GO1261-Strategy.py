# GO1261-Strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()  # Distribui automaticamente
