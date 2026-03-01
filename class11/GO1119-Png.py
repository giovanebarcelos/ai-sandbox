# GO1119-Png
import tensorflow as tf

class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, n_rules):
        super().__init__()
        self.n_rules = n_rules

    def call(self, inputs):
        # Fuzzificação
        fuzzified = self.fuzzify(inputs)
        # Aplicar regras
        fired = self.apply_rules(fuzzified)
        # Defuzzificação
        output = self.defuzzify(fired)
        return output

# Usar no modelo


if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        FuzzyLayer(n_rules=10),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
