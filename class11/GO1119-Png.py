# GO1119-Png
# FuzzyLayer — camada personalizada Keras que implementa as 3 etapas do
# Mamdani (fuzzificação, regras, defuzzificação) como uma operação diferenciável,
# permitindo que os parâmetros das MFs sejam aprendidos por backpropagation.
#
# Exemplo: usada entre camadas densas para injetar raciocínio fuzzy na rede.
# n_rules=10 significa 10 regras fuzzy aprendíveis nesta camada.
#
# Para outro problema: ajuste n_rules conforme a complexidade do domínio.
# Os métodos fuzzify(), apply_rules() e defuzzify() precisam ser implementados
# conforme o tipo de MF escolhido (triangular, gaussiana, etc.).
# Requer: pip install tensorflow
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
