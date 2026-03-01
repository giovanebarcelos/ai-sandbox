# GO0421-Problema10ResultadosNãoReproduzíveis
# Fixar todas as seeds:
import numpy as np
import random
import os

def set_seeds(seed=42):
    """Fixar seeds para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Se estiver usando TensorFlow/PyTorch:
    # import tensorflow as tf
    # tf.random.set_seed(seed)

    # import torch
    # torch.manual_seed(seed)

# Usar no início do código:


if __name__ == "__main__":
    set_seeds(42)

    # E sempre usar random_state nos métodos:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
