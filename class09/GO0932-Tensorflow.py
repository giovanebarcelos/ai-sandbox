# GO0932-Tensorflow
# ═══════════════════════════════════════════════════════════════════
# TENSORFLOW — VERIFICAÇÃO E CONFIGURAÇÃO
# Slide 22: Setup TensorFlow/Keras
# ═══════════════════════════════════════════════════════════════════
"""
Verifica instalação do TensorFlow, versão, GPU disponível,
e roda um smoke-test básico para confirmar que o ambiente está OK.
"""

import sys
import numpy as np

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False


def verificar_tensorflow():
    """Verifica instalação e configuração do TensorFlow."""
    print("=" * 55)
    print("VERIFICAÇÃO DO AMBIENTE TENSORFLOW")
    print("=" * 55)

    if not TF_OK:
        print("❌ TensorFlow NÃO instalado.")
        print("   Instale com: pip install tensorflow")
        return False

    print(f"✅ TensorFlow instalado: v{tf.__version__}")
    assert tf.__version__ >= "2.0", "Requer TensorFlow >= 2.0"

    # GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU disponível: {len(gpus)} dispositivo(s)")
        for g in gpus:
            print(f"   → {g.name}")
    else:
        print("ℹ️  GPU não detectada — rodando em CPU (OK para datasets pequenos)")

    # Versão Python e NumPy
    print(f"✅ Python: {sys.version.split()[0]}")
    print(f"✅ NumPy:  {np.__version__}")

    return True


def smoke_test():
    """Smoke test básico: criar e avaliar um modelo simples."""
    print("\n" + "=" * 55)
    print("SMOKE TEST — Modelo mínimo")
    print("=" * 55)

    # Dados sintéticos
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y = tf.keras.utils.to_categorical(
        np.random.randint(0, 3, 200), 3).astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(10,)),
        tf.keras.layers.Dense(3,  activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(X, y, epochs=5, verbose=0, validation_split=0.2)
    loss, acc = model.evaluate(X, y, verbose=0)

    print(f"  Loss final:    {loss:.4f}")
    print(f"  Accuracy:      {acc*100:.1f}%")
    print("  ✅ Smoke test OK — TensorFlow funcionando corretamente!")


if __name__ == "__main__":
    ok = verificar_tensorflow()
    if ok:
        smoke_test()
        print("\n📌 Próximo passo: GO0931 — Modelo Keras MLP para MNIST")
    else:
        print("\n📌 Instale TensorFlow antes de prosseguir:")
        print("   pip install tensorflow")
        print("   # ou para GPU: pip install tensorflow[and-cuda]")
