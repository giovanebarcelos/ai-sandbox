# GO0915-AumentarLearning
# ═══════════════════════════════════════════════════════════════════
# GUIA DE DEBUGGING - MNIST
# Problema 1: Accuracy muito baixa (<50% após 10 épocas)
# ═══════════════════════════════════════════════════════════════════
#
# Sintoma : Train accuracy abaixo de 50% após 10 épocas
# Causa   : Learning rate muito baixa, inicialização ruim ou dados
#           não normalizados
#
# Outros problemas → ver arquivos:
#   GO0916-ReduzirLearning.py      → Loss explode (NaN/Inf)
#   GO0917-ReduzirComplexidade.py  → Overfitting severo
#   GO0918-UsarSubset.py           → Treinamento lento (>2 min/época)
#   GO0919-VerificarShapes.py      → ValueError: shapes not aligned
# ═══════════════════════════════════════════════════════════════════

import numpy as np

# ───────────────────────────────────────────────────────────────────
# CORREÇÃO 1: Ajustar learning rate
# ───────────────────────────────────────────────────────────────────
# Valores para testar em ordem: 0.001 → 0.01 → 0.1 → 0.5
# Se accuracy sobe mas oscila muito: reduza (ver GO0916)
LEARNING_RATE = 0.1

# ───────────────────────────────────────────────────────────────────
# CORREÇÃO 2: He initialization (melhor que inicialização aleatória simples para ReLU)
# ───────────────────────────────────────────────────────────────────
def init_weights_he(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

# ───────────────────────────────────────────────────────────────────
# CORREÇÃO 3: Verificar normalização dos dados
# ───────────────────────────────────────────────────────────────────
def verificar_normalizacao(X_train):
    print(f"X_train min={X_train.min():.4f}, max={X_train.max():.4f}")
    if X_train.max() > 1.0:
        print("⚠️  PROBLEMA: dados não normalizados! Divida por 255.0")
        print("   X_train = X_train / 255.0")
    else:
        print("✓ Normalização OK: valores em [0, 1]")

# ───────────────────────────────────────────────────────────────────
# DIAGNÓSTICO RÁPIDO
# ───────────────────────────────────────────────────────────────────
def diagnosticar_accuracy_baixa(model, X_train, y_train_oh):
    """Imprime diagnóstico para identificar a causa de accuracy baixa."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO - Accuracy Baixa")
    print("="*60)

    verificar_normalizacao(X_train)

    pred = model.forward(X_train[:100])
    loss = -np.mean(np.sum(y_train_oh[:100] * np.log(np.clip(pred, 1e-15, 1)), axis=1))
    acc  = np.mean(np.argmax(pred, axis=1) == np.argmax(y_train_oh[:100], axis=1))

    print(f"Loss inicial : {loss:.4f}  (esperado ≈ 2.30 = -log(1/10))")
    print(f"Accuracy     : {acc*100:.1f}%  (esperado ≈ 10% aleatório no início)")

    if loss > 10:
        print("⚠️  Loss muito alta → verifique normalização e one-hot encoding")
    if acc < 0.15 and loss < 3:
        print("⚠️  Accuracy próxima ao acaso → aumente o learning rate")

    print("\nSugestão: recrie o modelo com LEARNING_RATE = 0.1 e He initialization")
    print("="*60)
