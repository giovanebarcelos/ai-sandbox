# GO0907-IniciandoTreinamento
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: RECONHECIMENTO DE DÍGITOS (MNIST)
# Etapa 3: TREINAR MODELO
# ═══════════════════════════════════════════════════════════════════

# TREINAR REDE NO MNIST
print("\n" + "="*70)
print("INICIANDO TREINAMENTO")
print("="*70)
# Criar rede
nn_mnist = MulticlassNN(
    input_size=784,
    hidden_sizes=[128, 64],
    output_size=10,
    learning_rate=0.1
)
# Treinar
nn_mnist.fit(
    X_train, y_train_oh,
    X_val, y_val_oh,
    epochs=50,
    batch_size=64,
    verbose=True
)
print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO")
print("="*70)
# AVALIAR NO TEST SET
test_acc = nn_mnist.accuracy(X_test, y_test_oh)
y_pred = nn_mnist.predict(X_test)
print(f"\n✅ Acurácia Final no Test Set: {test_acc*100:.2f}%")
# VISUALIZAR CURVAS DE APRENDIZADO
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ───────────────────────────────────────────────────────────────────
# ✅ CHECKPOINT ETAPA 3:
# ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("✅ CHECKPOINT ETAPA 3 - VALIDAÇÕES")
print("="*70)

# Validações
train_final_acc = nn_mnist.train_accs[-1]
val_final_acc = nn_mnist.val_accs[-1]
acc_gap = train_final_acc - val_final_acc

print(f"✓ Treinamento concluído: 50 épocas")
print(f"✓ Train accuracy final: {train_final_acc*100:.2f}%")
print(f"✓ Val accuracy final: {val_final_acc*100:.2f}%")
print(f"✓ Test accuracy: {test_acc*100:.2f}%")
print(f"✓ Gap treino/val: {acc_gap*100:.2f}% {'⚠️ Overfitting!' if acc_gap > 0.15 else '✅ OK'}")
print(f"✓ Loss convergiu: {'✅ Sim' if nn_mnist.train_losses[-1] < 0.5 else '⚠️ Não (loss ainda alto)'}")

# Diagnóstico
if test_acc < 0.85:
    print("\n⚠️ ATENÇÃO: Acurácia abaixo do mínimo (85%)!")
    print("   Sugestões:")
    print("   - Aumentar learning rate (tentar 0.2 ou 0.3)")
    print("   - Treinar mais épocas (100 ao invés de 50)")
    print("   - Verificar normalização dos dados")
elif test_acc >= 0.93:
    print("\n🌟 EXCELENTE! Acurácia acima de 93% (nível avançado)")
elif test_acc >= 0.90:
    print("\n✅ MUITO BOM! Acurácia acima de 90% (nível intermediário)")
else:
    print("\n✅ BOM! Acurácia entre 85-90% (nível mínimo)")

print("\n🎉 Etapa 3 completa! Prossiga para Slide 24 (Etapa 4)")
