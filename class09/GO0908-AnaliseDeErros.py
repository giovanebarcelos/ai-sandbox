# GO0908-AnáliseDeErros
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: RECONHECIMENTO DE DÍGITOS (MNIST)
# Etapa 4: ANÁLISE DE ERROS E INSIGHTS
# ═══════════════════════════════════════════════════════════════════
ca')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Curvas de Acurácia')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.5, 1.0])
plt.tight_layout()
plt.show()
# MATRIZ DE CONFUSÃO
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title(f'Matriz de Confusão - MNIST (Acc: {test_acc*100:.2f}%)')
plt.tight_layout()
plt.show()
# Classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# ANÁLISE DE ERROS
print("\n" + "="*70)
print("ANÁLISE DE ERROS")
print("="*70)
# Encontrar predições erradas
errors = y_test != y_pred
error_indices = np.where(errors)[0]
print(f"\nTotal de erros: {errors.sum()} / {len(y_test)} ({errors.sum()/len(y_test)*100:.2f}%)")
# VISUALIZAR ERROS MAIS CONFIANTES
# Calcular probabilidades
probs = nn_mnist.forward(X_test)
pred_confidences = np.max(probs, axis=1)
# Erros com maior confiança (rede estava muito confiante mas errou!)
error_confidences = pred_confidences[error_indices]
most_confident_errors_idx = error_indices[np.argsort(error_confidences)[-20:]]
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.ravel()
for i, idx in enumerate(most_confident_errors_idx):
    # Imagem original
    img = scaler.inverse_transform(X_test[idx].reshape(1, -1)).reshape(28, 28)
    # Informações
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = pred_confidences[idx]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}',
                     fontsize=9)
    axes[i].axis('off')
plt.suptitle('Top 20 Erros Mais Confiantes', fontsize=14)
plt.tight_layout()
plt.show()
# ANÁLISE POR DÍGITO
print("\n" + "="*70)
print("ACURÁCIA POR DÍGITO")
print("="*70)
for digit in range(10):
    digit_mask = (y_test == digit)
    digit_acc = np.mean(y_pred[digit_mask] == y_test[digit_mask])
    digit_count = digit_mask.sum()
    print(f"Dígito {digit}: {digit_acc*100:5.2f}% ({digit_count} amostras)")
# CONFUSÕES MAIS COMUNS
print("\n" + "="*70)
print("TOP 10 CONFUSÕES MAIS COMUNS")
print("="*70)
confusion_counts = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_counts.append((cm[i, j], i, j))
confusion_counts.sort(reverse=True)
print(f"\n{'Count':>5} | {'True':>4} | {'Pred':>4} | Confusão")
print("-" * 40)
for count, true_digit, pred_digit in confusion_counts[:10]:
    print(f"{count:5d} | {true_digit:4d} | {pred_digit:4d} | {true_digit} confundido com {pred_digit}")
print("\n✅ Análise de erros completa!")

# ───────────────────────────────────────────────────────────────────
# ✅ CHECKPOINT FINAL - RESUMO DA ATIVIDADE:
# ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("✅ CHECKPOINT FINAL - RESUMO DA ATIVIDADE")
print("="*70)

print(f"\n📊 RESULTADOS FINAIS:")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Total de erros: {errors.sum()} / {len(y_test)} ({errors.sum()/len(y_test)*100:.1f}%)")

# Encontrar dígitos mais fáceis e difíceis
digit_accs = [(d, np.mean(y_pred[y_test==d] == y_test[y_test==d])*100) for d in range(10) if (y_test==d).sum() > 0]
digit_accs_sorted = sorted(digit_accs, key=lambda x: x[1])
easiest = digit_accs_sorted[-1]
hardest = digit_accs_sorted[0]
print(f"   Dígito mais fácil: {easiest[0]} ({easiest[1]:.1f}% acc)")
print(f"   Dígito mais difícil: {hardest[0]} ({hardest[1]:.1f}% acc)")

print(f"\n🎯 NÍVEL ALCANÇADO:")
if test_acc >= 0.93:
    print("   🌟 AVANÇADO (≥93%) - Excelente desempenho!")
    print("   Sugestões de próximos passos:")
    print("   - Implementar desafios opcionais (Batch Norm, Early Stopping)")
    print("   - Comparar com Keras/TensorFlow")
    print("   - Testar em dataset completo (70k amostras)")
elif test_acc >= 0.90:
    print("   ✅ INTERMEDIÁRIO (≥90%) - Muito bom!")
    print("   Sugestões de melhoria:")
    print("   - Experimentar learning rate maior (0.2)")
    print("   - Testar arquiteturas diferentes (mais neurônios)")
elif test_acc >= 0.85:
    print("   ✅ MÍNIMO (≥85%) - Objetivo alcançado!")
    print("   Sugestões de melhoria:")
    print("   - Treinar mais épocas (100)")
    print("   - Aumentar complexidade do modelo ([256, 128] hidden)")
else:
    print("   ⚠️ ABAIXO DO MÍNIMO (<85%) - Revisar implementação")
    print("   Verifique:")
    print("   - Normalização dos dados (StandardScaler aplicado?)")
    print("   - Learning rate adequado (tentar 0.1)")
    print("   - He initialization implementada corretamente")

print(f"\n📚 CONCEITOS APLICADOS:")
print(f"   ✓ Forward propagation com múltiplas camadas")
print(f"   ✓ Backpropagation com chain rule")
print(f"   ✓ Mini-batch gradient descent")
print(f"   ✓ ReLU activation (hidden layers)")
print(f"   ✓ Softmax activation (output layer)")
print(f"   ✓ Categorical cross-entropy loss")
print(f"   ✓ He initialization para estabilidade")
print(f"   ✓ Análise de overfitting (treino vs validação)")

print(f"\n🎓 PRÓXIMOS PASSOS:")
print(f"   1. Revisar Slide 25 (Regularização) para evitar overfitting")
print(f"   2. Explorar Slide 26 (Inicialização de Pesos) teoria")
print(f"   3. Avançar para Aula 10 (MLP com Keras/TensorFlow)")

print("\n🎉🎉🎉 ATIVIDADE COMPLETA! PARABÉNS! 🎉🎉🎉")
