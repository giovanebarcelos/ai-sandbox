# GO1014-Codigo
# Avalia o modelo treinado no conjunto de teste e exibe a acurácia final,
# representando o desempenho real esperado em dados nunca vistos.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎉 Test Accuracy: {test_acc*100:.2f}%")
# 🎉 Test Accuracy: 99.23%
