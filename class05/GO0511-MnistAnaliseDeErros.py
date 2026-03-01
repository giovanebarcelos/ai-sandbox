# GO0511-MnistAnáliseDeErros
# ═══════════════════════════════════════════════════════════════════
# MNIST - ANÁLISE DE ERROS
# ═══════════════════════════════════════════════════════════════════

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Vamos analisar o melhor modelo (provavelmente KNN)
y_pred = y_pred_knn  # ou y_pred_tree, y_pred_nb

print("="*60)
print("ANÁLISE DE ERROS")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# MATRIZ DE CONFUSÃO
# ───────────────────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Dígito Predito')
plt.ylabel('Dígito Verdadeiro')
plt.title('Matriz de Confusão - MNIST')
plt.show()

# ───────────────────────────────────────────────────────────────────
# RELATÓRIO DE CLASSIFICAÇÃO
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RELATÓRIO DETALHADO POR CLASSE")
print("="*60)
print(classification_report(y_test, y_pred))

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR ERROS
# ───────────────────────────────────────────────────────────────────

# Encontrar exemplos classificados incorretamente
errors_idx = np.where(y_pred != y_test)[0]

print(f"\nTotal de erros: {len(errors_idx)} de {len(y_test)} " +
      f"({len(errors_idx)/len(y_test)*100:.1f}%)")

# Plotar alguns erros
if len(errors_idx) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(errors_idx):
            idx = errors_idx[i]
            img = X_test.iloc[idx].values.reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Real: {y_test.iloc[idx]}\n' +
                         f'Pred: {y_pred[idx]}',
                         color='red')
            ax.axis('off')
    plt.suptitle('Exemplos de Classificações Incorretas', color='red')
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────────────────────────────────
# CONFUSÕES MAIS COMUNS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PARES MAIS CONFUNDIDOS")
print("="*60)

# Extrair confusões (fora da diagonal)
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0)

# Top 5 confusões
top_confusions = []
for i in range(10):
    for j in range(10):
        if cm_errors[i, j] > 0:
            top_confusions.append((i, j, cm_errors[i, j]))

top_confusions.sort(key=lambda x: x[2], reverse=True)

print("\nTop 10 confusões:")
for real, pred, count in top_confusions[:10]:
    print(f"  {real} confundido com {pred}: {count} vezes")
