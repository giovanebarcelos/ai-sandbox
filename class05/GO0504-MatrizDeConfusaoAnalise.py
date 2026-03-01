# GO0504-MatrizDeConfusãoAnálise
# ═══════════════════════════════════════════════════════════════════
# MATRIZ DE CONFUSÃO - ANÁLISE COMPLETA
# ═══════════════════════════════════════════════════════════════════

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Supondo que temos y_test e y_pred de algum modelo

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR MATRIZ DE CONFUSÃO
# ───────────────────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

# Método 1: Sklearn
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=iris.target_names)
disp.plot(ax=ax1, cmap='Blues', values_format='d')
ax1.set_title('Matriz de Confusão - Sklearn')

# Método 2: Seaborn (mais customizável)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax2)
ax2.set_xlabel('Predito')
ax2.set_ylabel('Verdadeiro')
ax2.set_title('Matriz de Confusão - Seaborn')

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# RELATÓRIO DE CLASSIFICAÇÃO
# ───────────────────────────────────────────────────────────────────

print("="*60)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("="*60)
print(classification_report(y_test, y_pred, 
                            target_names=iris.target_names))

# ───────────────────────────────────────────────────────────────────
# ANÁLISE MANUAL DA MATRIZ
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ANÁLISE POR CLASSE")
print("="*60)

for i, class_name in enumerate(iris.target_names):
    # Extrair TP, FP, FN, TN para classe i
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{class_name}:")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
