# GO0531-Sklearn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt

def evaluate_imbalanced_model(model, X_test, y_test, model_name):
    """
    Avaliação completa para dados desbalanceados
    """
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades classe 1

    # 1. CLASSIFICATION REPORT
    print(f"\n{'='*50}")
    print(f"AVALIAÇÃO: {model_name}")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred, 
                              target_names=['Classe 0', 'Classe 1']))

    # 2. CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"              Pred 0    Pred 1")
    print(f"Real 0:       {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"Real 1:       {cm[1,0]:6d}    {cm[1,1]:6d}")

    # 3. MÉTRICAS ESPECÍFICAS
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMÉTRICAS DA CLASSE MINORITÁRIA:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Negatives (FN): {fn}  ← Positivos que PERDEMOS")
    print(f"  False Positives (FP): {fp}  ← Falsos alarmes")
    print(f"  True Negatives (TN):  {tn}")
    print(f"\n  Precision:  {precision:.3f}")
    print(f"  Recall:     {recall:.3f}")
    print(f"  F1-Score:   {f1:.3f}")

    # 4. ROC-AUC e PR-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n  ROC-AUC:    {roc_auc:.3f}")
    print(f"  PR-AUC:     {pr_auc:.3f}")

    # 5. GRÁFICOS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True)

    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(recall_curve, precision_curve, label=f'PR (AUC={pr_auc:.3f})')
    axes[1].axhline(y=sum(y_test)/len(y_test), color='k', linestyle='--', 
                   label='Baseline')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

# ================================
# COMPARAR TODAS AS TÉCNICAS
# ================================

# Modelo baseline (sem balanceamento)
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
metrics_baseline = evaluate_imbalanced_model(baseline, X_test, y_test, "BASELINE")

# Modelo com SMOTE
model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
metrics_smote = evaluate_imbalanced_model(model_smote, X_test, y_test, "COM SMOTE")

# Modelo com Class Weights
model_weights = RandomForestClassifier(class_weight='balanced', random_state=42)
model_weights.fit(X_train, y_train)
metrics_weights = evaluate_imbalanced_model(model_weights, X_test, y_test, "CLASS WEIGHTS")

# ================================
# TABELA COMPARATIVA
# ================================
import pandas as pd

results = pd.DataFrame({
    'Baseline': metrics_baseline,
    'SMOTE': metrics_smote,
    'Class Weights': metrics_weights
}).T

print("\n" + "="*60)
print("COMPARAÇÃO FINAL DAS TÉCNICAS")
print("="*60)
print(results.round(3))
print("\n🏆 VENCEDOR: Melhor F1-Score na classe minoritária")
