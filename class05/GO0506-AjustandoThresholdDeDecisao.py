# GO0506-AjustandoThresholdDeDecisão
# ═══════════════════════════════════════════════════════════════════
# AJUSTANDO THRESHOLD DE DECISÃO
# ═══════════════════════════════════════════════════════════════════

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Obter probabilidades
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade classe positiva

# ───────────────────────────────────────────────────────────────────
# TESTAR VÁRIOS THRESHOLDS
# ───────────────────────────────────────────────────────────────────

thresholds_to_test = np.arange(0.1, 1.0, 0.05)

results = []
for threshold in thresholds_to_test:
    y_pred_threshold = (y_proba >= threshold).astype(int)

    precision = precision_score(y_binary, y_pred_threshold, zero_division=0)
    recall = recall_score(y_binary, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_binary, y_pred_threshold, zero_division=0)

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# ───────────────────────────────────────────────────────────────────
# PLOTAR RESULTADOS
# ───────────────────────────────────────────────────────────────────

import pandas as pd

df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
plt.plot(df['threshold'], df['precision'], 'o-', label='Precision', linewidth=2)
plt.plot(df['threshold'], df['recall'], 's-', label='Recall', linewidth=2)
plt.plot(df['threshold'], df['f1'], '^-', label='F1-Score', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall e F1 vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ───────────────────────────────────────────────────────────────────
# ENCONTRAR MELHORES THRESHOLDS
# ───────────────────────────────────────────────────────────────────

print("="*60)
print("MELHORES THRESHOLDS")
print("="*60)

best_f1_idx = df['f1'].idxmax()
print(f"\nMaior F1-Score:")
print(f"  Threshold: {df.loc[best_f1_idx, 'threshold']:.2f}")
print(f"  F1: {df.loc[best_f1_idx, 'f1']:.3f}")
print(f"  Precision: {df.loc[best_f1_idx, 'precision']:.3f}")
print(f"  Recall: {df.loc[best_f1_idx, 'recall']:.3f}")

best_precision_idx = df['precision'].idxmax()
print(f"\nMaior Precision:")
print(f"  Threshold: {df.loc[best_precision_idx, 'threshold']:.2f}")
print(f"  Precision: {df.loc[best_precision_idx, 'precision']:.3f}")

best_recall_idx = df['recall'].idxmax()
print(f"\nMaior Recall:")
print(f"  Threshold: {df.loc[best_recall_idx, 'threshold']:.2f}")
print(f"  Recall: {df.loc[best_recall_idx, 'recall']:.3f}")
