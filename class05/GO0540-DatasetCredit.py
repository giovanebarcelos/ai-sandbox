# GO0540-DatasetCredit
# Dataset real: Credit Card Fraud Detection (Kaggle)
# 284,807 transações, 492 fraudes (0.17%)

# TAREFA:
# 1. Carregar dataset de fraude
# 2. Implementar 3 abordagens:
#    - Baseline (sem balanceamento)
#    - SMOTE
#    - Class Weights
# 3. Comparar:
#    - Precision, Recall, F1
#    - ROC-AUC, PR-AUC
#    - Confusion Matrix
# 4. Escolher melhor modelo
# 5. Justificar escolha (custo de FP vs FN)

# PERGUNTA CRÍTICA:
# "O que é pior neste contexto?"
# - False Positive (bloquear transação legítima)?
# - False Negative (deixar fraude passar)?
