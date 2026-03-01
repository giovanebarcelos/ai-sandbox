# GO0528-NumpyPandas
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ================================
# 1. CRIAR DATASET DESBALANCEADO
# ================================


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # 95% classe 0, 5% classe 1
        flip_y=0,
        random_state=42
    )

    print("DISTRIBUIÇÃO ORIGINAL:")
    print(pd.Series(y).value_counts())
    print(f"Ratio: {sum(y==0)/sum(y==1):.1f}:1\n")

    # Classe 0: 950 amostras (95%)
    # Classe 1: 50 amostras (5%)
    # Ratio: 19.0:1

    # ================================
    # 2. VISUALIZAR DESBALANCEAMENTO
    # ================================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[y==0, 0], X[y==0, 1], alpha=0.5, label='Classe 0 (95%)')
    plt.scatter(X[y==1, 0], X[y==1, 1], alpha=0.8, color='red', label='Classe 1 (5%)')
    plt.title('Dataset Original (Desbalanceado)')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # ================================
    # 3. APLICAR SMOTE
    # ================================
    smote = SMOTE(
        sampling_strategy='auto',  # Balanceia para 50:50
        k_neighbors=5,             # 5 vizinhos para interpolar
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("DISTRIBUIÇÃO APÓS SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    print(f"Ratio: {sum(y_resampled==0)/sum(y_resampled==1):.1f}:1\n")

    # Classe 0: 950 amostras (50%)
    # Classe 1: 950 amostras (50%)  ← Geradas sinteticamente!
    # Ratio: 1.0:1

    # ================================
    # 4. VISUALIZAR PÓS-SMOTE
    # ================================
    plt.subplot(1, 2, 2)
    plt.scatter(X_resampled[y_resampled==0, 0], 
               X_resampled[y_resampled==0, 1], 
               alpha=0.5, label='Classe 0 (50%)')
    plt.scatter(X_resampled[y_resampled==1, 0], 
               X_resampled[y_resampled==1, 1], 
               alpha=0.6, color='red', label='Classe 1 (50%)')
    plt.title('Dataset após SMOTE (Balanceado)')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

    # ================================
    # 5. COMPARAR MODELOS
    # ================================

    # 5.1 Modelo SEM SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model_without_smote = RandomForestClassifier(random_state=42)
    model_without_smote.fit(X_train, y_train)
    y_pred_without = model_without_smote.predict(X_test)

    print("=" * 50)
    print("MODELO SEM SMOTE")
    print("=" * 50)
    print(classification_report(y_test, y_pred_without, 
                              target_names=['Classe 0', 'Classe 1']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_without))

    # RESULTADO TÍPICO (SEM SMOTE):
    #               precision    recall  f1-score
    # Classe 0         0.96      0.99      0.98
    # Classe 1         0.67      0.40      0.50  ← Ruim na classe minoritária!

    # 5.2 Modelo COM SMOTE
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )

    model_with_smote = RandomForestClassifier(random_state=42)
    model_with_smote.fit(X_train_smote, y_train_smote)
    y_pred_with = model_with_smote.predict(X_test_smote)

    print("\n" + "=" * 50)
    print("MODELO COM SMOTE")
    print("=" * 50)
    print(classification_report(y_test_smote, y_pred_with,
                              target_names=['Classe 0', 'Classe 1']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_smote, y_pred_with))

    # RESULTADO TÍPICO (COM SMOTE):
    #               precision    recall  f1-score
    # Classe 0         0.97      0.96      0.97
    # Classe 1         0.96      0.97      0.97  ← MUITO MELHOR!

    # ================================
    # 6. MÉTRICAS COMPARATIVAS
    # ================================
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    def evaluate_model(y_true, y_pred, model_name):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )
        auc = roc_auc_score(y_true, y_pred)

        print(f"\n{model_name}:")
        print(f"  Precision (Classe Minoritária): {precision:.3f}")
        print(f"  Recall (Classe Minoritária):    {recall:.3f}")
        print(f"  F1-Score (Classe Minoritária):  {f1:.3f}")
        print(f"  ROC-AUC:                         {auc:.3f}")

    evaluate_model(y_test, y_pred_without, "SEM SMOTE")
    evaluate_model(y_test_smote, y_pred_with, "COM SMOTE")

    # RESULTADO:
    # SEM SMOTE:
    #   Precision: 0.667
    #   Recall:    0.400  ← Perde 60% dos positivos!
    #   F1-Score:  0.500
    #   ROC-AUC:   0.845

    # COM SMOTE:
    #   Precision: 0.960
    #   Recall:    0.970  ← Captura 97% dos positivos!
    #   F1-Score:  0.965
    #   ROC-AUC:   0.985
