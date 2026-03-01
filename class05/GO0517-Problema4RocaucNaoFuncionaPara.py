# GO0517-Problema4RocaucNãoFuncionaPara
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 80)
    print("TESTE: ROC-AUC Score - Binário vs Multiclasse")
    print("=" * 80)

    # ========================================================================
    # CASO 1: PROBLEMA BINÁRIO (2 classes)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CASO 1: Classificação BINÁRIA (Breast Cancer - 2 classes)")
    print("=" * 80)

    # Carregar dataset binário
    cancer = load_breast_cancer()
    X_bin = cancer.data
    y_bin = cancer.target

    print(f"\n📊 Dataset: {len(y_bin)} amostras, {X_bin.shape[1]} features")
    print(f"Classes: {np.unique(y_bin)} - {cancer.target_names}")

    # Dividir dados
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.3, random_state=42
    )

    # Treinar modelo
    model_bin = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_bin.fit(X_train_bin, y_train_bin)

    print("\n✅ Modelo treinado!")

    # ✅ CORRETO: Para BINÁRIO, usar predict_proba e pegar coluna da classe positiva
    y_proba_bin = model_bin.predict_proba(X_test_bin)[:, 1]
    roc_auc_bin = roc_auc_score(y_test_bin, y_proba_bin)

    print(f"\n📈 ROC-AUC Score (binário): {roc_auc_bin:.4f}")

    # Calcular curva ROC
    fpr_bin, tpr_bin, thresholds_bin = roc_curve(y_test_bin, y_proba_bin)

    # ========================================================================
    # CASO 2: PROBLEMA MULTICLASSE (3+ classes)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CASO 2: Classificação MULTICLASSE (Iris - 3 classes)")
    print("=" * 80)

    # Carregar dataset multiclasse
    iris = load_iris()
    X_multi = iris.data
    y_multi = iris.target

    print(f"\n📊 Dataset: {len(y_multi)} amostras, {X_multi.shape[1]} features")
    print(f"Classes: {np.unique(y_multi)} - {iris.target_names}")

    # Dividir dados
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42
    )

    # Treinar modelo
    model_multi = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_multi.fit(X_train_multi, y_train_multi)

    print("\n✅ Modelo treinado!")

    # ✅ CORRETO: Para MULTICLASSE, usar predict_proba com multi_class='ovr'
    y_proba_multi = model_multi.predict_proba(X_test_multi)
    roc_auc_multi = roc_auc_score(y_test_multi, y_proba_multi, multi_class='ovr')

    print(f"\n📈 ROC-AUC Score (multiclasse OvR): {roc_auc_multi:.4f}")

    # ========================================================================
    # VISUALIZAÇÃO: Comparar curvas ROC
    # ========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZAÇÃO: Curvas ROC")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Curva ROC Binária
    axes[0].plot(fpr_bin, tpr_bin, linewidth=2, label=f'ROC (AUC = {roc_auc_bin:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('ROC Curve - BINÁRIO (Breast Cancer)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot 2: Mostrar diferença entre abordagens multiclasse
    # Calcular ROC para cada classe (OvR)
    from sklearn.preprocessing import label_binarize
    y_test_bin_multi = label_binarize(y_test_multi, classes=[0, 1, 2])

    for i in range(3):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin_multi[:, i], y_proba_multi[:, i])
        auc_i = roc_auc_score(y_test_bin_multi[:, i], y_proba_multi[:, i])
        axes[1].plot(fpr_i, tpr_i, linewidth=2, label=f'{iris.target_names[i]} (AUC = {auc_i:.3f})')

    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title('ROC Curves - MULTICLASSE (Iris OvR)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # RESUMO: Diferenças entre Binário e Multiclasse
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 RESUMO: Como usar ROC-AUC corretamente")
    print("=" * 80)

    print("\n✅ Para BINÁRIO (2 classes):")
    print("   - y_proba = model.predict_proba(X_test)[:, 1]  # Coluna da classe positiva")
    print("   - roc_auc = roc_auc_score(y_test, y_proba)")
    print("   - fpr, tpr, _ = roc_curve(y_test, y_proba)")

    print("\n✅ Para MULTICLASSE (3+ classes):")
    print("   - y_proba = model.predict_proba(X_test)  # Todas as colunas")
    print("   - roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')")
    print("   - Estratégias: 'ovr' (One-vs-Rest) ou 'ovo' (One-vs-One)")

    print("\n❌ ERRO COMUM:")
    print("   - NotFittedError: model.predict_proba() antes de model.fit()")
    print("   - Solução: Sempre treinar o modelo primeiro!")

    print("\n" + "=" * 80)
    print("Teste concluído com sucesso! ✅")
    print("=" * 80)

if __name__ == "__main__":
    main()
