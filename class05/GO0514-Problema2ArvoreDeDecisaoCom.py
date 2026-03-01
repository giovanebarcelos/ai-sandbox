# GO0514-Problema2ÁrvoreDeDecisãoCom
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ───────────────────────────────────────────────────────────────────
# CRIAR DATASET PARA DEMONSTRAR OVERFITTING
# ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("="*60)
    print("TESTANDO SOLUÇÕES PARA OVERFITTING EM ÁRVORES DE DECISÃO")
    print("="*60)

    # Dataset com ruído para provocar overfitting
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=2,
                              random_state=42, flip_y=0.1)  # 10% ruído

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\nTamanho dos dados:")
    print(f"  Treino: {X_train.shape}")
    print(f"  Teste: {X_test.shape}")

    # ───────────────────────────────────────────────────────────────────
    # PROBLEMA: ÁRVORE SEM RESTRIÇÕES (OVERFIT)
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("❌ PROBLEMA: Árvore SEM restrições (overfit)")
    print("-"*60)

    model_overfit = DecisionTreeClassifier(random_state=42)
    model_overfit.fit(X_train, y_train)

    acc_train_overfit = accuracy_score(y_train, model_overfit.predict(X_train))
    acc_test_overfit = accuracy_score(y_test, model_overfit.predict(X_test))

    print(f"Acurácia Treino: {acc_train_overfit:.3f}")
    print(f"Acurácia Teste:  {acc_test_overfit:.3f}")
    print(f"Diferença:       {(acc_train_overfit - acc_test_overfit):.3f}")
    print(f"Profundidade:    {model_overfit.tree_.max_depth}")
    print(f"Nº de folhas:    {model_overfit.tree_.n_leaves}")
    print("\n⚠️  Grande diferença indica OVERFITTING!")

    # ───────────────────────────────────────────────────────────────────
    # SOLUÇÃO 1: ÁRVORE COM REGULARIZAÇÃO
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("✅ SOLUÇÃO 1: Árvore COM regularização")
    print("-"*60)

    model_regularized = DecisionTreeClassifier(
        max_depth=10,              # Limitar profundidade máxima
        min_samples_split=20,      # Mínimo para dividir nó
        min_samples_leaf=10,       # Mínimo por folha
        max_features='sqrt',       # Features aleatórias (+ generalização)
        random_state=42
    )
    model_regularized.fit(X_train, y_train)

    acc_train_reg = accuracy_score(y_train, model_regularized.predict(X_train))
    acc_test_reg = accuracy_score(y_test, model_regularized.predict(X_test))

    print(f"Acurácia Treino: {acc_train_reg:.3f}")
    print(f"Acurácia Teste:  {acc_test_reg:.3f}")
    print(f"Diferença:       {(acc_train_reg - acc_test_reg):.3f}")
    print(f"Profundidade:    {model_regularized.tree_.max_depth}")
    print(f"Nº de folhas:    {model_regularized.tree_.n_leaves}")
    print("\n✅ Diferença menor = melhor generalização!")

    # ───────────────────────────────────────────────────────────────────
    # SOLUÇÃO 2: PODA (COST-COMPLEXITY PRUNING)
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("✅ SOLUÇÃO 2: Poda (Cost-Complexity Pruning)")
    print("-"*60)

    model_pruned = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
    model_pruned.fit(X_train, y_train)

    acc_train_pruned = accuracy_score(y_train, model_pruned.predict(X_train))
    acc_test_pruned = accuracy_score(y_test, model_pruned.predict(X_test))

    print(f"Acurácia Treino: {acc_train_pruned:.3f}")
    print(f"Acurácia Teste:  {acc_test_pruned:.3f}")
    print(f"Diferença:       {(acc_train_pruned - acc_test_pruned):.3f}")
    print(f"Profundidade:    {model_pruned.tree_.max_depth}")
    print(f"Nº de folhas:    {model_pruned.tree_.n_leaves}")

    # ───────────────────────────────────────────────────────────────────
    # RESUMO COMPARATIVO
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    print(f"{'Método':<25} {'Acc Train':<12} {'Acc Test':<12} {'Dif':<8} {'Prof':<8} {'Folhas':<8}")
    print("-"*60)
    print(f"{'Sem restrições':<25} {acc_train_overfit:<12.3f} {acc_test_overfit:<12.3f} "
          f"{(acc_train_overfit - acc_test_overfit):<8.3f} "
          f"{model_overfit.tree_.max_depth:<8} {model_overfit.tree_.n_leaves:<8}")
    print(f"{'Com regularização':<25} {acc_train_reg:<12.3f} {acc_test_reg:<12.3f} "
          f"{(acc_train_reg - acc_test_reg):<8.3f} "
          f"{model_regularized.tree_.max_depth:<8} {model_regularized.tree_.n_leaves:<8}")
    print(f"{'Com poda (pruning)':<25} {acc_train_pruned:<12.3f} {acc_test_pruned:<12.3f} "
          f"{(acc_train_pruned - acc_test_pruned):<8.3f} "
          f"{model_pruned.tree_.max_depth:<8} {model_pruned.tree_.n_leaves:<8}")
    print("="*60)
    print("\n💡 Observe:")
    print("  - Sem restrições: Alta acurácia treino, baixa teste (OVERFIT)")
    print("  - Com regularização: Acurácias mais equilibradas")
    print("  - Árvores menores (menos profundidade/folhas) = melhor generalização")
