# GO0507-MétricasParaProblemasMulticlasse
# ═══════════════════════════════════════════════════════════════════
# MÉTRICAS PARA PROBLEMAS MULTICLASSE
# ═══════════════════════════════════════════════════════════════════

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              cohen_kappa_score, matthews_corrcoef,
                              balanced_accuracy_score)

# ───────────────────────────────────────────────────────────────────
# PREPARAR DADOS E MODELO
# ───────────────────────────────────────────────────────────────────

# Carregar dados Iris (3 classes)


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinar modelo
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Fazer predições
    y_pred = model.predict(X_test)

    # ───────────────────────────────────────────────────────────────────
    # DIFERENTES ESTRATÉGIAS DE AGREGAÇÃO
    # ───────────────────────────────────────────────────────────────────

    print("="*60)
    print("MÉTRICAS MULTICLASSE - DIFERENTES AGREGAÇÕES")
    print("="*60)

    # Macro: Média simples entre classes (trata todas iguais)
    print("\nMACRO (média simples):")
    print(f"  Precision: {precision_score(y_test, y_pred, average='macro'):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, average='macro'):.3f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred, average='macro'):.3f}")

    # Weighted: Média ponderada pelo número de amostras
    print("\nWEIGHTED (média ponderada por classe):")
    print(f"  Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.3f}")

    # Micro: Agregação global (TP, FP, FN somados)
    print("\nMICRO (agregação global):")
    print(f"  Precision: {precision_score(y_test, y_pred, average='micro'):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, average='micro'):.3f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred, average='micro'):.3f}")

    # ───────────────────────────────────────────────────────────────────
    # MÉTRICAS ADICIONAIS
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("OUTRAS MÉTRICAS ÚTEIS")
    print("="*60)

    # Balanced Accuracy (útil para classes desbalanceadas)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print("  (Média do recall de cada classe)")

    # Cohen's Kappa (concordância acima do acaso)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"\nCohen's Kappa: {kappa:.3f}")
    print("  < 0:    Sem concordância")
    print("  0-0.2:  Concordância leve")
    print("  0.2-0.4: Concordância fraca")
    print("  0.4-0.6: Concordância moderada")
    print("  0.6-0.8: Concordância substancial")
    print("  0.8-1:   Concordância quase perfeita")

    # Matthews Correlation Coefficient (melhor que acurácia para desbalanceamento)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"\nMatthews Correlation Coefficient: {mcc:.3f}")
    print("  -1: Total disagreement")
    print("   0: Random prediction")
    print("  +1: Perfect prediction")
