# GO0513-Problema1KnnMuitoLentoCom
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ───────────────────────────────────────────────────────────────────
# CRIAR DATASET GRANDE PARA DEMONSTRAR O PROBLEMA
# ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("="*60)
    print("TESTANDO SOLUÇÕES PARA KNN LENTO")
    print("="*60)

    # Dataset grande: 10000 amostras
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=3,
                              n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTamanho dos dados:")
    print(f"  Treino: {X_train.shape}")
    print(f"  Teste: {X_test.shape}")

    # ───────────────────────────────────────────────────────────────────
    # OPÇÃO 1: REDUZIR K
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("OPÇÃO 1: Reduzir k (de 10 para 3)")
    print("-"*60)

    start = time.time()
    model_k3 = KNeighborsClassifier(n_neighbors=3)
    model_k3.fit(X_train, y_train)
    y_pred_k3 = model_k3.predict(X_test)
    time_k3 = time.time() - start

    acc_k3 = accuracy_score(y_test, y_pred_k3)
    print(f"Tempo: {time_k3:.2f}s")
    print(f"Acurácia: {acc_k3:.3f}")

    # ───────────────────────────────────────────────────────────────────
    # OPÇÃO 2: USAR BALL TREE OU KD TREE
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("OPÇÃO 2: Usar Ball Tree (mais rápido)")
    print("-"*60)

    start = time.time()
    model_ball = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    model_ball.fit(X_train, y_train)
    y_pred_ball = model_ball.predict(X_test)
    time_ball = time.time() - start

    acc_ball = accuracy_score(y_test, y_pred_ball)
    print(f"Tempo: {time_ball:.2f}s")
    print(f"Acurácia: {acc_ball:.3f}")

    # ───────────────────────────────────────────────────────────────────
    # OPÇÃO 3: AMOSTRAR DADOS DE TREINO
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("OPÇÃO 3: Usar apenas 30% dos dados de treino")
    print("-"*60)

    start = time.time()
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=0.3, random_state=42
    )
    model_sample = KNeighborsClassifier(n_neighbors=5)
    model_sample.fit(X_sample, y_sample)
    y_pred_sample = model_sample.predict(X_test)
    time_sample = time.time() - start

    acc_sample = accuracy_score(y_test, y_pred_sample)
    print(f"Tempo: {time_sample:.2f}s")
    print(f"Acurácia: {acc_sample:.3f}")
    print(f"Amostras usadas: {X_sample.shape[0]} (de {X_train.shape[0]})")

    # ───────────────────────────────────────────────────────────────────
    # OPÇÃO 4: USAR ALGORITMO ALTERNATIVO (RANDOM FOREST)
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "-"*60)
    print("OPÇÃO 4: Usar Random Forest (alternativa)")
    print("-"*60)

    start = time.time()
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    time_rf = time.time() - start

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Tempo: {time_rf:.2f}s")
    print(f"Acurácia: {acc_rf:.3f}")

    # ───────────────────────────────────────────────────────────────────
    # RESUMO COMPARATIVO
    # ───────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    print(f"{'Método':<30} {'Tempo (s)':<12} {'Acurácia':<10}")
    print("-"*60)
    print(f"{'KNN k=3':<30} {time_k3:<12.2f} {acc_k3:<10.3f}")
    print(f"{'KNN Ball Tree':<30} {time_ball:<12.2f} {acc_ball:<10.3f}")
    print(f"{'KNN com 30% dados':<30} {time_sample:<12.2f} {acc_sample:<10.3f}")
    print(f"{'Random Forest':<30} {time_rf:<12.2f} {acc_rf:<10.3f}")
    print("="*60)
