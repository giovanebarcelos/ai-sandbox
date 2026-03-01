# GO0501-KnearestNeighborsDoZero
# ═══════════════════════════════════════════════════════════════════
# K-NEAREST NEIGHBORS - IMPLEMENTAÇÃO DO ZERO
# ═══════════════════════════════════════════════════════════════════

import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors Classifier implementado do zero
    """
    def __init__(self, k=3):
        """
        Parâmetros:
            k: número de vizinhos a considerar
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        "Treinar" KNN = apenas armazenar dados de treino
        KNN é lazy learning: não aprende nada durante fit!
        """
        self.X_train = X
        self.y_train = y
        return self

    def euclidean_distance(self, x1, x2):
        """
        Distância Euclidiana entre dois vetores
        """
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict_single(self, x):
        """
        Prediz a classe para uma única amostra
        """
        # 1. Calcular distâncias para todos os pontos de treino
        distances = [self.euclidean_distance(x, x_train) 
                    for x_train in self.X_train]

        # 2. Obter índices dos k vizinhos mais próximos
        k_indices = np.argsort(distances)[:self.k]

        # 3. Obter classes dos k vizinhos
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Votação majoritária
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Prediz classes para múltiplas amostras
        """
        return np.array([self.predict_single(x) for x in X])


# ═══════════════════════════════════════════════════════════════════
# EXEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ⚠️ IMPORTANTE: Normalizar!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar KNN
knn = KNN(k=5)
knn.fit(X_train_scaled, y_train)

# Prever
y_pred = knn.predict(X_test_scaled)

# Avaliar
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# USANDO SCIKIT-LEARN (Produção)
# ═══════════════════════════════════════════════════════════════════

from sklearn.neighbors import KNeighborsClassifier

# Treinar
knn_sklearn = KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(X_train_scaled, y_train)

# Prever
y_pred_sklearn = knn_sklearn.predict(X_test_scaled)

# Avaliar
print("\n" + "="*60)
print("SKLEARN KNN")
print("="*60)
print(classification_report(y_test, y_pred_sklearn, 
                           target_names=iris.target_names))

# ═══════════════════════════════════════════════════════════════════
# ENCONTRAR MELHOR K
# ═══════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

k_values = range(1, 31)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plotar
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linewidth=2)
plt.xlabel('Número de Vizinhos (k)', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.title('Acurácia vs K em Iris Dataset', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axvline(x=k_values[np.argmax(accuracies)], 
            color='r', linestyle='--', 
            label=f'Melhor k={k_values[np.argmax(accuracies)]}')
plt.legend()
plt.show()

print(f"\nMelhor k: {k_values[np.argmax(accuracies)]}")
print(f"Acurácia máxima: {max(accuracies)*100:.2f}%")
