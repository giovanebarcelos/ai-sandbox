# GO2006-10FastapiSalvarERodarModelo
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

iris = load_iris()
X, y = iris.data, iris.target

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo treinado e salvo em 'model.pkl'")
print(f"Acurácia no conjunto de treino: {clf.score(X, y):.4f}")

# Gráfico de importância das features
plt.figure(figsize=(8, 5))
plt.barh(iris.feature_names, clf.feature_importances_, color="steelblue")
plt.xlabel("Importância")
plt.title("Importância das features - RandomForestClassifier (Iris)")
plt.tight_layout()
plt.show()
