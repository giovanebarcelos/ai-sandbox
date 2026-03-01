# GO0502-ÁrvoreDeDecisãoComSklearn
# ═══════════════════════════════════════════════════════════════════
# ÁRVORE DE DECISÃO COM SKLEARN
# ═══════════════════════════════════════════════════════════════════

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ───────────────────────────────────────────────────────────────────
# TREINAR ÁRVORE
# ───────────────────────────────────────────────────────────────────

tree = DecisionTreeClassifier(
    max_depth=3,           # Profundidade máxima
    min_samples_split=5,   # Mínimo de amostras para dividir
    min_samples_leaf=2,    # Mínimo de amostras em folha
    criterion='gini',      # ou 'entropy'
    random_state=42
)

tree.fit(X_train, y_train)

# ───────────────────────────────────────────────────────────────────
# AVALIAR
# ───────────────────────────────────────────────────────────────────

train_score = tree.score(X_train, y_train)
test_score = tree.score(X_test, y_test)

print("="*60)
print("DECISION TREE - RESULTADOS")
print("="*60)
print(f"Acurácia Treino: {train_score:.3f}")
print(f"Acurácia Teste:  {test_score:.3f}")
print(f"Profundidade: {tree.get_depth()}")
print(f"Número de folhas: {tree.get_n_leaves()}")

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR ÁRVORE
# ───────────────────────────────────────────────────────────────────

plt.figure(figsize=(20, 10))
plot_tree(tree, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Árvore de Decisão - Dataset Iris")
plt.show()

# ───────────────────────────────────────────────────────────────────
# IMPORTÂNCIA DAS FEATURES
# ───────────────────────────────────────────────────────────────────

importances = tree.feature_importances_
for name, importance in zip(iris.feature_names, importances):
    print(f"{name:20s}: {importance:.3f}")
