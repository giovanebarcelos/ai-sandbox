# GO0525 - Problema de desbalanceamento de classes (imbalanced dataset)
# Dataset: 100 amostras, 95 classe 0, 5 classe 1
# Modelo que SEMPRE prediz classe 0 tem acuracia de 95% mas e PESSIMO
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
# Dataset desbalanceado: 95 negativos, 5 positivos
X = np.random.randn(100, 2)
y = np.array([0]*95 + [1]*5)

# Modelo ruim: sempre prediz 0
dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
y_pred_dummy = dummy.predict(X)
acuracia_dummy = (y_pred_dummy == y).mean()
print(f"Acuracia modelo sempre-0: {acuracia_dummy*100:.0f}%  <- PARECE BOM!")
print("Mas recall da classe 1:", (y_pred_dummy[y==1]==1).sum(), "de", (y==1).sum())

# Modelo melhor com class_weight
lr = LogisticRegression(class_weight="balanced").fit(X, y)
y_pred_lr = lr.predict(X)
print("\nModelo balanceado:")
print(classification_report(y, y_pred_lr, target_names=["negativo","positivo"]))

# Comparacao visual
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (modelo, pred), titulo in zip(axes,
    [(dummy, y_pred_dummy), (lr, y_pred_lr)],
    ["Sempre prediz 0 (acur=95%)", "Balanceado (class_weight)"]):
    cm = confusion_matrix(y, pred)
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, v, ha="center", va="center", fontsize=14)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title(titulo)
plt.tight_layout()
plt.savefig("GO0525_imbalanced.png", dpi=100, bbox_inches="tight")
plt.show()
print("Salvo: GO0525_imbalanced.png")
