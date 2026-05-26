# GO0516 - Problema 4: ROC AUC nao funciona para multiclasse diretamente
# ERRO COMUM: roc_auc_score sem multi_class='ovr' em problema multiclasse
# SOLUCAO: usar multi_class="ovr" e average="macro"
#
# Mensagem de erro original:
#   ValueError: multiclass format is not supported
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression(max_iter=200).fit(X_train, y_train)
y_prob = model.predict_proba(X_test)

# SOLUCAO: multi_class="ovr"
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
print(f"ROC AUC (ovr, macro): {auc:.4f}")

# Plot curvas ROC por classe
classes = ["setosa", "versicolor", "virginica"]
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(6, 5))
for i, cls in enumerate(classes):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc_score(y_bin[:,i], y_prob[:,i]):.2f})")
ax.plot([0,1],[0,1],"k--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curvas -- Iris Multiclasse")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("GO0516_roc_multiclasse.png", dpi=100, bbox_inches="tight")
plt.show()
print("Salvo: GO0516_roc_multiclasse.png")
