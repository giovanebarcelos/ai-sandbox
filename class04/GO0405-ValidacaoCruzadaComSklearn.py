# GO0405-ValidaçãoCruzadaComSklearn
# ═══════════════════════════════════════════════════════════════════
# VALIDAÇÃO CRUZADA COM SKLEARN
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target

# Criar modelo
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# ───────────────────────────────────────────────────────────────────
# CROSS-VALIDATION SIMPLES
# ───────────────────────────────────────────────────────────────────

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("="*60)
print("VALIDAÇÃO CRUZADA (5-Fold)")
print("="*60)
print(f"Scores por fold: {scores}")
print(f"Média: {scores.mean():.3f}")
print(f"Desvio padrão: {scores.std():.3f}")
print(f"Intervalo de confiança 95%: {scores.mean():.3f} ± {1.96*scores.std():.3f}")

# ───────────────────────────────────────────────────────────────────
# STRATIFIED K-FOLD (mantém proporção de classes)
# ───────────────────────────────────────────────────────────────────

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\n" + "="*60)
print("STRATIFIED K-FOLD")
print("="*60)
print(f"Scores: {stratified_scores}")
print(f"Média: {stratified_scores.mean():.3f}")

# ───────────────────────────────────────────────────────────────────
# COMPARAR MÚLTIPLAS MÉTRICAS
# ───────────────────────────────────────────────────────────────────

from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\n" + "="*60)
print("MÚLTIPLAS MÉTRICAS")
print("="*60)
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric:20s}: {scores.mean():.3f} ± {scores.std():.3f}")
