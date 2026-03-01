# GO0407-PipelineCompletoDeMachineLearning
# ═══════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO DE MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits

# ───────────────────────────────────────────────────────────────────
# CARREGAR DADOS
# ───────────────────────────────────────────────────────────────────

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*60)
print("PIPELINE COMPLETO DE ML")
print("="*60)
print(f"Treino: {X_train.shape}")
print(f"Teste: {X_test.shape}")

# ───────────────────────────────────────────────────────────────────
# CRIAR PIPELINE
# ───────────────────────────────────────────────────────────────────

pipeline = Pipeline([
    ('scaler', StandardScaler()),           # 1. Normalização
    ('pca', PCA(n_components=0.95)),        # 2. Redução dimensional
    ('classifier', RandomForestClassifier(  # 3. Classificador
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])

print("\nPipeline:")
for step_name, step in pipeline.steps:
    print(f"  {step_name}: {step.__class__.__name__}")

# ───────────────────────────────────────────────────────────────────
# TREINAR
# ───────────────────────────────────────────────────────────────────

print("\nTreinando pipeline...")
pipeline.fit(X_train, y_train)

# ───────────────────────────────────────────────────────────────────
# AVALIAR
# ───────────────────────────────────────────────────────────────────

train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(f"Acurácia Treino: {train_score:.3f}")
print(f"Acurácia Teste:  {test_score:.3f}")
print(f"Gap: {abs(train_score - test_score):.3f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"\nCross-validation (5-fold):")
print(f"  Scores: {cv_scores}")
print(f"  Média: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ───────────────────────────────────────────────────────────────────
# INSPEÇÃO DO PIPELINE
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("INSPEÇÃO DOS STEPS")
print("="*60)

# PCA
pca = pipeline.named_steps['pca']
print(f"PCA manteve {pca.n_components_} componentes")
print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.2%}")

# Classifier
clf = pipeline.named_steps['classifier']
print(f"\nClassificador: {clf.n_estimators} árvores")

print("\n✅ Pipeline completo funcional!")
