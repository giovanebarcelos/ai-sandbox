# GO0418-ExercicioAvancado1PipelineAutomatizado
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PIPELINE AUTOMATIZADO COM OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("=" * 70)

# 1. CARREGAR DATASET
data = load_breast_cancer()
X, y = data.data, data.target

print(f"\n📊 Dataset: {X.shape[0]} amostras, {X.shape[1]} features")

# 2. DIVIDIR DADOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. DEFINIR PIPELINES
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42, probability=True))
])

# 4. GRID DE HIPERPARÂMETROS
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

param_grid_svm = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['rbf', 'linear']
}

# 5. GRID SEARCH
print("\n🔍 EXECUTANDO GRID SEARCH...")

models = {
    'Random Forest': (pipeline_rf, param_grid_rf),
    'SVM': (pipeline_svm, param_grid_svm)
}

results = {}

for name, (pipeline, param_grid) in models.items():
    print(f"\n⚙️  Otimizando {name}...")

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    results[name] = {
        'best_score': grid_search.best_score_,
        'test_score': grid_search.score(X_test, y_test),
        'best_params': grid_search.best_params_
    }

    print(f"   ✓ CV Score: {grid_search.best_score_:.4f}")
    print(f"   ✓ Test Score: {results[name]['test_score']:.4f}")

# 6. COMPARAÇÃO
print("\n" + "=" * 70)
print("COMPARAÇÃO")
print("=" * 70)

for name, res in results.items():
    print(f"\n{name}:")
    print(f"  CV: {res['best_score']:.4f}")
    print(f"  Test: {res['test_score']:.4f}")

print("\n✅ Pipeline Automatizado concluído!")
