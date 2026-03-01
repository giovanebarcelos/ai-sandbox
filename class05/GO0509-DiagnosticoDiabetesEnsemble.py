# GO0509-DiagnósticoDiabetesEnsemble
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                              BaggingClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# 1. CARREGAR E PREPARAR DADOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("DIAGNÓSTICO DE DIABETES - ENSEMBLE METHODS")
print("="*70)

# Criar dataset simulado de diabetes (baseado em features reais)
np.random.seed(42)

n_samples = 500

# Features: Glucose, BMI, Age, Blood Pressure, Insulin
glucose = np.random.normal(120, 30, n_samples)
bmi = np.random.normal(28, 7, n_samples)
age = np.random.randint(21, 80, n_samples)
blood_pressure = np.random.normal(80, 15, n_samples)
insulin = np.random.normal(100, 50, n_samples)

# Target: Diabetes (1) ou Não (0)
# Regra simplificada: alta probabilidade se glucose alto E BMI alto
diabetes_proba = 1 / (1 + np.exp(-(0.05*glucose + 0.1*bmi + 0.02*age - 10)))
diabetes = (diabetes_proba + np.random.normal(0, 0.1, n_samples) > 0.6).astype(int)

# Criar DataFrame
df = pd.DataFrame({
    'Glucose': glucose,
    'BMI': bmi,
    'Age': age,
    'BloodPressure': blood_pressure,
    'Insulin': insulin,
    'Diabetes': diabetes
})

print(f"\n📊 Dataset criado: {len(df)} pacientes")
print(f"   • Com diabetes: {sum(diabetes)} ({sum(diabetes)/len(diabetes)*100:.1f}%)")
print(f"   • Sem diabetes: {len(diabetes)-sum(diabetes)} ({(len(diabetes)-sum(diabetes))/len(diabetes)*100:.1f}%)")

print("\n📈 Estatísticas descritivas:")
print(df.describe())

# ═══════════════════════════════════════════════════════════════════
# 2. PRÉ-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📊 Divisão dos dados:")
print(f"   • Treino: {X_train.shape[0]} pacientes")
print(f"   • Teste: {X_test.shape[0]} pacientes")

# ═══════════════════════════════════════════════════════════════════
# 3. MODELOS BASE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PARTE 1: MODELOS BASE (SEM ENSEMBLE)")
print("="*70)

modelos_base = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

resultados_base = {}

for nome, modelo in modelos_base.items():
    print(f"\n🔄 Treinando {nome}...")
    modelo.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')

    # Predições
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    # AUC
    auc = roc_auc_score(y_test, y_proba)

    resultados_base[nome] = {
        'modelo': modelo,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"   ✅ AUC: {auc:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 4. VOTING CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PARTE 2: VOTING CLASSIFIER")
print("="*70)

# Hard Voting (maioria de votos)
voting_hard = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', random_state=42))
    ],
    voting='hard'
)

# Soft Voting (média de probabilidades)
voting_soft = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ],
    voting='soft'
)

resultados_voting = {}

for nome, modelo in [('Voting Hard', voting_hard), ('Voting Soft', voting_soft)]:
    print(f"\n🔄 Treinando {nome}...")
    modelo.fit(X_train, y_train)

    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    y_pred = modelo.predict(X_test)

    if nome == 'Voting Soft':
        y_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None
        y_proba = None

    resultados_voting[nome] = {
        'modelo': modelo,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    if auc:
        print(f"   ✅ AUC: {auc:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 5. BAGGING
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PARTE 3: BAGGING (BOOTSTRAP AGGREGATING)")
print("="*70)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)

print("\n🔄 Treinando Bagging...")
bagging.fit(X_train, y_train)

cv_scores = cross_val_score(bagging, X_train, y_train, cv=5, scoring='accuracy')
y_pred = bagging.predict(X_test)
y_proba = bagging.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

resultados_bagging = {
    'Bagging': {
        'modelo': bagging,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
}

print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"   ✅ AUC: {auc:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 6. BOOSTING
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PARTE 4: BOOSTING (ADABOOST E GRADIENT BOOSTING)")
print("="*70)

# AdaBoost
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Gradient Boosting
gradient_boost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

resultados_boosting = {}

for nome, modelo in [('AdaBoost', adaboost), ('Gradient Boosting', gradient_boost)]:
    print(f"\n🔄 Treinando {nome}...")
    modelo.fit(X_train, y_train)

    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    resultados_boosting[nome] = {
        'modelo': modelo,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"   ✅ AUC: {auc:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 7. COMPARAÇÃO GERAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPARAÇÃO GERAL DE TODOS OS MÉTODOS")
print("="*70)

# Consolidar todos os resultados
todos_resultados = {**resultados_base, **resultados_voting, 
                    **resultados_bagging, **resultados_boosting}

# Ordenar por acurácia
ranking = sorted(todos_resultados.items(), 
                key=lambda x: x[1]['cv_mean'], reverse=True)

print("\n🏆 RANKING DE MODELOS:\n")
for i, (nome, res) in enumerate(ranking, 1):
    auc_str = f"AUC: {res['auc']:.4f}" if res['auc'] else "AUC: N/A"
    print(f"   {i}. {nome:20s} | Acc: {res['cv_mean']:.4f} (±{res['cv_std']:.4f}) | {auc_str}")

# ═══════════════════════════════════════════════════════════════════
# 8. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n🎨 Gerando visualizações...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8.1 Comparação de Acurácias
ax1 = fig.add_subplot(gs[0, :])
nomes = [nome for nome, _ in ranking]
accs = [res['cv_mean'] for _, res in ranking]
stds = [res['cv_std'] for _, res in ranking]

colors = ['#3498db' if 'Voting' not in nome and 'Bagging' not in nome and 'Boost' not in nome 
          else '#e74c3c' for nome in nomes]

bars = ax1.barh(nomes, accs, xerr=stds, color=colors, alpha=0.7)
ax1.set_xlabel('Acurácia (CV)', fontsize=12)
ax1.set_title('Comparação de Todos os Modelos', fontsize=14, fontweight='bold')
ax1.axvline(x=0.75, color='green', linestyle='--', alpha=0.5, label='75% threshold')
ax1.legend()

# 8.2 Curvas ROC
ax2 = fig.add_subplot(gs[1, 0])
for nome, res in todos_resultados.items():
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax2.plot(fpr, tpr, label=f"{nome} (AUC={res['auc']:.3f})", linewidth=2)

ax2.plot([0, 1], [0, 1], 'k--', label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Curvas ROC - Todos os Modelos')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 8.3 Matriz de Confusão do Melhor Modelo
ax3 = fig.add_subplot(gs[1, 1])
melhor_nome, melhor_res = ranking[0]
cm = confusion_matrix(y_test, melhor_res['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax3,
            xticklabels=['Sem Diabetes', 'Com Diabetes'],
            yticklabels=['Sem Diabetes', 'Com Diabetes'])
ax3.set_title(f'Matriz de Confusão\n{melhor_nome}')
ax3.set_ylabel('Real')
ax3.set_xlabel('Predito')

# 8.4 Feature Importance (para modelos que suportam)
ax4 = fig.add_subplot(gs[1, 2])
if hasattr(melhor_res['modelo'], 'feature_importances_'):
    importances = melhor_res['modelo'].feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    ax4.barh(range(len(importances)), importances[indices], color='teal')
    ax4.set_yticks(range(len(importances)))
    ax4.set_yticklabels([feature_names[i] for i in indices])
    ax4.set_xlabel('Importância')
    ax4.set_title(f'Feature Importance\n{melhor_nome}')
    ax4.invert_yaxis()
else:
    ax4.text(0.5, 0.5, 'Feature Importance\nNão disponível\npara este modelo',
            ha='center', va='center', fontsize=12)
    ax4.axis('off')

# 8.5 Comparação AUC
ax5 = fig.add_subplot(gs[2, 0])
nomes_auc = [nome for nome, res in todos_resultados.items() if res['auc']]
aucs = [res['auc'] for nome, res in todos_resultados.items() if res['auc']]

ax5.bar(range(len(nomes_auc)), aucs, color='coral')
ax5.set_xticks(range(len(nomes_auc)))
ax5.set_xticklabels(nomes_auc, rotation=45, ha='right')
ax5.set_ylabel('AUC Score')
ax5.set_title('Comparação de AUC')
ax5.axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
ax5.set_ylim([0.5, 1.0])

# 8.6 Distribuição de Features
ax6 = fig.add_subplot(gs[2, 1:])
df_plot = df.copy()
df_plot['Diabetes'] = df_plot['Diabetes'].map({0: 'Sem Diabetes', 1: 'Com Diabetes'})

for i, col in enumerate(X.columns):
    ax_sub = plt.subplot(2, 3, i+1)
    df_plot.boxplot(column=col, by='Diabetes', ax=ax_sub)
    ax_sub.set_title(col)
    ax_sub.set_xlabel('')
plt.suptitle('Distribuição de Features por Classe', fontsize=12, y=0.02)

fig.suptitle('Análise Completa - Diagnóstico de Diabetes com Ensemble Methods', 
            fontsize=16, fontweight='bold', y=0.995)

plt.show()

# ═══════════════════════════════════════════════════════════════════
# 9. CONCLUSÕES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONCLUSÕES E INSIGHTS")
print("="*70)

print(f"\n🏆 MELHOR MODELO: {melhor_nome}")
print(f"   • Acurácia: {melhor_res['cv_mean']:.4f}")
print(f"   • AUC: {melhor_res['auc']:.4f}" if melhor_res['auc'] else "")

print("\n💡 PRINCIPAIS APRENDIZADOS:")
print("   1. Ensemble methods geralmente superam modelos individuais")
print("   2. Voting combina diferentes perspectivas (diversidade)")
print("   3. Bagging reduz variância (overfitting)")
print("   4. Boosting reduz bias (underfitting)")
print("   5. Gradient Boosting frequentemente oferece melhor desempenho")

print("\n📊 QUANDO USAR CADA MÉTODO:")
print("   • Voting: Quando modelos têm desempenhos similares")
print("   • Bagging: Quando modelo base tem alta variância")
print("   • AdaBoost: Quando precisamos focar em exemplos difíceis")
print("   • Gradient Boosting: Quando buscamos melhor desempenho geral")

print("\n🔧 PRÓXIMOS PASSOS:")
print("   • Otimizar hiperparâmetros (GridSearchCV)")
print("   • Testar XGBoost, LightGBM, CatBoost")
print("   • Adicionar mais features (histórico familiar, exercício)")
print("   • Implementar SHAP para explicabilidade")
print("   • Deploy em produção com monitoramento")

print("\n" + "="*70)
print("FIM DO EXERCÍCIO 2")
print("="*70)
