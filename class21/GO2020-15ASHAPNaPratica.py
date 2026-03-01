# GO2020-15ASHAPNaPrática
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# ═══════════════════════════════════════════════════════════
# 1. CARREGAR DADOS (German Credit Dataset)
# ═══════════════════════════════════════════════════════════
data = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
X = data.data
y = (data.target == 'bad').astype(int)  # 1 = inadimplente, 0 = bom pagador

# Selecionar features numéricas para simplicidade
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Taxa inadimplência: {y.mean():.1%}")

# ═══════════════════════════════════════════════════════════
# 2. TREINAR MODELO XGBOOST
# ═══════════════════════════════════════════════════════════
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Acurácia treino: {model.score(X_train, y_train):.3f}")
print(f"Acurácia teste: {model.score(X_test, y_test):.3f}")

# ═══════════════════════════════════════════════════════════
# 3. CALCULAR SHAP VALUES
# ═══════════════════════════════════════════════════════════
# TreeExplainer: Otimizado para árvores (XGBoost, RF, LightGBM)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values.shape: (200, 7) = (amostras, features)
# Cada valor = contribuição daquela feature para aquela amostra

print(f"SHAP values shape: {shap_values.shape}")
print(f"Valor base (expected value): {explainer.expected_value:.3f}")

# ═══════════════════════════════════════════════════════════
# 4. ANÁLISE GLOBAL - FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════
# Summary Plot: Visão geral do impacto de todas as features
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title("SHAP Feature Importance (Global)")
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=150)
plt.show()

# Summary Plot detalhado (beeswarm)
shap.summary_plot(shap_values, X_test)
plt.title("SHAP Beeswarm Plot - Feature Impact")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════
# 5. ANÁLISE LOCAL - EXPLICAR PREDIÇÃO INDIVIDUAL
# ═══════════════════════════════════════════════════════════
# Pegar um cliente específico (índice 42)
sample_idx = 42
sample = X_test.iloc[sample_idx]
prediction = model.predict_proba(sample.values.reshape(1, -1))[0]
shap_value = shap_values[sample_idx]

print(f"\n{'='*60}")
print(f"EXPLICAÇÃO PARA CLIENTE #{sample_idx}")
print(f"{'='*60}")
print(f"Probabilidade inadimplência: {prediction[1]:.1%}")
print(f"Decisão: {'❌ NEGAR CRÉDITO' if prediction[1] > 0.5 else '✅ APROVAR CRÉDITO'}")
print(f"\nFeatures do cliente:")
print(sample)

# Force Plot: Visualização de "empurra e puxa"
shap.force_plot(
    explainer.expected_value,
    shap_value,
    sample,
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot - Cliente #{sample_idx}")
plt.tight_layout()
plt.savefig(f"shap_force_plot_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
plt.show()

# Waterfall Plot: Decomposição passo a passo
shap.waterfall_plot(
    shap.Explanation(
        values=shap_value,
        base_values=explainer.expected_value,
        data=sample.values,
        feature_names=sample.index.tolist()
    )
)
plt.title(f"SHAP Waterfall Plot - Cliente #{sample_idx}")
plt.tight_layout()
plt.savefig(f"shap_waterfall_sample_{sample_idx}.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════
# 6. ANÁLISE DE DEPENDÊNCIA - RELAÇÃO FEATURE vs SHAP VALUE
# ═══════════════════════════════════════════════════════════
# Exemplo: Como a idade (duration) afeta as predições?
feature_name = 'duration'
shap.dependence_plot(
    feature_name,
    shap_values,
    X_test,
    interaction_index="auto"  # Detecta interação automática
)
plt.title(f"SHAP Dependence Plot - {feature_name}")
plt.tight_layout()
plt.savefig(f"shap_dependence_{feature_name}.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════
# 7. INTERPRETAÇÃO NUMÉRICA
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"DECOMPOSIÇÃO MATEMÁTICA DA PREDIÇÃO")
print(f"{'='*60}")
print(f"Valor Base (média do modelo): {explainer.expected_value:.4f}")
print(f"\nContribuições por feature (SHAP values):")

# Ordenar por impacto absoluto
feature_contributions = pd.DataFrame({
    'Feature': X_test.columns,
    'Value': sample.values,
    'SHAP': shap_value
}).sort_values('SHAP', key=abs, ascending=False)

print(feature_contributions.to_string(index=False))

print(f"\nSoma das contribuições: {shap_value.sum():.4f}")
print(f"Predição final (log-odds): {explainer.expected_value + shap_value.sum():.4f}")
print(f"Probabilidade (sigmoid): {prediction[1]:.4f}")

# ═══════════════════════════════════════════════════════════
# 8. EXPORTAR RELATÓRIO EM TEXTO
# ═══════════════════════════════════════════════════════════
with open(f"shap_report_cliente_{sample_idx}.txt", "w") as f:
    f.write(f"RELATÓRIO DE EXPLICAÇÃO - Cliente #{sample_idx}\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Decisão: {'NEGAR' if prediction[1] > 0.5 else 'APROVAR'}\n")
    f.write(f"Probabilidade inadimplência: {prediction[1]:.1%}\n\n")
    f.write(f"Principais fatores:\n")

    top_features = feature_contributions.head(5)
    for idx, row in top_features.iterrows():
        direction = "aumentou" if row['SHAP'] > 0 else "diminuiu"
        f.write(f"- {row['Feature']} (valor={row['Value']:.2f}): "
                f"{direction} risco em {abs(row['SHAP']):.4f}\n")

print(f"\n✅ Relatório salvo em: shap_report_cliente_{sample_idx}.txt")
