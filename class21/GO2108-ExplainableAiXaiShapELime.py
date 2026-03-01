# GO2108-ExplainableAiXaiShapELime
# ═══════════════════════════════════════════════════════════════════
# EXPLAINABLE AI (XAI) - SHAP E LIME
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
import shap
from lime import lime_tabular
import seaborn as sns

# ─── 1. CARREGAR DATASET ───
print("📦 Carregando dataset de câncer de mama...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Informações
print(f"  Features: {X.shape[1]}")
print(f"  Amostras: {X.shape[0]}")
print(f"  Classes: {data.target_names}")
print(f"  Distribuição: Maligno={sum(y==0)}, Benigno={sum(y==1)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 2. TREINAR MODELOS ───
print("\n🔨 Treinando modelos...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)
print(f"  ✓ Random Forest - Accuracy: {rf_score:.4f}")

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train, y_train)
gb_score = gb_model.score(X_test, y_test)
print(f"  ✓ Gradient Boosting - Accuracy: {gb_score:.4f}")

# Usar melhor modelo
model = rf_model if rf_score > gb_score else gb_model
model_name = "Random Forest" if rf_score > gb_score else "Gradient Boosting"
print(f"\n✅ Usando: {model_name}")

# ─── 3. SHAP - GLOBAL EXPLANATIONS ───
print("\n📊 Gerando explicações SHAP (globais)...")

# Criar explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Se classificação binária, shap_values pode ter 2 arrays
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Classe positiva (benigno)

# 3.1 - Summary Plot (Importância + Valores)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot - Importância e Impacto', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
print("  ✓ Summary plot salvo: shap_summary_plot.png")

# 3.2 - Bar Plot (Importância média)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
print("  ✓ Feature importance salvo: shap_feature_importance.png")

# 3.3 - Dependence Plot (feature específica)
top_feature = X_test.columns[np.abs(shap_values).mean(0).argmax()]
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    top_feature, 
    shap_values, 
    X_test, 
    show=False
)
plt.title(f'SHAP Dependence Plot - {top_feature}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_dependence_plot.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Dependence plot salvo: shap_dependence_plot.png")

# ─── 4. SHAP - LOCAL EXPLANATION (instância específica) ───
print("\n🔍 Gerando explicações SHAP (locais)...")

# Selecionar instâncias de interesse
# Caso verdadeiro positivo (benigno previsto corretamente)
tp_idx = np.where((model.predict(X_test) == 1) & (y_test == 1))[0][0]

# Caso falso positivo (maligno previsto como benigno - ERRO!)
fp_idx = np.where((model.predict(X_test) == 1) & (y_test == 0))[0][0]

# Explicar TP
plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[tp_idx], 
        base_values=explainer.expected_value,
        data=X_test.iloc[tp_idx],
        feature_names=X_test.columns
    ),
    show=False
)
plt.title(f'SHAP Waterfall - Caso Benigno (Correto)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_tp.png', dpi=150, bbox_inches='tight')
print("  ✓ Waterfall (TP) salvo: shap_waterfall_tp.png")

# Explicar FP
plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[fp_idx], 
        base_values=explainer.expected_value,
        data=X_test.iloc[fp_idx],
        feature_names=X_test.columns
    ),
    show=False
)
plt.title(f'SHAP Waterfall - Falso Positivo (ERRO!)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_fp.png', dpi=150, bbox_inches='tight')
print("  ✓ Waterfall (FP) salvo: shap_waterfall_fp.png")

# Force plot para múltiplas instâncias
plt.figure(figsize=(16, 4))
shap.force_plot(
    explainer.expected_value,
    shap_values[:50],
    X_test.iloc[:50],
    matplotlib=True,
    show=False
)
plt.title('SHAP Force Plot - 50 primeiras amostras', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_force_plot.png', dpi=150, bbox_inches='tight')
print("  ✓ Force plot salvo: shap_force_plot.png")

# ─── 5. LIME - LOCAL EXPLANATION ───
print("\n🍋 Gerando explicações LIME (locais)...")

# Criar explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=data.target_names,
    mode='classification'
)

# Explicar mesma instância TP
lime_exp_tp = lime_explainer.explain_instance(
    data_row=X_test.iloc[tp_idx].values,
    predict_fn=model.predict_proba,
    num_features=10
)

# Visualizar
fig = lime_exp_tp.as_pyplot_figure()
plt.title(f'LIME - Caso Benigno (Correto)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lime_explanation_tp.png', dpi=150, bbox_inches='tight')
print("  ✓ LIME (TP) salvo: lime_explanation_tp.png")
plt.close()

# Explicar FP
lime_exp_fp = lime_explainer.explain_instance(
    data_row=X_test.iloc[fp_idx].values,
    predict_fn=model.predict_proba,
    num_features=10
)

fig = lime_exp_fp.as_pyplot_figure()
plt.title(f'LIME - Falso Positivo (ERRO!)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lime_explanation_fp.png', dpi=150, bbox_inches='tight')
print("  ✓ LIME (FP) salvo: lime_explanation_fp.png")
plt.close()

# ─── 6. COMPARAÇÃO SHAP vs LIME ───
print("\n📊 Comparando SHAP e LIME...")

# Top features por SHAP
shap_importance = np.abs(shap_values).mean(0)
shap_top = pd.Series(shap_importance, index=X_test.columns).nlargest(10)

# Top features por LIME (da instância TP)
lime_weights = dict(lime_exp_tp.as_list())
lime_features = [f.split('<=')[0].split('>')[0].strip() for f in lime_weights.keys()]
lime_values = list(lime_weights.values())
lime_top = pd.Series(np.abs(lime_values), index=lime_features).nlargest(10)

# Plotar comparação
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# SHAP
shap_top.plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Top 10 Features - SHAP (Global)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('|SHAP Value| médio')
axes[0].invert_yaxis()

# LIME
lime_top.plot(kind='barh', ax=axes[1], color='lightcoral')
axes[1].set_title('Top 10 Features - LIME (Local - Instância TP)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('|Weight|')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('shap_vs_lime_comparison.png', dpi=150)
print("  ✓ Comparação salva: shap_vs_lime_comparison.png")

# ─── 7. ANÁLISE DE CONSISTÊNCIA ───
print("\n🔬 Análise de consistência SHAP vs LIME...")

# Calcular overlap
shap_top_set = set(shap_top.index)
lime_top_set = set(lime_top.index)
overlap = shap_top_set & lime_top_set
overlap_pct = len(overlap) / 10 * 100

print(f"\n  Features em comum (top 10): {len(overlap)}/10 ({overlap_pct:.0f}%)")
print(f"  Features comuns: {', '.join(overlap)}")

# ─── 8. ANÁLISE DE CASOS CRÍTICOS ───
print("\n⚠️ Analisando casos críticos (baixa confiança)...")

# Predições
probs = model.predict_proba(X_test)
confidence = np.max(probs, axis=1)

# Casos com baixa confiança (0.5 - 0.7)
low_conf_mask = (confidence >= 0.5) & (confidence <= 0.7)
low_conf_indices = np.where(low_conf_mask)[0]

print(f"  Casos com baixa confiança: {len(low_conf_indices)}/{len(X_test)}")

if len(low_conf_indices) > 0:
    # Explicar primeiro caso de baixa confiança
    lc_idx = low_conf_indices[0]
    lc_pred = model.predict(X_test.iloc[lc_idx:lc_idx+1])[0]
    lc_prob = probs[lc_idx].max()
    lc_true = y_test.iloc[lc_idx]

    print(f"  Exemplo: Predito={data.target_names[lc_pred]}, "
          f"Real={data.target_names[lc_true]}, Confiança={lc_prob:.2f}")

    # SHAP para caso de baixa confiança
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[lc_idx], 
            base_values=explainer.expected_value,
            data=X_test.iloc[lc_idx],
            feature_names=X_test.columns
        ),
        show=False
    )
    plt.title(f'SHAP - Caso de Baixa Confiança ({lc_prob:.2f})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_low_confidence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Análise salva: shap_low_confidence.png")

# ─── 9. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ ANÁLISE XAI CONCLUÍDA!")
print("="*70)

print("\n📊 Arquivos gerados:")
print("  SHAP (Global):")
print("    - shap_summary_plot.png")
print("    - shap_feature_importance.png")
print("    - shap_dependence_plot.png")
print("  SHAP (Local):")
print("    - shap_waterfall_tp.png")
print("    - shap_waterfall_fp.png")
print("    - shap_force_plot.png")
print("    - shap_low_confidence.png")
print("  LIME (Local):")
print("    - lime_explanation_tp.png")
print("    - lime_explanation_fp.png")
print("  Comparação:")
print("    - shap_vs_lime_comparison.png")

print("\n💡 Insights:")
print(f"  - Feature mais importante: {shap_top.index[0]}")
print(f"  - Overlap SHAP-LIME: {overlap_pct:.0f}%")
print(f"  - Modelo: {model_name} - Accuracy: {model.score(X_test, y_test):.4f}")
print(f"  - Casos de baixa confiança: {len(low_conf_indices)}")
