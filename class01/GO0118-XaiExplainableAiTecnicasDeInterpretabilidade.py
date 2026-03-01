# GO0118-XaiExplainableAiTécnicasDeInterpretabilidade
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# ═══════════════════════════════════════════════════════════════════
# XAI (EXPLAINABLE AI) - TÉCNICAS DE INTERPRETABILIDADE
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("XAI: EXPLAINABLE AI - ABRINDO A CAIXA PRETA")
print("="*70)

# Por que XAI importa?
print("\n🎯 POR QUE EXPLICABILIDADE IMPORTA?")
print("="*70)

motivos = {
    "Confiança": "Usuários precisam entender decisões críticas (médico, juiz)",
    "Debug": "Desenvolvedores precisam diagnosticar erros e viés",
    "Regulação": "GDPR (Art. 22) e LGPD (Art. 20) - direito à explicação",
    "Ciência": "Pesquisadores querem entender o que o modelo aprendeu",
    "Ética": "Detectar discriminação e viés oculto"
}

for motivo, razao in motivos.items():
    print(f"• {motivo}: {razao}")

# Exemplo: Modelo Black-Box
print("\n" + "="*70)
print("PROBLEMA: MODELO BLACK-BOX")
print("="*70)

# Gerar dados sintéticos
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                          n_redundant=2, random_state=42)

# Feature names
feature_names = [f'Feature_{i}' for i in range(10)]
feature_names[0] = "Idade"
feature_names[1] = "Renda"
feature_names[2] = "Score Crédito"
feature_names[3] = "Dívidas"

# Treinar Random Forest (modelo black-box)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

accuracy = rf.score(X, y)
print(f"\n✅ Random Forest treinada: {accuracy*100:.1f}% acurácia")
print("❌ PROBLEMA: Não sabemos POR QUÊ o modelo decidiu!")

# Exemplo de predição
exemplo_idx = 5
exemplo = X[exemplo_idx:exemplo_idx+1]
predicao = rf.predict(exemplo)[0]
prob = rf.predict_proba(exemplo)[0]

print(f"\n📋 EXEMPLO DE PREDIÇÃO:")
print(f"   Idade: {exemplo[0, 0]:.2f}")
print(f"   Renda: {exemplo[0, 1]:.2f}")
print(f"   Score Crédito: {exemplo[0, 2]:.2f}")
print(f"   Dívidas: {exemplo[0, 3]:.2f}")
print(f"   ...")
print(f"\n   🤖 Predição: {'Aprovado' if predicao == 1 else 'Negado'}")
print(f"   Probabilidade: {prob[predicao]*100:.1f}%")
print(f"\n   ❓ POR QUÊ? → XAI responde!")

# Técnica 1: Feature Importance (Global)
print("\n" + "="*70)
print("TÉCNICA 1: FEATURE IMPORTANCE (Global)")
print("="*70)
print("Pergunta: Quais features são mais importantes para o modelo?")
print("="*70)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n📊 IMPORTÂNCIA DAS FEATURES:")
for i in range(len(feature_names)):
    idx = indices[i]
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f} "
          f"({'█' * int(importances[idx] * 100)})")

print("\n💡 Interpretação:")
print(f"   → Feature mais importante: {feature_names[indices[0]]}")
print(f"   → Modelo foca nessa feature para decisões")

# Técnica 2: Partial Dependence Plot (Global)
print("\n" + "="*70)
print("TÉCNICA 2: PARTIAL DEPENDENCE (Global)")
print("="*70)
print("Pergunta: Como cada feature afeta a predição?")
print("="*70)

def partial_dependence(model, X, feature_idx, n_points=50):
    """Calcula dependência parcial para uma feature"""
    feature_values = np.linspace(X[:, feature_idx].min(), 
                                 X[:, feature_idx].max(), n_points)
    predictions = []

    for value in feature_values:
        X_temp = X.copy()
        X_temp[:, feature_idx] = value
        pred = model.predict_proba(X_temp)[:, 1].mean()
        predictions.append(pred)

    return feature_values, predictions

# Calcular PDP para top 3 features
print("\n📈 Calculando Partial Dependence Plots...")

pdp_data = {}
for i in range(3):
    feature_idx = indices[i]
    values, preds = partial_dependence(rf, X, feature_idx)
    pdp_data[feature_names[feature_idx]] = (values, preds)

    print(f"   ✓ {feature_names[feature_idx]}")

# Técnica 3: LIME (Local - Instância Específica)
print("\n" + "="*70)
print("TÉCNICA 3: LIME (Local Interpretable Model-Agnostic Explanations)")
print("="*70)
print("Pergunta: Por que ESTA instância foi classificada assim?")
print("="*70)

def lime_explanation(model, X_train, instance, n_samples=1000):
    """Simplificação de LIME para demonstração"""
    # Garantir que instance seja 1D
    if instance.ndim > 1:
        instance = instance.flatten()

    # Gerar amostras perturbadas ao redor da instância
    n_features = len(instance)
    perturbations = np.random.normal(instance, 0.1, size=(n_samples, n_features))

    # Predizer amostras
    predictions = model.predict_proba(perturbations)[:, 1]

    # Treinar modelo linear local (interpretável)
    from sklearn.linear_model import Ridge
    local_model = Ridge()
    local_model.fit(perturbations, predictions)

    # Coeficientes explicam importância local
    return local_model.coef_

print(f"\n🔍 Explicando instância específica (#{exemplo_idx}):")
# Garantir que exemplo[0] seja 1D
instance_to_explain = exemplo[0] if exemplo[0].ndim == 1 else exemplo[0].flatten()
lime_coefs = lime_explanation(rf, X, instance_to_explain)

print("\n📊 IMPORTÂNCIA LOCAL (para ESTA instância):")
lime_indices = np.argsort(np.abs(lime_coefs))[::-1]

for i in range(5):  # Top 5
    idx = lime_indices[i]
    coef = lime_coefs[idx]
    sinal = "aumenta" if coef > 0 else "diminui"
    print(f"{i+1}. {feature_names[idx]}: {coef:+.4f} ({sinal} chance de aprovação)")

print("\n💡 Interpretação:")
print(f"   → Para ESTA instância, {feature_names[lime_indices[0]]} é mais relevante")
print(f"   → Diferente da importância global!")

# Técnica 4: Counterfactual Explanations
print("\n" + "="*70)
print("TÉCNICA 4: COUNTERFACTUAL EXPLANATIONS")
print("="*70)
print("Pergunta: O que MUDAR para obter decisão diferente?")
print("="*70)

def find_counterfactual(model, instance, target_class, max_iter=100):
    """Busca contrafactual mais próximo"""
    best_counterfactual = None
    best_distance = float('inf')

    for _ in range(max_iter):
        # Perturbar aleatoriamente
        perturbation = np.random.normal(0, 0.3, size=instance.shape)
        candidate = instance + perturbation

        # Verificar se muda classe
        pred = model.predict(candidate.reshape(1, -1))[0]
        if pred == target_class:
            distance = np.linalg.norm(candidate - instance)
            if distance < best_distance:
                best_distance = distance
                best_counterfactual = candidate

    return best_counterfactual

if predicao == 0:  # Se foi negado
    print(f"\n❌ Instância foi NEGADA")
    print("🔄 Buscando contrafactual (como obter APROVAÇÃO)...")

    # Garantir que seja 1D
    instance_for_cf = exemplo[0] if exemplo[0].ndim == 1 else exemplo[0].flatten()
    counterfactual = find_counterfactual(rf, instance_for_cf, target_class=1)

    if counterfactual is not None:
        print("\n✅ CONTRAFACTUAL ENCONTRADO:")
        print("   Se mudar as features assim:")

        changes = []
        for i in range(min(5, len(feature_names))):
            # Acessar corretamente o valor original
            original = instance_for_cf[i]
            new_value = counterfactual[i]
            change = new_value - original

            if abs(change) > 0.1:  # Mudança significativa
                print(f"      • {feature_names[i]}: {original:.2f} → {new_value:.2f} "
                      f"({change:+.2f})")
                changes.append((feature_names[i], change))

        if changes:
            maior_mudanca = max(changes, key=lambda x: abs(x[1]))
            print(f"\n   💡 Maior mudança necessária: {maior_mudanca[0]} ({maior_mudanca[1]:+.2f})")
            print(f"      → Cliente deveria melhorar isso para aprovação!")
else:
    print(f"\n✅ Instância já foi APROVADA (sem contrafactual necessário)")

# Visualização
print("\n📊 GERANDO VISUALIZAÇÕES XAI...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance
ax1 = axes[0, 0]
top_n = 8
top_indices = indices[:top_n]
top_importances = importances[top_indices]
top_names = [feature_names[i] for i in top_indices]

colors = plt.cm.viridis(np.linspace(0, 1, top_n))
bars = ax1.barh(top_names, top_importances, color=colors)
ax1.set_xlabel("Importância", fontsize=12)
ax1.set_title("Feature Importance (Global)", fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# 2. Partial Dependence Plots
ax2 = axes[0, 1]
for feature_name, (values, preds) in pdp_data.items():
    ax2.plot(values, preds, marker='o', linewidth=2, label=feature_name, markersize=4)

ax2.set_xlabel("Valor da Feature", fontsize=12)
ax2.set_ylabel("Probabilidade Predita (Classe 1)", fontsize=12)
ax2.set_title("Partial Dependence Plot (Global)", fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. LIME Local Importance
ax3 = axes[1, 0]
top_lime = 6
top_lime_indices = lime_indices[:top_lime]
top_lime_coefs = lime_coefs[top_lime_indices]
top_lime_names = [feature_names[i] for i in top_lime_indices]

colors_lime = ['green' if c > 0 else 'red' for c in top_lime_coefs]
bars_lime = ax3.barh(top_lime_names, top_lime_coefs, color=colors_lime, alpha=0.7)
ax3.set_xlabel("Coeficiente LIME (Local)", fontsize=12)
ax3.set_title(f"LIME: Explicação Local (Instância #{exemplo_idx})", 
             fontsize=13, fontweight='bold')
ax3.invert_yaxis()
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(True, alpha=0.3, axis='x')

# 4. Comparação: Global vs Local
ax4 = axes[1, 1]
ax4.axis('off')

texto_comparacao = f"""
📊 GLOBAL vs LOCAL IMPORTANCE

GLOBAL (Feature Importance):
   Top 1: {feature_names[indices[0]]}
   Top 2: {feature_names[indices[1]]}
   Top 3: {feature_names[indices[2]]}

   → Importância geral no dataset

LOCAL (LIME - Instância #{exemplo_idx}):
   Top 1: {feature_names[lime_indices[0]]}
   Top 2: {feature_names[lime_indices[1]]}
   Top 3: {feature_names[lime_indices[2]]}

   → Importância para ESTA instância

⚠️ DIFERENÇA:
   Global e Local podem divergir!
   Use Local para explicar decisões individuais.

🛠️ FERRAMENTAS XAI (Python):
   • SHAP (SHapley Additive exPlanations)
   • LIME (Local Interpretable Model-Agnostic)
   • ELI5 (Explain Like I'm 5)
   • InterpretML (Microsoft)
   • Captum (PyTorch)

📜 REGULAÇÃO:
   • GDPR (EU) - Art. 22: Direito à explicação
   • LGPD (BR) - Art. 20: Revisão de decisões automatizadas
   • Fair Credit Reporting Act (EUA)
"""

ax4.text(0.1, 0.5, texto_comparacao, fontsize=9, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax4.set_title("Comparação e Ferramentas", fontsize=13, fontweight='bold')

plt.suptitle("XAI: Explicando Modelos Black-Box", 
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Resumo
print("\n💡 RESUMO DAS TÉCNICAS XAI:")
print("="*70)
print("1️⃣ FEATURE IMPORTANCE: Quais features importam globalmente?")
print("2️⃣ PARTIAL DEPENDENCE: Como features afetam predições?")
print("3️⃣ LIME: Por que ESTA instância foi classificada assim?")
print("4️⃣ COUNTERFACTUAL: O que mudar para obter decisão diferente?")

print("\n🎯 QUANDO USAR CADA TÉCNICA:")
print("   • Feature Importance: Entender modelo geral")
print("   • Partial Dependence: Visualizar efeitos de features")
print("   • LIME/SHAP: Explicar predições individuais (medicina, justiça)")
print("   • Counterfactual: Recomendar ações (crédito, RH)")

print("\n⚖️ TRADE-OFF:")
print("   Interpretabilidade ↔ Performance")
print("   • Modelos simples (linear, árvore): Interpretável mas menos preciso")
print("   • Modelos complexos (neural, ensemble): Preciso mas opaco")
print("   • XAI: Tenta ter os dois (modelo complexo + explicação post-hoc)")

print("\n✅ XAI COMPLETO!")
