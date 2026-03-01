# GO0636-NumpyMatplotlib
# GO06EX3-ValidacaoCruzadaAvancada
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    cross_val_score, cross_validate, 
    KFold, StratifiedKFold, LeaveOneOut,
    learning_curve, validation_curve
)
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

print("="*70)
print("EXERCÍCIO 3: VALIDAÇÃO CRUZADA AVANÇADA")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 1. GERAR DATASET DE REGRESSÃO
# ═══════════════════════════════════════════════════════════════════

X, y = make_regression(
    n_samples=200, 
    n_features=10, 
    n_informative=5,
    noise=10.0, 
    random_state=42
)

print("\n📊 DATASET:")
print(f"   • {X.shape[0]} amostras")
print(f"   • {X.shape[1]} features")
print(f"   • Target range: [{y.min():.2f}, {y.max():.2f}]")

# ═══════════════════════════════════════════════════════════════════
# 2. COMPARAR ESTRATÉGIAS DE VALIDAÇÃO CRUZADA
# ═══════════════════════════════════════════════════════════════════

model = Ridge(alpha=1.0)

print("\n🔄 EXECUTANDO VALIDAÇÃO CRUZADA...")

# K-Fold (5 folds)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(model, X, y, cv=kfold, 
                               scoring='neg_mean_squared_error')
mse_kfold = -scores_kfold
print(f"\n📐 K-FOLD (5 folds):")
print(f"   • MSE médio: {mse_kfold.mean():.4f}")
print(f"   • Desvio padrão: {mse_kfold.std():.4f}")
print(f"   • MSE por fold: {mse_kfold}")

# K-Fold (10 folds)
kfold10 = KFold(n_splits=10, shuffle=True, random_state=42)
scores_kfold10 = cross_val_score(model, X, y, cv=kfold10, 
                                 scoring='neg_mean_squared_error')
mse_kfold10 = -scores_kfold10
print(f"\n📐 K-FOLD (10 folds):")
print(f"   • MSE médio: {mse_kfold10.mean():.4f}")
print(f"   • Desvio padrão: {mse_kfold10.std():.4f}")

# Leave-One-Out (apenas em dataset pequeno!)
if X.shape[0] <= 100:  # Só executar se dataset for pequeno
    loo = LeaveOneOut()
    scores_loo = cross_val_score(model, X, y, cv=loo, 
                                scoring='neg_mean_squared_error')
    mse_loo = -scores_loo
    print(f"\n📐 LEAVE-ONE-OUT:")
    print(f"   • MSE médio: {mse_loo.mean():.4f}")
    print(f"   • Desvio padrão: {mse_loo.std():.4f}")
else:
    print(f"\n⚠️ LEAVE-ONE-OUT: Pulado (dataset muito grande: {X.shape[0]} amostras)")

# ═══════════════════════════════════════════════════════════════════
# 3. VALIDAÇÃO CRUZADA COM MÚLTIPLAS MÉTRICAS
# ═══════════════════════════════════════════════════════════════════

print("\n📊 VALIDAÇÃO COM MÚLTIPLAS MÉTRICAS...")

scoring = {
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}

cv_results = cross_validate(
    model, X, y, cv=5, scoring=scoring, 
    return_train_score=True
)

print(f"\n🎯 RESULTADOS (5-fold CV):")
print(f"   • MSE Teste:  {-cv_results['test_neg_mse'].mean():.4f} "
      f"± {cv_results['test_neg_mse'].std():.4f}")
print(f"   • MAE Teste:  {-cv_results['test_neg_mae'].mean():.4f} "
      f"± {cv_results['test_neg_mae'].std():.4f}")
print(f"   • R² Teste:   {cv_results['test_r2'].mean():.4f} "
      f"± {cv_results['test_r2'].std():.4f}")
print(f"   • MSE Treino: {-cv_results['train_neg_mse'].mean():.4f} "
      f"± {cv_results['train_neg_mse'].std():.4f}")
print(f"   • R² Treino:  {cv_results['train_r2'].mean():.4f} "
      f"± {cv_results['train_r2'].std():.4f}")

# Detectar overfitting
gap = cv_results['train_r2'].mean() - cv_results['test_r2'].mean()
if gap > 0.1:
    print(f"   ⚠️ OVERFITTING detectado (gap R²: {gap:.4f})")
else:
    print(f"   ✅ Modelo balanceado (gap R²: {gap:.4f})")

# ═══════════════════════════════════════════════════════════════════
# 4. CURVA DE APRENDIZADO (Learning Curve)
# ═══════════════════════════════════════════════════════════════════

print("\n📈 GERANDO CURVA DE APRENDIZADO...")

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

train_scores_mean = -train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = -test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', 
        linewidth=3, markersize=8, label='Erro Treino')
plt.plot(train_sizes, test_scores_mean, 's-', color='red', 
        linewidth=3, markersize=8, label='Erro Teste')

plt.fill_between(train_sizes, 
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, 
                 alpha=0.2, color='blue')
plt.fill_between(train_sizes, 
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, 
                 alpha=0.2, color='red')

plt.xlabel('Tamanho do Dataset de Treino', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
plt.title('Learning Curve: Erro vs Tamanho do Dataset', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 5. CURVA DE VALIDAÇÃO (Validation Curve)
# ═══════════════════════════════════════════════════════════════════

print("\n📈 GERANDO CURVA DE VALIDAÇÃO...")

param_range = np.logspace(-2, 2, 10)
train_scores_val, test_scores_val = validation_curve(
    Ridge(), X, y, 
    param_name='alpha', 
    param_range=param_range,
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

train_scores_val_mean = -train_scores_val.mean(axis=1)
train_scores_val_std = train_scores_val.std(axis=1)
test_scores_val_mean = -test_scores_val.mean(axis=1)
test_scores_val_std = test_scores_val.std(axis=1)

plt.figure(figsize=(12, 6))
plt.semilogx(param_range, train_scores_val_mean, 'o-', color='blue', 
            linewidth=3, markersize=8, label='Erro Treino')
plt.semilogx(param_range, test_scores_val_mean, 's-', color='red', 
            linewidth=3, markersize=8, label='Erro Teste')

plt.fill_between(param_range, 
                 train_scores_val_mean - train_scores_val_std,
                 train_scores_val_mean + train_scores_val_std, 
                 alpha=0.2, color='blue')
plt.fill_between(param_range, 
                 test_scores_val_mean - test_scores_val_std,
                 test_scores_val_mean + test_scores_val_std, 
                 alpha=0.2, color='red')

# Marcar melhor alpha
best_alpha = param_range[np.argmin(test_scores_val_mean)]
plt.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2,
           label=f'Melhor Alpha = {best_alpha:.4f}')

plt.xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
plt.title('Validation Curve: Erro vs Alpha (Ridge)', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n🎯 MELHOR ALPHA: {best_alpha:.4f}")
print(f"   • MSE Teste: {test_scores_val_mean.min():.4f}")

# ═══════════════════════════════════════════════════════════════════
# 6. COMPARAÇÃO VISUAL DE ESTRATÉGIAS DE CV
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))

strategies = ['5-Fold', '10-Fold']
mse_means = [mse_kfold.mean(), mse_kfold10.mean()]
mse_stds = [mse_kfold.std(), mse_kfold10.std()]

x_pos = np.arange(len(strategies))
bars = ax.bar(x_pos, mse_means, yerr=mse_stds, 
             capsize=10, alpha=0.7, color=['blue', 'orange'])

ax.set_xlabel('Estratégia de Validação', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax.set_title('Comparação de Estratégias de Validação Cruzada', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(strategies)
ax.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for i, (mean, std) in enumerate(zip(mse_means, mse_stds)):
    ax.text(i, mean + std + 5, f'{mean:.2f}±{std:.2f}', 
           ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 7. CONCLUSÕES E INTERPRETAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎓 CONCLUSÕES DO EXERCÍCIO:")
print("="*70)

print("""
📚 ESTRATÉGIAS DE VALIDAÇÃO CRUZADA:

📐 K-FOLD CROSS-VALIDATION:
   • Divide dataset em K partes (folds)
   • Treina K vezes, cada vez usando fold diferente como teste
   • K=5 ou K=10 são valores comuns
   • Trade-off: K maior → mais treino, mais computação

🎯 LEAVE-ONE-OUT (LOO):
   • Caso extremo: K = n (cada amostra é um fold)
   • Sem viés, mas MUITO computacionalmente caro
   • Variância alta (um outlier pode afetar muito)
   • Só usar em datasets pequenos (<100 amostras)

📊 STRATIFIED K-FOLD:
   • Mantém proporção de classes em cada fold
   • Importante para datasets desbalanceados
   • Para classificação, não para regressão

⚡ TIME SERIES SPLIT:
   • Para dados temporais
   • Respeita ordem cronológica
   • Evita data leakage

📈 LEARNING CURVE:
   • Mostra como erro varia com tamanho do dataset
   • Identifica se precisa de mais dados
   • Curvas convergindo → modelo estável
   • Gap grande → overfitting

📉 VALIDATION CURVE:
   • Mostra como erro varia com hiperparâmetro
   • Identifica melhor valor do hiperparâmetro
   • U-shape: underfitting à esquerda, overfitting à direita

✅ MELHORES PRÁTICAS:

1️⃣ SEMPRE use validação cruzada para:
   • Avaliar modelo de forma robusta
   • Evitar overfitting no conjunto de validação
   • Estimar variância do desempenho

2️⃣ NÃO use validação cruzada para:
   • Treinar modelo final (use todo o dataset)
   • Dados temporais sem time series split
   • Datasets gigantes (muito caro computacionalmente)

3️⃣ INTERPRETE os resultados:
   • Média → desempenho esperado
   • Desvio padrão → estabilidade do modelo
   • Gap treino/teste → overfitting

🎯 REGRA DE OURO:
   Validação cruzada é seu MELHOR AMIGO para avaliar modelos de forma honesta!
""")

print("="*70)
print("✅ EXERCÍCIO 3 COMPLETO!")
print("="*70)
print("\n🎊 PARABÉNS! Você completou todos os exercícios da seção complementar!")
