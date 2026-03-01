# GO0634-NumpyMatplotlib
# GO06EX1-RegressaoPolinomialOverfitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("="*70)
print("EXERCÍCIO 1: REGRESSÃO POLINOMIAL E OVERFITTING")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 1. GERAR DADOS SINTÉTICOS (função cúbica com ruído)
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = 0.5 * X**3 - X**2 + 2 * X + 3  # Função verdadeira (grau 3)
y = y_true + np.random.normal(0, 3, size=X.shape)  # Adicionar ruído

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\n📊 DADOS GERADOS:")
print(f"   • Tamanho treino: {len(X_train)} amostras")
print(f"   • Tamanho teste: {len(X_test)} amostras")
print(f"   • Função verdadeira: y = 0.5x³ - x² + 2x + 3 + ruído")

# ═══════════════════════════════════════════════════════════════════
# 2. TREINAR MODELOS DE DIFERENTES GRAUS
# ═══════════════════════════════════════════════════════════════════

graus = [1, 3, 5, 9]  # Graus polinomiais a testar
resultados = {}

print("\n🔄 TREINANDO MODELOS...")

for grau in graus:
    # Transformação polinomial
    poly = PolynomialFeatures(degree=grau)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Treinar modelo
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predições
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Métricas
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    resultados[grau] = {
        'model': model,
        'poly': poly,
        'mse_train': mse_train,
        'mse_test': mse_test,
        'r2_train': r2_train,
        'r2_test': r2_test
    }

    print(f"\n📈 MODELO GRAU {grau}:")
    print(f"   • MSE Treino: {mse_train:.2f}")
    print(f"   • MSE Teste:  {mse_test:.2f}")
    print(f"   • R² Treino:  {r2_train:.4f}")
    print(f"   • R² Teste:   {r2_test:.4f}")

    # Detectar overfitting
    diferenca = mse_test - mse_train
    if diferenca > mse_train * 0.5:  # Se teste > 50% maior que treino
        print(f"   ⚠️ OVERFITTING DETECTADO! (diferença: {diferenca:.2f})")
    elif grau == 1:
        print(f"   ⚠️ UNDERFITTING POSSÍVEL (modelo muito simples)")
    else:
        print(f"   ✅ Modelo balanceado")

# ═══════════════════════════════════════════════════════════════════
# 3. VISUALIZAÇÃO COMPARATIVA
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÕES...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Regressão Polinomial: Análise de Overfitting', 
             fontsize=16, fontweight='bold')

# Preparar dados para plotagem suave
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
y_plot_true = 0.5 * X_plot**3 - X_plot**2 + 2 * X_plot + 3

for idx, grau in enumerate(graus):
    ax = axes[idx // 2, idx % 2]

    # Transformar dados de plotagem
    X_plot_poly = resultados[grau]['poly'].transform(X_plot)
    y_plot_pred = resultados[grau]['model'].predict(X_plot_poly)

    # Plot
    ax.scatter(X_train, y_train, color='blue', alpha=0.4, 
              s=30, label='Treino')
    ax.scatter(X_test, y_test, color='red', alpha=0.4, 
              s=30, label='Teste')
    ax.plot(X_plot, y_plot_true, 'g--', linewidth=2, 
           label='Função Verdadeira', alpha=0.7)
    ax.plot(X_plot, y_plot_pred, 'orange', linewidth=3, 
           label=f'Polinômio Grau {grau}')

    # Anotações
    mse_test = resultados[grau]['mse_test']
    r2_test = resultados[grau]['r2_test']
    ax.set_title(f'Grau {grau}: MSE={mse_test:.2f}, R²={r2_test:.3f}', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 4. CURVA DE APRENDIZADO (Learning Curve)
# ═══════════════════════════════════════════════════════════════════

print("\n📈 CURVA DE APRENDIZADO...")

plt.figure(figsize=(12, 6))

# Plot MSE vs Grau
graus_plot = list(resultados.keys())
mse_train_plot = [resultados[g]['mse_train'] for g in graus_plot]
mse_test_plot = [resultados[g]['mse_test'] for g in graus_plot]

plt.plot(graus_plot, mse_train_plot, 'o-', linewidth=2, 
        markersize=8, label='MSE Treino', color='blue')
plt.plot(graus_plot, mse_test_plot, 's-', linewidth=2, 
        markersize=8, label='MSE Teste', color='red')

plt.xlabel('Grau do Polinômio', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
plt.title('MSE vs Complexidade do Modelo (Overfitting Analysis)', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(graus_plot)

# Marcar região de overfitting
plt.axvspan(5, 9, alpha=0.2, color='red', label='Zona de Overfitting')
plt.axvspan(1, 3, alpha=0.2, color='green', label='Zona Saudável')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 5. CONCLUSÕES E INTERPRETAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎓 CONCLUSÕES DO EXERCÍCIO:")
print("="*70)

print("""
📚 APRENDIZADOS PRINCIPAIS:

1️⃣ MODELO LINEAR (Grau 1):
   • UNDERFITTING: Não captura a curvatura dos dados
   • MSE alto tanto no treino quanto no teste
   • R² baixo indica ajuste ruim

2️⃣ MODELO POLINOMIAL GRAU 3:
   • IDEAL: Captura a estrutura verdadeira dos dados (y = x³)
   • MSE equilibrado entre treino e teste
   • Melhor trade-off bias-variance

3️⃣ MODELO POLINOMIAL ALTO (Grau 5-9):
   • OVERFITTING: Memoriza ruído do treino
   • MSE treino muito baixo, mas teste alto
   • Curva extremamente oscilante fora dos dados

⚖️ TRADE-OFF BIAS-VARIANCE:
   • Bias alto (grau baixo) → Underfitting
   • Variance alta (grau alto) → Overfitting
   • Objetivo: ENCONTRAR O EQUILÍBRIO

🎯 COMO EVITAR OVERFITTING:
   ✅ Usar validação cruzada
   ✅ Aplicar regularização (Ridge/Lasso)
   ✅ Aumentar tamanho do dataset
   ✅ Reduzir complexidade do modelo
   ✅ Usar early stopping
""")

print("="*70)
print("✅ EXERCÍCIO 1 COMPLETO!")
print("="*70)
