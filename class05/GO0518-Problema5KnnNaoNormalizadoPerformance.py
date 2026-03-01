# GO0518-Problema5KnnNãoNormalizadoPerformance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print("TESTE: KNN - Impacto da Normalização na Performance")
    print("=" * 80)

    # ========================================================================
    # CRIAR DATASET COM FEATURES EM ESCALAS MUITO DIFERENTES
    # ========================================================================
    print("\n" + "=" * 80)
    print("CRIANDO DATASET COM ESCALAS DIFERENTES")
    print("=" * 80)

    # Gerar dataset sintético
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    # Manipular escalas: multiplicar features por valores diferentes
    # Feature 0: escala pequena (0-1)
    X[:, 0] = X[:, 0] * 0.1
    # Feature 1: escala média (0-10)
    X[:, 1] = X[:, 1] * 1.0
    # Feature 2: escala grande (0-100)
    X[:, 2] = X[:, 2] * 10.0
    # Feature 3: escala enorme (0-1000)
    X[:, 3] = X[:, 3] * 100.0

    print("\n📊 Estatísticas das features (escalas diferentes):")
    for i in range(X.shape[1]):
        print(f"   Feature {i}: min={X[:, i].min():.2f}, max={X[:, i].max():.2f}, "
              f"std={X[:, i].std():.2f}")

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ========================================================================
    # MÉTODO 1: ❌ SEM NORMALIZAÇÃO (PERFORMANCE RUIM)
    # ========================================================================
    print("\n" + "=" * 80)
    print("MÉTODO 1: ❌ SEM NORMALIZAÇÃO (ERRADO)")
    print("=" * 80)

    model_no_norm = KNeighborsClassifier(n_neighbors=5)
    model_no_norm.fit(X_train, y_train)

    y_pred_no_norm = model_no_norm.predict(X_test)
    acc_no_norm = accuracy_score(y_test, y_pred_no_norm)

    print(f"\n📉 Acurácia SEM normalização: {acc_no_norm:.4f} ({acc_no_norm*100:.2f}%)")
    print("⚠️  Problema: KNN usa distância euclidiana - features com valores grandes")
    print("    dominam o cálculo, ignorando features com valores pequenos!")

    # ========================================================================
    # MÉTODO 2: ✅ COM NORMALIZAÇÃO USANDO PIPELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("MÉTODO 2: ✅ COM NORMALIZAÇÃO - PIPELINE (CORRETO)")
    print("=" * 80)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    pipeline.fit(X_train, y_train)

    y_pred_pipeline = pipeline.predict(X_test)
    acc_pipeline = accuracy_score(y_test, y_pred_pipeline)

    print(f"\n📈 Acurácia COM normalização (Pipeline): {acc_pipeline:.4f} ({acc_pipeline*100:.2f}%)")
    print("✅ Vantagem: Pipeline aplica transformação automaticamente em novos dados!")

    # ========================================================================
    # MÉTODO 3: ✅ COM NORMALIZAÇÃO MANUAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("MÉTODO 3: ✅ COM NORMALIZAÇÃO - MANUAL (CORRETO)")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_manual = KNeighborsClassifier(n_neighbors=5)
    model_manual.fit(X_train_scaled, y_train)

    y_pred_manual = model_manual.predict(X_test_scaled)
    acc_manual = accuracy_score(y_test, y_pred_manual)

    print(f"\n📈 Acurácia COM normalização (Manual): {acc_manual:.4f} ({acc_manual*100:.2f}%)")
    print("⚠️  Atenção: Usar fit_transform() no treino e transform() no teste!")

    # Mostrar como ficaram as escalas após normalização
    print("\n📊 Estatísticas após normalização (média≈0, std≈1):")
    for i in range(X_train_scaled.shape[1]):
        print(f"   Feature {i}: mean={X_train_scaled[:, i].mean():.2f}, "
              f"std={X_train_scaled[:, i].std():.2f}")

    # ========================================================================
    # VISUALIZAÇÃO: Comparar escalas antes e depois
    # ========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZAÇÃO: Escalas Antes e Depois da Normalização")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribuição das features ANTES da normalização
    ax1 = axes[0, 0]
    for i in range(4):
        ax1.hist(X_train[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
    ax1.set_xlabel('Valor', fontsize=11)
    ax1.set_ylabel('Frequência', fontsize=11)
    ax1.set_title('Distribuição ANTES da Normalização\n(escalas muito diferentes)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Distribuição das features DEPOIS da normalização
    ax2 = axes[0, 1]
    for i in range(4):
        ax2.hist(X_train_scaled[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
    ax2.set_xlabel('Valor (normalizado)', fontsize=11)
    ax2.set_ylabel('Frequência', fontsize=11)
    ax2.set_title('Distribuição DEPOIS da Normalização\n(média≈0, std≈1)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Plot 3: Comparação de acurácia
    ax3 = axes[1, 0]
    methods = ['Sem\nNormalização', 'Com Pipeline', 'Manual']
    accuracies = [acc_no_norm, acc_pipeline, acc_manual]
    colors = ['#f44336', '#4caf50', '#4caf50']
    bars = ax3.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Acurácia', fontsize=11)
    ax3.set_title('Comparação de Performance', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)

    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}\n({acc*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: Ganho percentual
    ax4 = axes[1, 1]
    gain_pipeline = ((acc_pipeline - acc_no_norm) / acc_no_norm) * 100
    gain_manual = ((acc_manual - acc_no_norm) / acc_no_norm) * 100

    methods_gain = ['Pipeline', 'Manual']
    gains = [gain_pipeline, gain_manual]
    bars_gain = ax4.bar(methods_gain, gains, color='#2196f3', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Ganho de Performance (%)', fontsize=11)
    ax4.set_title('Ganho Relativo com Normalização', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # Adicionar valores nas barras
    for bar, gain in zip(bars_gain, gains):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'+{gain:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # RESUMO: Melhores práticas
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 RESUMO: Por que normalizar no KNN?")
    print("=" * 80)

    print("\n❌ SEM normalização:")
    print("   - Features com valores grandes dominam o cálculo de distância")
    print("   - Features importantes com valores pequenos são ignoradas")
    print(f"   - Resultado: Acurácia baixa ({acc_no_norm*100:.1f}%)")

    print("\n✅ COM normalização:")
    print("   - Todas as features têm a mesma escala (média≈0, std≈1)")
    print("   - Distância euclidiana considera todas as features igualmente")
    print(f"   - Resultado: Acurácia alta ({acc_pipeline*100:.1f}%)")
    print(f"   - Ganho: +{gain_pipeline:.1f}% de melhoria!")

    print("\n🔧 MÉTODOS DE NORMALIZAÇÃO:")
    print("   1. Pipeline (RECOMENDADO):")
    print("      - Pipeline([('scaler', StandardScaler()), ('knn', KNN())])")
    print("      - Vantagem: Aplica automaticamente em novos dados")

    print("\n   2. Manual:")
    print("      - scaler.fit_transform(X_train)  # Treino")
    print("      - scaler.transform(X_test)        # Teste")
    print("      - CUIDADO: fit_transform() só no treino!")

    print("\n⚠️  ERRO COMUM:")
    print("   - Usar fit_transform() no teste → DATA LEAKAGE!")
    print("   - Sempre: fit_transform(treino) + transform(teste)")

    print("\n" + "=" * 80)
    print("Teste concluído com sucesso! ✅")
    print("=" * 80)

if __name__ == "__main__":
    main()
