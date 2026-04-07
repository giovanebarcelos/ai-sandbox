# GO0615-25ResumoDaAula
# ═══════════════════════════════════════════════════════════════════
# 📋 RESUMO DA AULA 06 — REGRESSÃO, REGULARIZAÇÃO E DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("📋 RESUMO DA AULA 06")
print("   REGRESSÃO, REGULARIZAÇÃO E DIAGNÓSTICO DE MODELOS")
print("=" * 70)

# ───────────────────────────────────────────────────────────────────
# 1. REGRESSÃO LINEAR SIMPLES E MÚLTIPLA (GO0601)
# ───────────────────────────────────────────────────────────────────
print("""
┌─────────────────────────────────────────────────────────────────┐
│  1. REGRESSÃO LINEAR SIMPLES E MÚLTIPLA                         │
└─────────────────────────────────────────────────────────────────┘
  Objetivo: prever um valor numérico contínuo (ex: preço de imóvel)

  • Regressão Simples:  ŷ = β₀ + β₁·x₁
  • Regressão Múltipla: ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ

  Métricas de Avaliação:
  ┌──────┬─────────────────────────────────────────────────────────┐
  │ MSE  │ Média dos erros quadrados. Penaliza erros grandes.      │
  │      │ Quanto menor, melhor. Unidade: (valor)²                 │
  ├──────┼─────────────────────────────────────────────────────────┤
  │ RMSE │ Raiz do MSE. Erro médio típico na mesma unidade do alvo.│
  │      │ Ex: RMSE=30,11 → modelo erra ~R$ 30.110 por imóvel      │
  ├──────┼─────────────────────────────────────────────────────────┤
  │ MAE  │ Erro absoluto médio. Menos sensível a outliers.         │
  │      │ Ex: MAE=24,18 → erro médio de R$ 24.180 por imóvel      │
  ├──────┼─────────────────────────────────────────────────────────┤
  │ R²   │ % da variância explicada pelo modelo. [0, 1]            │
  │      │ Ex: R²=0,9619 → 96,19% da variância explicada. Excelente│
  └──────┴─────────────────────────────────────────────────────────┘

  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
""")

# ───────────────────────────────────────────────────────────────────
# 2. REGRESSÃO POLINOMIAL — OVERFITTING E UNDERFITTING (GO0602)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  2. REGRESSÃO POLINOMIAL — OVERFITTING E UNDERFITTING           │
└─────────────────────────────────────────────────────────────────┘
  Transforma features para capturar relações não-lineares.
  Ex: x → x, x², x³  (PolynomialFeatures)

  Problema central — Dilema Bias-Variância:
  ┌─────────────────┬────────────────┬────────────────────────────┐
  │ Situação        │ Treino         │ Teste                      │
  ├─────────────────┼────────────────┼────────────────────────────┤
  │ Underfitting    │ Score baixo    │ Score baixo (alto bias)    │
  │ (grau muito baixo)│              │                            │
  ├─────────────────┼────────────────┼────────────────────────────┤
  │ Ponto ideal     │ Score alto     │ Score alto                 │
  ├─────────────────┼────────────────┼────────────────────────────┤
  │ Overfitting     │ Score muito alto│ Score baixo (alta variância)│
  │ (grau muito alto)│               │                            │
  └─────────────────┴────────────────┴────────────────────────────┘

  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.pipeline import make_pipeline
  model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
""")

# ───────────────────────────────────────────────────────────────────
# 3. VALIDAÇÃO CRUZADA (GO0603)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  3. VALIDAÇÃO CRUZADA (Cross-Validation)                        │
└─────────────────────────────────────────────────────────────────┘
  Avalia o modelo em múltiplas divisões dos dados → estimativa
  mais confiável que um único train/test split.

  Técnicas:
  • KFold(n_splits=5)  → divide em 5 partes, treina/avalia 5 vezes
  • cross_val_score    → retorna array de scores por fold
  • cross_validate     → retorna múltiplas métricas ao mesmo tempo

  Resultado: média ± desvio padrão dos scores
  Ex: R² = 0.847 ± 0.023  (modelo estável)
      R² = 0.847 ± 0.150  (modelo instável — investigar)

  from sklearn.model_selection import cross_val_score, KFold
  scores = cross_val_score(model, X, y, cv=5, scoring='r2')
  print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")
""")

# ───────────────────────────────────────────────────────────────────
# 4. REGULARIZAÇÃO (GO0604, GO0605, GO0606)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  4. REGULARIZAÇÃO — RIDGE, LASSO E ELASTIC NET                  │
└─────────────────────────────────────────────────────────────────┘
  Regularização adiciona uma penalidade à função de custo para
  evitar overfitting, controlando o tamanho dos coeficientes.

  ┌─────────────┬──────────────────────────────────────────────────┐
  │ Método      │ Comportamento                                    │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ Ridge (L2)  │ Penaliza Σβ² → encolhe coeficientes, nunca       │
  │             │ zera. Bom quando todas as features importam.     │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ Lasso (L1)  │ Penaliza Σ|β| → pode zerar coeficientes.         │
  │             │ Faz seleção automática de features.              │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ ElasticNet  │ Combina L1 + L2. Parâmetro l1_ratio controla     │
  │ (L1 + L2)   │ o equilíbrio. Mais flexível que os dois sozinhos.│
  └─────────────┴──────────────────────────────────────────────────┘

  Hiperparâmetro alpha (λ): controla a força da regularização.
  Usar RidgeCV / LassoCV / ElasticNetCV para escolher alpha
  automaticamente via validação cruzada.

  from sklearn.linear_model import Ridge, Lasso, ElasticNet
  from sklearn.preprocessing import StandardScaler
  # OBRIGATÓRIO: normalizar features antes de regularizar!
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_train)
  ridge = Ridge(alpha=1.0).fit(X_scaled, y_train)
""")

# ───────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING (GO0607, GO0611, GO0617)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  5. FEATURE ENGINEERING                                         │
└─────────────────────────────────────────────────────────────────┘
  Criar novas features para melhorar a capacidade do modelo.

  Técnicas aplicadas na aula:
  ┌────────────────────────┬───────────────────────────────────────┐
  │ Técnica                │ Exemplo                               │
  ├────────────────────────┼───────────────────────────────────────┤
  │ Features polinomiais   │ x → x, x², x³  (PolynomialFeatures)  │
  │ Interações manuais     │ preco_por_m2 = preco / area           │
  │ Transformações         │ log(area), √(idade)  → normalizar dist│
  │ Box-Cox                │ PowerTransformer (requer valores > 0) │
  │ One-Hot Encoding       │ bairro → bairro_Centro, bairro_Norte  │
  │ Target Encoding        │ média do alvo por categoria           │
  │ Binning                │ area → ['Pequeno','Médio','Grande']   │
  │ Features temporais     │ mês, trimestre, dias_desde_data       │
  └────────────────────────┴───────────────────────────────────────┘

  Regra de ouro: normalizar (StandardScaler) ANTES de passar
  para qualquer modelo que use regularização ou distância.
""")

# ───────────────────────────────────────────────────────────────────
# 6. ANÁLISE DE RESÍDUOS (GO0608)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  6. ANÁLISE DE RESÍDUOS                                         │
└─────────────────────────────────────────────────────────────────┘
  Resíduo = valor_real − valor_predito
  Um bom modelo deve ter resíduos com comportamento aleatório.

  Plots de diagnóstico:
  • Resíduos vs Preditos  → deve ser ruído aleatório em torno de 0
  • Histograma            → distribuição deve ser aprox. normal
  • Q-Q Plot              → pontos sobre a diagonal = normalidade
  • Real vs Predito       → pontos próximos da linha y = x

  Testes estatísticos:
  • Shapiro-Wilk  → testa normalidade dos resíduos (p > 0,05 = ok)
  • Levene        → testa homocedasticidade (p > 0,05 = variância constante)

  Problemas comuns:
  • Padrão em funil → heterocedasticidade (transformar alvo com log)
  • Padrão em curva → relação não-linear não capturada (usar polinomial)
  • Outliers graves → investigar dados ou usar modelos robustos
""")

# ───────────────────────────────────────────────────────────────────
# 7. ATIVIDADE PRÁTICA: PREVISÃO DE PREÇOS (GO0609–GO0614)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  7. ATIVIDADE PRÁTICA — PIPELINE COMPLETO DE REGRESSÃO          │
└─────────────────────────────────────────────────────────────────┘
  Pipeline aplicado ao problema de previsão de preços de imóveis:

  Etapa 1 — EDA (GO0610)
    • Visualizar distribuições, correlações, outliers
    • Entender quais features têm relação com o alvo

  Etapa 2 — Feature Engineering (GO0611)
    • Criar/transformar features com base nos insights do EDA

  Etapa 3 — Treinar modelos (GO0612)
    • LinearRegression, Ridge, Lasso, ElasticNet

  Etapa 4 — Analisar o melhor modelo (GO0613)
    • Comparar R², RMSE, MAE em treino e teste
    • Verificar coeficientes e importância de features

  Etapa 5 — Validação cruzada com múltiplas métricas (GO0614)
    • cross_validate com scoring=['r2', 'neg_rmse', 'neg_mae']
    • Confirmar estabilidade do modelo
""")

# ───────────────────────────────────────────────────────────────────
# 8. TÉCNICAS PARA MELHORAR PERFORMANCE (GO0617–GO0621)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  8. TÉCNICAS PARA MELHORAR PERFORMANCE                          │
└─────────────────────────────────────────────────────────────────┘
  Coeficientes Padronizados (GO0618):
  • Normalizar X antes de treinar → coeficientes comparáveis
  • Coeficiente maior (em módulo) = feature mais influente
  • Visualizar com gráfico de barras horizontais (GO0619)

  Tabela Comparativa de Modelos (GO0620):
  • Comparar LinearRegression, Ridge, Lasso, ElasticNet
  • Colunas: R²_treino, R²_teste, RMSE_teste, n_features_ativas

  Learning Curves (GO0621):
  • Plota score de treino e validação vs quantidade de dados
  • Diagnóstico automático:
    ┌──────────────────────────────────────────────────────────┐
    │ Gap grande (treino >> val)  → Overfitting (alta variância)│
    │ Ambos baixos               → Underfitting (alto bias)    │
    │ Curva val ainda subindo    → Mais dados podem ajudar     │
    │ Convergência com gap pequeno → Modelo balanceado ✅      │
    └──────────────────────────────────────────────────────────┘

  from sklearn.model_selection import learning_curve
""")

# ───────────────────────────────────────────────────────────────────
# 9. PRÓXIMA AULA (GO0622)
# ───────────────────────────────────────────────────────────────────
print("""┌─────────────────────────────────────────────────────────────────┐
│  9. PRÓXIMA AULA — AULA 07                                      │
└─────────────────────────────────────────────────────────────────┘
  Tema: Clustering — Aprendizado Não-supervisionado
  • Quando não há rótulo (y) nos dados
  • Algoritmos: K-Means, DBSCAN, Hierarchical Clustering
  • Métricas: Silhouette Score, Inertia

  Atenção: R² negativo (ex: -0,5) indica modelo pior que
  simplesmente prever a média — investigar antes de usar.
""")

# ───────────────────────────────────────────────────────────────────
# QUADRO COMPARATIVO FINAL
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("QUADRO COMPARATIVO — MODELOS DE REGRESSÃO DA AULA 06")
print("=" * 70)
print(f"{'Modelo':<20} {'Regularização':<15} {'Selec. Features':<17} {'Quando usar'}")
print("-" * 70)
modelos = [
    ("Linear Simples",    "Nenhuma",     "Não",     "Baseline, relação linear clara"),
    ("Linear Múltipla",   "Nenhuma",     "Não",     "Múltiplas features, sem overfitting"),
    ("Polinomial",        "Nenhuma",     "Não",     "Relação não-linear entre x e y"),
    ("Ridge (L2)",        "L2 (Σβ²)",    "Não",     "Multicolinearidade, todas features importam"),
    ("Lasso (L1)",        "L1 (Σ|β|)",   "Sim",     "Muitas features, quer esparsidade"),
    ("Elastic Net",       "L1 + L2",     "Parcial", "Combinar vantagens de Ridge e Lasso"),
]
for nome, reg, sel, uso in modelos:
    print(f"  {nome:<18} {reg:<15} {sel:<17} {uso}")
print("=" * 70)
print("\n✅ Fim do Resumo da Aula 06")
