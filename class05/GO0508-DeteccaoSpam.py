# GO0508-DetecçãoSpam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# 1. DATASET DE E-MAILS (SIMULADO)
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("SISTEMA DE DETECÇÃO DE SPAM - CLASSIFICAÇÃO DE TEXTOS")
print("="*70)

# Criar dataset simulado de e-mails
emails_spam = [
    "Ganhe dinheiro rápido! Clique aqui agora!!!",
    "Você ganhou um prêmio de 1 milhão! Resgate já!",
    "PROMOÇÃO IMPERDÍVEL! Desconto de 90% hoje!",
    "Clique aqui para ganhar um iPhone grátis",
    "Oportunidade única! Trabalhe de casa e ganhe muito",
    "Compre agora com 99% de desconto!!!",
    "Parabéns! Você foi selecionado para ganhar",
    "Dinheiro fácil! Sem esforço! Clique aqui!",
    "Remédio milagroso! Perca 20kg em 1 semana",
    "URGENTE: Sua conta será bloqueada! Clique aqui",
    "Ganhe R$10.000 trabalhando 2 horas por dia",
    "ÚLTIMO DIA de promoção! Não perca!",
    "Você tem uma mensagem importante! Abra agora!",
    "Compre réplicas de relógios de luxo baratos",
    "Aumente sua renda trabalhando online agora"
]

emails_ham = [
    "Reunião de equipe amanhã às 14h na sala 3",
    "Segue em anexo o relatório mensal solicitado",
    "Confirme sua presença no evento de sexta-feira",
    "Obrigado pelo feedback sobre o projeto",
    "Vamos almoçar juntos na próxima semana?",
    "O documento foi revisado e está pronto",
    "Parabéns pelo excelente trabalho apresentado",
    "Podemos marcar uma call para discutir o orçamento?",
    "A entrega do material está prevista para terça",
    "Boa tarde, seguem as informações solicitadas",
    "Reunião cancelada, reagendaremos em breve",
    "Seu pedido foi enviado e chegará em 3 dias",
    "Obrigado por participar da pesquisa de satisfação",
    "Lembrete: prazo de entrega do relatório é sexta",
    "Compartilho o link da apresentação de ontem"
]

# Criar DataFrame
data = {
    'email': emails_spam + emails_ham,
    'label': ['spam']*len(emails_spam) + ['ham']*len(emails_ham)
}
df = pd.DataFrame(data)

print(f"\n📊 Dataset criado: {len(df)} e-mails")
print(f"   • Spam: {sum(df['label']=='spam')} e-mails")
print(f"   • Ham: {sum(df['label']=='ham')} e-mails")

# ═══════════════════════════════════════════════════════════════════
# 2. PRÉ-PROCESSAMENTO E VETORIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("VETORIZAÇÃO COM TF-IDF")
print("="*70)

# Converter texto em features numéricas usando TF-IDF
vectorizer = TfidfVectorizer(
    max_features=100,      # Top 100 palavras mais importantes
    stop_words='english',  # Remover palavras comuns
    ngram_range=(1, 2)     # Unigramas e bigramas
)

X = vectorizer.fit_transform(df['email'])
y = df['label']

print(f"\n✅ Vetorização concluída:")
print(f"   • Shape: {X.shape}")
print(f"   • Features: {len(vectorizer.get_feature_names_out())} palavras/termos")

# Mostrar top 10 features mais importantes
feature_names = vectorizer.get_feature_names_out()
print(f"\n📝 Top 10 termos identificados:")
for i, term in enumerate(feature_names[:10]):
    print(f"   {i+1}. {term}")

# ═══════════════════════════════════════════════════════════════════
# 3. DIVISÃO TREINO/TESTE
# ═══════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Divisão dos dados:")
print(f"   • Treino: {X_train.shape[0]} e-mails")
print(f"   • Teste: {X_test.shape[0]} e-mails")

# ═══════════════════════════════════════════════════════════════════
# 4. TREINAMENTO DE MÚLTIPLOS CLASSIFICADORES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TREINAMENTO E COMPARAÇÃO DE MODELOS")
print("="*70)

# Modelos a serem testados
modelos = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

resultados = {}

for nome, modelo in modelos.items():
    print(f"\n🔄 Treinando {nome}...")

    # Treinar
    modelo.fit(X_train, y_train)

    # Validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')

    # Predições
    y_pred = modelo.predict(X_test)

    # Salvar resultados
    resultados[nome] = {
        'modelo': modelo,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred
    }

    print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ═══════════════════════════════════════════════════════════════════
# 5. AVALIAÇÃO DETALHADA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("MÉTRICAS DE DESEMPENHO")
print("="*70)

for nome, resultado in resultados.items():
    print(f"\n📊 {nome}:")
    print(classification_report(y_test, resultado['y_pred'], 
                                target_names=['ham', 'spam']))

# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n🎨 Gerando visualizações...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análise Comparativa - Detecção de Spam', fontsize=16, fontweight='bold')

# 6.1 Matriz de Confusão para cada modelo
for idx, (nome, resultado) in enumerate(resultados.items()):
    cm = confusion_matrix(y_test, resultado['y_pred'], labels=['ham', 'spam'])

    ax = axes[0, idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    ax.set_title(f'{nome}\nConfusion Matrix')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predito')

# 6.2 Comparação de Acurácias
ax = axes[1, 0]
nomes = list(resultados.keys())
acuracias = [res['cv_mean'] for res in resultados.values()]
stds = [res['cv_std'] for res in resultados.values()]

bars = ax.bar(nomes, acuracias, yerr=stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_ylabel('Acurácia (CV)')
ax.set_title('Comparação de Modelos')
ax.set_ylim([0, 1.1])
ax.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
ax.legend()

# Adicionar valores nas barras
for bar, acc in zip(bars, acuracias):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

# 6.3 Curvas ROC
ax = axes[1, 1]
for nome, resultado in resultados.items():
    modelo = resultado['modelo']

    # Probabilidades
    if hasattr(modelo, 'predict_proba'):
        y_proba = modelo.predict_proba(X_test)[:, 1]
    else:
        y_proba = modelo.decision_function(X_test)

    # Calcular ROC
    y_test_binary = (y_test == 'spam').astype(int)
    fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curvas ROC')
ax.legend()
ax.grid(True, alpha=0.3)

# 6.4 Distribuição de Features (TF-IDF)
ax = axes[1, 2]
# Pegar top 15 features mais importantes
feature_importance = np.asarray(X.mean(axis=0)).ravel()
top_indices = feature_importance.argsort()[-15:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_values = feature_importance[top_indices]

ax.barh(range(len(top_features)), top_values, color='skyblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features)
ax.set_xlabel('TF-IDF Score Médio')
ax.set_title('Top 15 Features Mais Importantes')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 7. TESTE INTERATIVO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTE INTERATIVO")
print("="*70)

def classificar_email(texto, modelo_nome='Naive Bayes'):
    """Classificar novo e-mail"""
    modelo = resultados[modelo_nome]['modelo']

    # Vetorizar
    texto_vec = vectorizer.transform([texto])

    # Predição
    predicao = modelo.predict(texto_vec)[0]

    # Probabilidade (se disponível)
    if hasattr(modelo, 'predict_proba'):
        proba = modelo.predict_proba(texto_vec)[0]
        confianca = max(proba)
    else:
        confianca = 1.0

    return predicao, confianca

# Testar com novos e-mails
novos_emails = [
    "URGENTE: Ganhe 100 mil reais agora! Clique aqui!!!",
    "Oi! Podemos marcar aquela reunião para amanhã?",
    "Compre já! Última chance de desconto!"
]

print("\n🧪 Testando novos e-mails:\n")
for email in novos_emails:
    pred, conf = classificar_email(email)
    emoji = "🚫" if pred == 'spam' else "✅"
    print(f"{emoji} '{email[:50]}...'")
    print(f"   → Classificação: {pred.upper()} (confiança: {conf:.2%})\n")

# ═══════════════════════════════════════════════════════════════════
# 8. CONCLUSÕES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONCLUSÕES E RECOMENDAÇÕES")
print("="*70)

melhor_modelo = max(resultados.items(), key=lambda x: x[1]['cv_mean'])
print(f"\n🏆 Melhor modelo: {melhor_modelo[0]}")
print(f"   • Acurácia: {melhor_modelo[1]['cv_mean']:.4f}")
print(f"   • Std: ±{melhor_modelo[1]['cv_std']:.4f}")

print("\n💡 INSIGHTS:")
print("   • Naive Bayes é eficiente para classificação de texto")
print("   • TF-IDF captura bem a importância dos termos")
print("   • Palavras como 'ganhe', 'clique', 'urgente' indicam spam")
print("   • Dataset pequeno: resultados variam com mais dados")

print("\n🔧 MELHORIAS POSSÍVEIS:")
print("   • Aumentar dataset com mais exemplos reais")
print("   • Testar outros ngram_range (1,3) para trigramas")
print("   • Adicionar features de metadados (remetente, hora)")
print("   • Implementar ensemble methods (voting, stacking)")
print("   • Usar embeddings (Word2Vec, BERT) para capturar semântica")

print("\n" + "="*70)
print("FIM DO EXERCÍCIO 1")
print("="*70)
