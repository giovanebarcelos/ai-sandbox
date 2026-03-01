# GO1404-1422TextClassificationComL
# ═══════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO DE NOTÍCIAS COM LSTM EMPILHADO
# Dataset: AG News (4 categorias)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Embedding, Dropout, 
                                      Bidirectional, GRU, Input, concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── 1. CRIAR DATASET SINTÉTICO DE NOTÍCIAS ───
print("📰 Criando dataset sintético de notícias...")

np.random.seed(42)

# Templates de notícias por categoria
news_templates = {
    'Sports': [
        "{team1} defeats {team2} in {sport} championship final",
        "{player} scores {goals} goals in match against {opponent}",
        "{country} wins gold medal in {event} at Olympics",
        "Record broken by {athlete} in {sport} competition",
        "{team} signs new contract with star player {player}"
    ],
    'Technology': [
        "New {device} launched with {feature} technology",
        "{company} announces breakthrough in {field} research",
        "Latest {product} update includes {improvement} features",
        "{tech_company} releases new version of {software}",
        "Scientists develop innovative {invention} for {purpose}"
    ],
    'Business': [
        "{company} stock rises {percent}% after earnings report",
        "Merger announced between {corp1} and {corp2}",
        "{industry} sector faces challenges amid economic changes",
        "CEO of {company} announces new strategic direction",
        "{business} reports record profits in Q{quarter}"
    ],
    'Politics': [
        "{politician} announces new {policy} initiative",
        "Government introduces legislation on {topic}",
        "{country1} and {country2} sign trade agreement",
        "Election results show victory for {party} party",
        "Parliament debates controversial {issue} bill"
    ]
}

# Vocabulário para preencher templates
vocab = {
    'team1': ['Lakers', 'Bulls', 'Warriors', 'Celtics', 'Heat'],
    'team2': ['Nets', 'Rockets', 'Nuggets', 'Clippers', 'Spurs'],
    'sport': ['basketball', 'football', 'tennis', 'baseball'],
    'player': ['James', 'Durant', 'Curry', 'Jordan', 'Bryant'],
    'goals': ['2', '3', '4', '5'],
    'opponent': ['rivals', 'challengers', 'competitors', 'adversaries'],
    'country': ['USA', 'China', 'Brazil', 'Germany', 'Japan'],
    'event': ['100m sprint', 'swimming', 'gymnastics', 'marathon'],
    'athlete': ['Johnson', 'Williams', 'Chen', 'Silva', 'Schmidt'],
    'device': ['smartphone', 'tablet', 'laptop', 'smartwatch'],
    'feature': ['AI', '5G', 'quantum', 'holographic'],
    'company': ['TechCorp', 'InnoSoft', 'FutureTech', 'NextGen'],
    'field': ['quantum computing', 'artificial intelligence', 'robotics'],
    'product': ['software', 'app', 'platform', 'system'],
    'improvement': ['security', 'performance', 'user interface'],
    'tech_company': ['Google', 'Apple', 'Microsoft', 'Amazon'],
    'software': ['operating system', 'browser', 'productivity suite'],
    'invention': ['device', 'algorithm', 'material', 'process'],
    'purpose': ['healthcare', 'education', 'transportation', 'communication'],
    'percent': ['5', '10', '15', '20', '25'],
    'corp1': ['MegaCorp', 'GlobalTech', 'BigIndustries'],
    'corp2': ['SmallCo', 'StartupX', 'InnovateNow'],
    'industry': ['automotive', 'pharmaceutical', 'energy', 'retail'],
    'business': ['Retailer', 'Manufacturer', 'Service Provider'],
    'quarter': ['1', '2', '3', '4'],
    'politician': ['Senator Smith', 'Minister Jones', 'President Lee'],
    'policy': ['healthcare', 'education', 'environmental', 'economic'],
    'topic': ['climate change', 'healthcare', 'education', 'taxation'],
    'country1': ['USA', 'EU', 'China', 'India'],
    'country2': ['Japan', 'Brazil', 'Canada', 'Australia'],
    'party': ['Democratic', 'Republican', 'Liberal', 'Conservative'],
    'issue': ['budget', 'reform', 'infrastructure', 'security']
}

# Gerar notícias
def generate_news(n_per_category=500):
    """Gera notícias sintéticas"""
    data = []

    for category, templates in news_templates.items():
        for _ in range(n_per_category):
            template = np.random.choice(templates)

            # Substituir placeholders
            text = template
            for key in vocab:
                if f'{{{key}}}' in text:
                    text = text.replace(f'{{{key}}}', np.random.choice(vocab[key]))

            data.append({'text': text, 'category': category})

    return pd.DataFrame(data)

df = generate_news(n_per_category=500)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"  Total de notícias: {len(df)}")
print(f"  Categorias: {df['category'].unique()}")
print(f"\n  Distribuição:")
print(df['category'].value_counts())

# Exemplos
print(f"\n  Exemplos:")
for category in df['category'].unique()[:2]:
    example = df[df['category'] == category].iloc[0]['text']
    print(f"    {category}: {example}")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados para LSTM...")

# Encodar labels
label_map = {cat: idx for idx, cat in enumerate(df['category'].unique())}
df['label'] = df['category'].map(label_map)

# Tokenização
MAX_WORDS = 5000
MAX_LEN = 20

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label'].values

print(f"  Vocabulário: {len(tokenizer.word_index)} palavras")
print(f"  X shape: {X.shape}")
print(f"  Comprimento máximo: {MAX_LEN}")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\n  Treino: {len(X_train)}")
print(f"  Validação: {len(X_val)}")
print(f"  Teste: {len(X_test)}")

# ─── 3. MODELO 1: LSTM SIMPLES ───
print("\n🔨 Modelo 1: LSTM Simples...")

model_simple = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64),
    Dropout(0.5),
    Dense(4, activation='softmax')
], name='LSTM_Simple')

model_simple.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"  Parâmetros: {model_simple.count_params():,}")

history_simple = model_simple.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=0
)

acc_simple = model_simple.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_simple:.4f}")

# ─── 4. MODELO 2: LSTM EMPILHADO (STACKED) ───
print("\n🔨 Modelo 2: LSTM Empilhado (Stacked)...")

model_stacked = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(4, activation='softmax')
], name='LSTM_Stacked')

model_stacked.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"  Parâmetros: {model_stacked.count_params():,}")

history_stacked = model_stacked.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=0
)

acc_stacked = model_stacked.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_stacked:.4f}")

# ─── 5. MODELO 3: BIDIRECTIONAL LSTM ───
print("\n🔨 Modelo 3: Bidirectional LSTM...")

model_bidir = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(4, activation='softmax')
], name='BiLSTM')

model_bidir.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"  Parâmetros: {model_bidir.count_params():,}")

history_bidir = model_bidir.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=0
)

acc_bidir = model_bidir.evaluate(X_test, y_test, verbose=0)[1]
print(f"  ✓ Accuracy (teste): {acc_bidir:.4f}")

# ─── 6. COMPARAR MODELOS ───
print("\n📊 Comparando modelos...")

comparison = pd.DataFrame({
    'Modelo': ['LSTM Simples', 'LSTM Empilhado', 'Bidirectional LSTM'],
    'Parâmetros': [model_simple.count_params(), model_stacked.count_params(), model_bidir.count_params()],
    'Accuracy': [acc_simple, acc_stacked, acc_bidir]
})

print("\n" + comparison.to_string(index=False))

# Visualizar comparação
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
comparison.plot(x='Modelo', y='Accuracy', kind='bar', ax=axes[0], legend=False, color='skyblue')
axes[0].set_title('Accuracy por Modelo', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticklabels(comparison['Modelo'], rotation=45, ha='right')

# Parâmetros
comparison.plot(x='Modelo', y='Parâmetros', kind='bar', ax=axes[1], legend=False, color='coral')
axes[1].set_title('Número de Parâmetros', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Parâmetros')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticklabels(comparison['Modelo'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('news_classification_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: news_classification_comparison.png")

# ─── 7. ANÁLISE DETALHADA DO MELHOR MODELO ───
best_model = model_bidir if acc_bidir >= max(acc_simple, acc_stacked) else \
             (model_stacked if acc_stacked >= acc_simple else model_simple)
best_name = 'Bidirectional' if acc_bidir >= max(acc_simple, acc_stacked) else \
            ('Stacked' if acc_stacked >= acc_simple else 'Simple')

print(f"\n🏆 Melhor modelo: {best_name} LSTM")

# Predições
y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
categories = list(label_map.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=categories, yticklabels=categories)
plt.title(f'Matriz de Confusão - {best_name} LSTM', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('news_confusion_matrix.png', dpi=150)
print("  ✓ Matriz de confusão salva: news_confusion_matrix.png")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=categories))

# ─── 8. TESTAR COM NOVOS TEXTOS ───
print("\n🧪 Testando com novos textos...")

new_texts = [
    "Lakers win championship against Bulls in exciting finale",
    "Apple releases new iPhone with revolutionary camera technology",
    "Stock market reaches new heights as economy recovers",
    "President announces new climate change initiative"
]

# Processar
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=MAX_LEN, padding='post')

# Prever
new_predictions = best_model.predict(new_padded, verbose=0)
new_pred_classes = np.argmax(new_predictions, axis=1)

print("\n  Previsões:")
for text, pred_idx, probs in zip(new_texts, new_pred_classes, new_predictions):
    category = categories[pred_idx]
    confidence = probs[pred_idx] * 100
    print(f"    \"{text[:50]}...\"")
    print(f"    → {category} (confiança: {confidence:.1f}%)\n")

# ─── 9. RELATÓRIO FINAL ───
print("="*70)
print("✅ CLASSIFICAÇÃO DE NOTÍCIAS CONCLUÍDA!")
print("="*70)

print("\n📊 Resultados:")
print(f"  • LSTM Simples: {acc_simple:.4f}")
print(f"  • LSTM Empilhado: {acc_stacked:.4f}")
print(f"  • Bidirectional LSTM: {acc_bidir:.4f}")
print(f"  • Melhor: {best_name} LSTM ({max(acc_simple, acc_stacked, acc_bidir):.4f})")

print("\n📁 Arquivos gerados:")
print("  • news_classification_comparison.png")
print("  • news_confusion_matrix.png")

print("\n💡 Observações:")
print("  • Bidirectional LSTM geralmente performa melhor em classificação")
print("  • Stacked LSTM útil para capturar hierarquias complexas")
print("  • Trade-off entre complexidade e performance")
