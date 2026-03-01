# GO1412-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# MULTI-TASK LEARNING: SENTIMENTO + CATEGORIZAÇÃO SIMULTÂNEA
# Arquitetura: Encoder compartilhado → Múltiplos decoders
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── 1. CRIAR DATASET MULTI-TASK ───
print("📝 Criando dataset com múltiplas tarefas...")

np.random.seed(42)

# Templates por categoria e sentimento
templates = {
    ('Sports', 'Positive'): [
        "{team} wins championship in incredible match",
        "Amazing performance by {player} leads to victory",
        "{athlete} breaks world record in spectacular fashion",
        "Fans celebrate historic {sport} triumph"
    ],
    ('Sports', 'Negative'): [
        "{team} suffers devastating loss",
        "Disappointing performance costs {sport} team the game",
        "{player} injured in unfortunate accident",
        "Fans disappointed after another defeat"
    ],
    ('Technology', 'Positive'): [
        "Revolutionary {product} launches with amazing features",
        "{company} unveils groundbreaking {technology}",
        "New {device} receives excellent reviews",
        "Innovative solution transforms {industry}"
    ],
    ('Technology', 'Negative'): [
        "{product} faces serious security issues",
        "{company} recalls defective {device}",
        "Major bug discovered in popular {software}",
        "Users complain about terrible {app} update"
    ],
    ('Business', 'Positive'): [
        "{company} reports record-breaking profits",
        "Stock market celebrates {corp} success",
        "Successful merger creates industry leader",
        "{business} expands with new partnerships"
    ],
    ('Business', 'Negative'): [
        "{company} stock plummets after poor earnings",
        "Bankruptcy threatens struggling {business}",
        "Economic downturn hits {industry} hard",
        "Massive layoffs announced at {corp}"
    ]
}

vocab = {
    'team': ['Lakers', 'Warriors', 'Bulls', 'Heat'],
    'player': ['James', 'Curry', 'Durant', 'Jordan'],
    'athlete': ['Johnson', 'Williams', 'Smith'],
    'sport': ['basketball', 'football', 'tennis'],
    'product': ['smartphone', 'laptop', 'tablet'],
    'company': ['TechCorp', 'InnoSoft', 'FutureTech'],
    'technology': ['AI system', '5G network', 'quantum computer'],
    'device': ['phone', 'computer', 'smartwatch'],
    'industry': ['healthcare', 'education', 'finance'],
    'software': ['app', 'program', 'platform'],
    'app': ['social media', 'messaging', 'streaming'],
    'corp': ['MegaCorp', 'GlobalTech'],
    'business': ['retailer', 'startup', 'manufacturer']
}

# Gerar dados
n_per_combination = 100
texts = []
categories = []
sentiments = []

for (category, sentiment), template_list in templates.items():
    for _ in range(n_per_combination):
        template = np.random.choice(template_list)

        # Substituir placeholders
        text = template
        for key in vocab:
            if f'{{{key}}}' in text:
                text = text.replace(f'{{{key}}}', np.random.choice(vocab[key]))

        texts.append(text.lower())
        categories.append(category)
        sentiments.append(sentiment)

# Criar DataFrame
df = pd.DataFrame({
    'text': texts,
    'category': categories,
    'sentiment': sentiments
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Total de textos: {len(df)}")
print(f"  Categorias: {df['category'].unique()}")
print(f"  Sentimentos: {df['sentiment'].unique()}")

print("\n  Distribuição:")
print(df.groupby(['category', 'sentiment']).size())

# Exemplos
print("\n  Exemplos:")
for i in range(3):
    print(f"\n    Texto: {df.iloc[i]['text']}")
    print(f"    Categoria: {df.iloc[i]['category']}")
    print(f"    Sentimento: {df.iloc[i]['sentiment']}")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados...")

# Encodar labels
category_map = {cat: idx for idx, cat in enumerate(df['category'].unique())}
sentiment_map = {sent: idx for idx, sent in enumerate(df['sentiment'].unique())}

df['category_label'] = df['category'].map(category_map)
df['sentiment_label'] = df['sentiment'].map(sentiment_map)

# Tokenização
MAX_WORDS = 2000
MAX_LEN = 20

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_LEN, padding='post')

y_category = df['category_label'].values
y_sentiment = df['sentiment_label'].values

print(f"  Vocabulário: {len(tokenizer.word_index)} palavras")
print(f"  X shape: {X.shape}")
print(f"  Categorias: {len(category_map)} classes")
print(f"  Sentimentos: {len(sentiment_map)} classes")

# Split
X_train, X_test, y_cat_train, y_cat_test, y_sent_train, y_sent_test = train_test_split(
    X, y_category, y_sentiment, test_size=0.2, random_state=42
)

X_train, X_val, y_cat_train, y_cat_val, y_sent_train, y_sent_val = train_test_split(
    X_train, y_cat_train, y_sent_train, test_size=0.2, random_state=42
)

print(f"\n  Treino: {len(X_train)}")
print(f"  Validação: {len(X_val)}")
print(f"  Teste: {len(X_test)}")

# ─── 3. MODELO SINGLE-TASK (BASELINE) ───
print("\n🔨 Modelo 1: Single-Task (apenas categoria)...")

from tensorflow.keras.models import Sequential

model_single = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64),
    Dropout(0.5),
    Dense(len(category_map), activation='softmax')
], name='Single_Task')

model_single.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_single = model_single.fit(
    X_train, y_cat_train,
    validation_data=(X_val, y_cat_val),
    epochs=20,
    batch_size=32,
    verbose=0
)

acc_single = model_single.evaluate(X_test, y_cat_test, verbose=0)[1]
print(f"  ✓ Accuracy categoria (teste): {acc_single:.4f}")

# ─── 4. MODELO MULTI-TASK ───
print("\n🔨 Modelo 2: Multi-Task (categoria + sentimento)...")

# Encoder compartilhado
inputs = Input(shape=(MAX_LEN,))
embedding = Embedding(MAX_WORDS, 128)(inputs)
shared_lstm = LSTM(64)(embedding)

# Task 1: Categoria
category_branch = Dense(32, activation='relu', name='category_dense')(shared_lstm)
category_branch = Dropout(0.5)(category_branch)
category_output = Dense(len(category_map), activation='softmax', name='category_output')(category_branch)

# Task 2: Sentimento
sentiment_branch = Dense(32, activation='relu', name='sentiment_dense')(shared_lstm)
sentiment_branch = Dropout(0.5)(sentiment_branch)
sentiment_output = Dense(len(sentiment_map), activation='softmax', name='sentiment_output')(sentiment_branch)

# Modelo multi-task
model_multi = Model(inputs=inputs, outputs=[category_output, sentiment_output], name='Multi_Task')

model_multi.compile(
    optimizer='adam',
    loss={
        'category_output': 'sparse_categorical_crossentropy',
        'sentiment_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'category_output': 'accuracy',
        'sentiment_output': 'accuracy'
    }
)

print(f"  Parâmetros: {model_multi.count_params():,}")
model_multi.summary()

# Treinar
history_multi = model_multi.fit(
    X_train,
    {'category_output': y_cat_train, 'sentiment_output': y_sent_train},
    validation_data=(X_val, {'category_output': y_cat_val, 'sentiment_output': y_sent_val}),
    epochs=30,
    batch_size=32,
    verbose=1
)

# Avaliar
results = model_multi.evaluate(
    X_test,
    {'category_output': y_cat_test, 'sentiment_output': y_sent_test},
    verbose=0
)

acc_cat_multi = results[3]  # category_output_accuracy
acc_sent_multi = results[4]  # sentiment_output_accuracy

print(f"\n  ✓ Accuracy categoria (teste): {acc_cat_multi:.4f}")
print(f"  ✓ Accuracy sentimento (teste): {acc_sent_multi:.4f}")

# ─── 5. COMPARAR MODELOS ───
print("\n📊 Comparação Single-Task vs Multi-Task:")

comparison = pd.DataFrame({
    'Modelo': ['Single-Task (Categoria)', 'Multi-Task (Categoria)', 'Multi-Task (Sentimento)'],
    'Accuracy': [acc_single, acc_cat_multi, acc_sent_multi]
})

print("\n" + comparison.to_string(index=False))

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
comparison.plot(x='Modelo', y='Accuracy', kind='bar', ax=axes[0], legend=False, color='skyblue')
axes[0].set_title('Comparação de Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticklabels(comparison['Modelo'], rotation=45, ha='right')

# Training history (multi-task)
axes[1].plot(history_multi.history['category_output_accuracy'], label='Categoria (treino)', linewidth=2)
axes[1].plot(history_multi.history['val_category_output_accuracy'], label='Categoria (val)', linewidth=2)
axes[1].plot(history_multi.history['sentiment_output_accuracy'], label='Sentimento (treino)', linewidth=2, linestyle='--')
axes[1].plot(history_multi.history['val_sentiment_output_accuracy'], label='Sentimento (val)', linewidth=2, linestyle='--')
axes[1].set_title('Treinamento Multi-Task', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multitask_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: multitask_comparison.png")

# ─── 6. PREDIÇÕES EM NOVOS TEXTOS ───
print("\n🧪 Testando com novos textos...")

new_texts = [
    "lakers wins championship in amazing match",
    "smartphone faces terrible security problems",
    "company reports disappointing quarterly results"
]

# Processar
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=MAX_LEN, padding='post')

# Prever
predictions = model_multi.predict(new_padded, verbose=0)
pred_categories = np.argmax(predictions[0], axis=1)
pred_sentiments = np.argmax(predictions[1], axis=1)

# Mapas reversos
category_reverse = {v: k for k, v in category_map.items()}
sentiment_reverse = {v: k for k, v in sentiment_map.items()}

print("\n  Previsões:")
for text, cat_idx, sent_idx in zip(new_texts, pred_categories, pred_sentiments):
    category = category_reverse[cat_idx]
    sentiment = sentiment_reverse[sent_idx]

    print(f"\n    Texto: \"{text}\"")
    print(f"    → Categoria: {category}")
    print(f"    → Sentimento: {sentiment}")

# ─── 7. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ MULTI-TASK LEARNING CONCLUÍDO!")
print("="*70)

print(f"\n📊 Resultados:")
print(f"  • Single-Task (Categoria): {acc_single:.4f}")
print(f"  • Multi-Task (Categoria): {acc_cat_multi:.4f}")
print(f"  • Multi-Task (Sentimento): {acc_sent_multi:.4f}")

print(f"\n💡 Benefícios Multi-Task Learning:")
print(f"  • Compartilhamento de representações")
print(f"  • Regularização implícita")
print(f"  • Melhor generalização")
print(f"  • Eficiência computacional (um modelo, múltiplas tarefas)")

print("\n📁 Arquivos gerados:")
print("  • multitask_comparison.png - Comparação de modelos")

print("\n🔧 Aplicações:")
print("  • Análise de sentimento + classificação tópicos")
print("  • Detecção de spam + priorização de emails")
print("  • Reconhecimento de ações + localização em vídeo")
print("  • Tradução + análise de sentimento")
