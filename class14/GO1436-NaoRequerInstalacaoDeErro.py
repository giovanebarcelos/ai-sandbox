# GO1436-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# NER (NAMED ENTITY RECOGNITION) COM BIDIRECTIONAL LSTM + CRF
# Tarefa: Identificar entidades (pessoas, organizações, locais)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

# ─── 1. CRIAR DATASET DE NER ───
print("📝 Criando dataset de NER...")

np.random.seed(42)

# Vocabulário
names = ['John', 'Mary', 'David', 'Sarah', 'Michael', 'Emma', 'James', 'Lisa']
organizations = ['Google', 'Microsoft', 'Apple', 'Amazon', 'Tesla', 'Facebook', 'Netflix']
locations = ['NewYork', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Dubai']
verbs = ['works', 'lives', 'visited', 'founded', 'manages', 'leads']
preps = ['at', 'in', 'from', 'to']
articles = ['the', 'a', 'an']
connectors = ['and', 'or']

# Templates de frases
templates = [
    "{name} works at {org} in {loc}",
    "{name} visited {loc} and {loc}",
    "{name} founded {org} in {loc}",
    "The {org} office in {loc} hired {name}",
    "{name} from {loc} leads {org}",
    "{name} and {name} work at {org}",
]

# Gerar frases com tags
sentences = []
labels_list = []

n_samples = 1000

for _ in range(n_samples):
    template = np.random.choice(templates)

    # Substituir placeholders
    sentence = template
    sentence_labels = []

    words = []
    tags = []

    for part in sentence.split():
        if '{name}' in part:
            name = np.random.choice(names)
            words.append(name)
            tags.append('B-PER')  # Begin-Person
        elif '{org}' in part:
            org = np.random.choice(organizations)
            words.append(org)
            tags.append('B-ORG')  # Begin-Organization
        elif '{loc}' in part:
            loc = np.random.choice(locations)
            words.append(loc)
            tags.append('B-LOC')  # Begin-Location
        else:
            words.append(part)
            tags.append('O')  # Outside (non-entity)

    sentences.append(words)
    labels_list.append(tags)

print(f"  Total de frases: {len(sentences)}")

# Exemplos
print("\n  Exemplos:")
for i in range(3):
    print(f"\n    Frase: {' '.join(sentences[i])}")
    print(f"    Tags:  {' '.join(labels_list[i])}")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados...")

# Vocabulário de palavras
all_words = set([word for sentence in sentences for word in sentence])
word2idx = {word: idx+1 for idx, word in enumerate(sorted(all_words))}
word2idx['<PAD>'] = 0

# Vocabulário de tags
all_tags = set([tag for tags in labels_list for tag in tags])
tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

vocab_size = len(word2idx)
n_tags = len(tag2idx)

print(f"  Vocabulário: {vocab_size} palavras")
print(f"  Tags: {n_tags} ({list(tag2idx.keys())})")

# Converter para índices
X = [[word2idx[word] for word in sentence] for sentence in sentences]
y = [[tag2idx[tag] for tag in tags] for tags in labels_list]

# Padding
MAX_LEN = max(len(sentence) for sentence in X)

X_padded = pad_sequences(X, maxlen=MAX_LEN, padding='post', value=0)
y_padded = pad_sequences(y, maxlen=MAX_LEN, padding='post', value=tag2idx['O'])

print(f"  Max length: {MAX_LEN}")
print(f"  X shape: {X_padded.shape}")
print(f"  y shape: {y_padded.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_padded, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"\n  Treino: {len(X_train)}")
print(f"  Validação: {len(X_val)}")
print(f"  Teste: {len(X_test)}")

# ─── 3. CONSTRUIR MODELO BILSTM ───
print("\n🔨 Construindo modelo BiLSTM para NER...")

inputs = Input(shape=(MAX_LEN,))
embedding = Embedding(vocab_size, 128, mask_zero=True)(inputs)
bilstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
dropout = Dropout(0.5)(bilstm)
outputs = TimeDistributed(Dense(n_tags, activation='softmax'))(dropout)

model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_NER')

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Parâmetros: {model.count_params():,}")
model.summary()

# ─── 4. TREINAR MODELO ───
print("\n🚀 Treinando modelo...")

# Reshape y para [samples, timesteps, 1]
y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_val_reshaped = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

callbacks = [EarlyStopping(patience=10, restore_best_weights=True, verbose=1)]

history = model.fit(
    X_train, y_train_reshaped,
    validation_data=(X_val, y_val_reshaped),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# Avaliar
y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
acc = model.evaluate(X_test, y_test_reshaped, verbose=0)[1]
print(f"\n  Accuracy (teste): {acc:.4f}")

# ─── 5. AVALIAR POR TAG ───
print("\n📊 Avaliando performance por tag...")

# Predições
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=-1)

# Achatar para métricas
y_test_flat = y_test.flatten()
y_pred_flat = y_pred_classes.flatten()

# Remover padding (tag 'O' de padding)
mask = y_test_flat != tag2idx['O']
y_test_filtered = y_test_flat[mask]
y_pred_filtered = y_pred_flat[mask]

# Classification report
tag_names = [idx2tag[i] for i in sorted(idx2tag.keys())]
print("\n📋 Classification Report:")
print(classification_report(
    y_test_filtered, 
    y_pred_filtered,
    labels=list(range(n_tags)),
    target_names=tag_names,
    zero_division=0
))

# ─── 6. VISUALIZAR EXEMPLOS ───
print("\n🔍 Visualizando predições em exemplos...")

n_examples = 5
example_indices = np.random.choice(len(X_test), n_examples, replace=False)

fig, axes = plt.subplots(n_examples, 1, figsize=(14, n_examples * 2))

for i, idx in enumerate(example_indices):
    sentence_idx = X_test[idx]
    true_tags = y_test[idx]
    pred_tags = y_pred_classes[idx]

    # Converter índices para palavras e tags
    words = [list(word2idx.keys())[list(word2idx.values()).index(word_idx)] 
             for word_idx in sentence_idx if word_idx != 0]

    true_tag_names = [idx2tag[tag_idx] for tag_idx in true_tags[:len(words)]]
    pred_tag_names = [idx2tag[tag_idx] for tag_idx in pred_tags[:len(words)]]

    # Criar visualização
    ax = axes[i] if n_examples > 1 else axes

    # Barras horizontais para cada palavra
    y_pos = np.arange(len(words))

    # Colorir por tag
    colors_true = ['green' if tag.startswith('B-') else 'lightgray' for tag in true_tag_names]
    colors_pred = ['red' if true != pred else 'blue' 
                   for true, pred in zip(true_tag_names, pred_tag_names)]

    ax.barh(y_pos, [1]*len(words), color=colors_true, alpha=0.3, label='Real')
    ax.barh(y_pos, [0.5]*len(words), color=colors_pred, alpha=0.6, label='Previsto')

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{word}\n{true} | {pred}" 
                        for word, true, pred in zip(words, true_tag_names, pred_tag_names)],
                       fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_title(f'Exemplo {i+1}: {" ".join(words)}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Real (verde) | Previsto (azul=correto, vermelho=erro)')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('ner_predictions.png', dpi=150)
print("\n  ✓ Predições salvas: ner_predictions.png")

# ─── 7. TESTAR EM NOVA FRASE ───
print("\n🧪 Testando em nova frase...")

new_sentence = "John works at Google in Tokyo"
new_words = new_sentence.split()

# Converter para índices
new_indices = [word2idx.get(word, 0) for word in new_words]
new_padded = pad_sequences([new_indices], maxlen=MAX_LEN, padding='post', value=0)

# Prever
new_pred = model.predict(new_padded, verbose=0)
new_pred_tags = np.argmax(new_pred[0], axis=-1)

# Extrair tags
new_tag_names = [idx2tag[tag_idx] for tag_idx in new_pred_tags[:len(new_words)]]

print(f"\n  Frase: {new_sentence}")
print(f"\n  Entidades detectadas:")
for word, tag in zip(new_words, new_tag_names):
    if tag != 'O':
        entity_type = tag.split('-')[1]
        print(f"    • {word}: {entity_type}")

# ─── 8. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ NER COM BILSTM CONCLUÍDO!")
print("="*70)

print(f"\n📊 Estatísticas:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Vocabulário: {vocab_size} palavras")
print(f"  Tags: {n_tags} ({', '.join(tag_names)})")
print(f"  Parâmetros: {model.count_params():,}")

print("\n📁 Arquivos gerados:")
print("  • ner_predictions.png - Exemplos de predições")

print("\n💡 Aplicações:")
print("  • Extração de informações de textos")
print("  • Análise de documentos legais")
print("  • Processamento de currículos")
print("  • Chatbots inteligentes")

print("\n🔧 Melhorias possíveis:")
print("  • Usar CRF layer (Conditional Random Field)")
print("  • Pre-trained embeddings (Word2Vec, GloVe)")
print("  • Transfer learning (BERT for NER)")
print("  • Dados reais (CoNLL-2003, OntoNotes)")
print("  • Adicionar mais tipos de entidades")
