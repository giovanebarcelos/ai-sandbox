# GO1425-36jNamedEntityRecognitionNerLstm
# ═══════════════════════════════════════════════════════════════════
# NAMED ENTITY RECOGNITION (NER) COM LSTM BIDIRECIONAL
# Identificar entidades (pessoas, locais, organizações) em texto
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

print("🏷️ NAMED ENTITY RECOGNITION (NER)")
print("=" * 70)

# ─── 1. CRIAR DATASET SINTÉTICO ───
print("\n📝 Gerando dataset NER sintético...")

# Sentenças de exemplo
sentences = [
    "John works at Google in California",
    "Mary lives in New York",
    "Microsoft was founded by Bill Gates",
    "Tesla is located in Austin Texas",
    "Apple CEO Tim Cook announced new products",
    "Amazon opened offices in Seattle",
    "Jane visited Paris last summer",
    "IBM headquarters are in New York",
    "Steve Jobs founded Apple in California",
    "Facebook was created by Mark Zuckerberg",
] * 50  # Repetir para ter mais dados

# Tags BIO (Begin, Inside, Outside)
# B-PER: Begin Person, I-PER: Inside Person
# B-ORG: Begin Organization, I-ORG: Inside Organization  
# B-LOC: Begin Location, I-LOC: Inside Location
# O: Outside (não é entidade)

labels = [
    ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"],
    ["B-PER", "O", "O", "B-LOC", "I-LOC"],
    ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
    ["B-ORG", "O", "O", "O", "B-LOC", "B-LOC"],
    ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "O"],
    ["B-ORG", "O", "O", "O", "B-LOC"],
    ["B-PER", "O", "B-LOC", "O", "O"],
    ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC"],
    ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],
    ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
] * 50

print(f"  Número de sentenças: {len(sentences)}")

# ─── 2. PREPROCESSAR ───
print("\n🔧 Preprocessando...")

# Tokenizar
words_list = [s.split() for s in sentences]

# Criar vocabulários
all_words = set([word for sent in words_list for word in sent])
all_tags = set([tag for sent_tags in labels for tag in sent_tags])

word2idx = {word: idx+1 for idx, word in enumerate(sorted(all_words))}
word2idx['<PAD>'] = 0
tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

vocab_size = len(word2idx)
num_tags = len(tag2idx)

print(f"  Vocabulário: {vocab_size} palavras")
print(f"  Tags: {num_tags} ({list(tag2idx.keys())})")

# Converter para índices
X_data = [[word2idx[word] for word in sent] for sent in words_list]
y_data = [[tag2idx[tag] for tag in sent_tags] for sent_tags in labels]

# Pad sequences
max_len = max(len(sent) for sent in X_data)
X_data = pad_sequences(X_data, maxlen=max_len, padding='post', value=0)
y_data = pad_sequences(y_data, maxlen=max_len, padding='post', value=tag2idx['O'])

print(f"  Max length: {max_len}")
print(f"  X shape: {X_data.shape}")
print(f"  y shape: {y_data.shape}")

# ─── 3. CONSTRUIR MODELO ───
print("\n🏗️ Construindo modelo BiLSTM...")

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len, mask_zero=True),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    TimeDistributed(Dense(num_tags, activation='softmax'))
], name='NER_BiLSTM')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando modelo...")

# Reshape y para (samples, timesteps, 1)
y_train = y_data.reshape(*y_data.shape, 1)

history = model.fit(
    X_data, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 5. TESTAR ───
print("\n🔮 Testando NER...")

# Sentenças de teste
test_sentences = [
    "Barack Obama visited Microsoft",
    "Google opened offices in London",
    "Elon Musk founded SpaceX in Texas"
]

for test_sent in test_sentences:
    # Tokenizar
    test_words = test_sent.split()
    test_x = [word2idx.get(word, 0) for word in test_words]
    test_x_padded = pad_sequences([test_x], maxlen=max_len, padding='post', value=0)

    # Prever
    pred = model.predict(test_x_padded, verbose=0)[0]
    pred_tags = [idx2tag[np.argmax(pred[i])] for i in range(len(test_words))]

    # Exibir
    print(f"\n  Sentença: {test_sent}")
    for word, tag in zip(test_words, pred_tags):
        if tag != 'O':
            print(f"    {word:15s} → {tag}")

# ─── 6. VISUALIZAR ENTIDADES ───
print("\n📊 Visualizando extração de entidades...")

fig, ax = plt.subplots(figsize=(14, 8))

# Processar várias sentenças
sentences_to_viz = sentences[:10]
y_pos = np.arange(len(sentences_to_viz))

for i, (sent, sent_labels) in enumerate(zip(sentences_to_viz, labels[:10])):
    words = sent.split()

    x_offset = 0
    for word, label in zip(words, sent_labels):
        # Colorir por tipo de entidade
        if label.startswith('B-PER') or label.startswith('I-PER'):
            color = 'lightcoral'
            entity_type = 'PERSON'
        elif label.startswith('B-ORG') or label.startswith('I-ORG'):
            color = 'lightblue'
            entity_type = 'ORG'
        elif label.startswith('B-LOC') or label.startswith('I-LOC'):
            color = 'lightgreen'
            entity_type = 'LOCATION'
        else:
            color = 'white'
            entity_type = None

        # Desenhar palavra
        rect = plt.Rectangle((x_offset, i-0.4), len(word)*0.15, 0.8, 
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        ax.text(x_offset + len(word)*0.075, i, word, 
               ha='center', va='center', fontsize=9, fontweight='bold')

        x_offset += len(word)*0.15 + 0.1

ax.set_ylim(-0.5, len(sentences_to_viz)-0.5)
ax.set_xlim(0, 10)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Sent {i+1}" for i in range(len(sentences_to_viz))])
ax.set_xlabel('Words', fontsize=12)
ax.set_title('Named Entity Recognition - Entity Highlighting', fontsize=14, fontweight='bold')

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightcoral', edgecolor='black', label='PERSON'),
    Patch(facecolor='lightblue', edgecolor='black', label='ORGANIZATION'),
    Patch(facecolor='lightgreen', edgecolor='black', label='LOCATION'),
    Patch(facecolor='white', edgecolor='black', label='Other')
]
ax.legend(handles=legend_elements, loc='upper right')

ax.invert_yaxis()
plt.tight_layout()
plt.savefig('ner_entity_highlighting.png', dpi=150)
print("✅ Highlighting salvo: ner_entity_highlighting.png")

# ─── 7. MÉTRICAS POR ENTIDADE ───
print("\n📊 Estatísticas do dataset...")

entity_counts = {}
for sent_tags in labels:
    for tag in sent_tags:
        if tag != 'O':
            entity_type = tag.split('-')[1]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

print(f"\n  Entidades por tipo:")
for entity_type, count in sorted(entity_counts.items()):
    print(f"    {entity_type:10s}: {count}")

plt.figure(figsize=(10, 6))
plt.bar(entity_counts.keys(), entity_counts.values(), 
       color=['lightcoral', 'lightgreen', 'lightblue'], alpha=0.7)
plt.xlabel('Entity Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Entity Distribution in Dataset', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ner_entity_distribution.png', dpi=150)
print("✅ Distribuição salva: ner_entity_distribution.png")

print("\n💡 NER APPROACHES:")
print("  • Rule-based: Regex, dicionários (baixa cobertura)")
print("  • CRF: Conditional Random Fields (features manuais)")
print("  • BiLSTM-CRF: LSTM + CRF layer (SOTA clássico)")
print("  • BERT-based: Transformers (SOTA atual)")

print("\n📚 DATASETS POPULARES:")
print("  • CoNLL-2003: Inglês, 4 tipos (PER, LOC, ORG, MISC)")
print("  • OntoNotes: 18 tipos, múltiplos domínios")
print("  • WikiNER: Multilíngue, extraído da Wikipedia")

print("\n🎯 APLICAÇÕES:")
print("  • Information Extraction: Extrair fatos estruturados")
print("  • Question Answering: Identificar entidades relevantes")
print("  • Chatbots: Entender intenções do usuário")
print("  • Content Classification: Categorizar por entidades")

print("\n✅ NER COMPLETO!")
