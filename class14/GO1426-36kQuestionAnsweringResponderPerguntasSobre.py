# GO1426-36kQuestionAnsweringResponderPerguntasSobre
# ══════════════════════════════════════════════════════════════════
# QUESTION ANSWERING COM LSTM
# Responder perguntas baseadas em contexto
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

print("❓ QUESTION ANSWERING COM LSTM")
print("=" * 70)

# ─── 1. CRIAR DATASET SINTÉTICO ───
print("\n📚 Criando dataset de QA...")

# Dataset simples: contextos e perguntas
data = [
    {
        'context': 'John lives in New York',
        'question': 'Where does John live',
        'answer': 'New York'
    },
    {
        'context': 'Mary is a doctor',
        'question': 'What is Mary',
        'answer': 'doctor'
    },
    {
        'context': 'The cat is black',
        'question': 'What color is the cat',
        'answer': 'black'
    },
    {
        'context': 'Python was created in 1991',
        'question': 'When was Python created',
        'answer': '1991'
    },
    {
        'context': 'The book costs ten dollars',
        'question': 'How much does the book cost',
        'answer': 'ten dollars'
    },
    {
        'context': 'Sarah works at Google',
        'question': 'Where does Sarah work',
        'answer': 'Google'
    },
    {
        'context': 'The meeting is at noon',
        'question': 'When is the meeting',
        'answer': 'noon'
    },
    {
        'context': 'The car is red and fast',
        'question': 'What color is the car',
        'answer': 'red'
    },
] * 100  # Repetir para ter mais dados

print(f"  Dataset size: {len(data)} exemplos")

# ─── 2. PREPROCESSAR ───
print("\n🔧 Preprocessando...")

# Tokenizar
all_text = []
for item in data:
    all_text.extend(item['context'].split())
    all_text.extend(item['question'].split())
    all_text.extend(item['answer'].split())

vocab = sorted(set(all_text))
word2idx = {word.lower(): idx+1 for idx, word in enumerate(vocab)}
word2idx['<PAD>'] = 0
idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)

print(f"  Vocabulário: {vocab_size} palavras")

# Converter para sequências
contexts = [[word2idx[w.lower()] for w in item['context'].split()] for item in data]
questions = [[word2idx[w.lower()] for w in item['question'].split()] for item in data]

# Para simplificar, answer será classificação (índice da palavra no vocabulário)
answers = [word2idx[item['answer'].split()[0].lower()] for item in data]

# Pad
max_context_len = max(len(c) for c in contexts)
max_question_len = max(len(q) for q in questions)

contexts = pad_sequences(contexts, maxlen=max_context_len, padding='post')
questions = pad_sequences(questions, maxlen=max_question_len, padding='post')
answers = np.array(answers)

print(f"  Context shape: {contexts.shape}")
print(f"  Question shape: {questions.shape}")
print(f"  Answers shape: {answers.shape}")

# ─── 3. CONSTRUIR MODELO ───
print("\n🏗️ Construindo modelo QA...")

# Inputs
context_input = Input(shape=(max_context_len,), name='context')
question_input = Input(shape=(max_question_len,), name='question')

# Embeddings compartilhados
embedding_layer = Embedding(vocab_size, 64, mask_zero=True)

context_emb = embedding_layer(context_input)
question_emb = embedding_layer(question_input)

# Encoders
context_lstm = Bidirectional(LSTM(128, return_sequences=False))(context_emb)
question_lstm = Bidirectional(LSTM(128, return_sequences=False))(question_emb)

# Concatenar contexto + pergunta
combined = Concatenate()([context_lstm, question_lstm])

x = Dense(256, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output (classificação da palavra resposta)
output = Dense(vocab_size, activation='softmax')(x)

model = Model(inputs=[context_input, question_input], outputs=output, name='QA_Model')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando modelo QA...")

history = model.fit(
    [contexts, questions],
    answers,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=0
)

print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 5. TESTAR ───
print("\n❓ Testando perguntas...")

test_examples = [
    {'context': 'John lives in New York', 'question': 'Where does John live'},
    {'context': 'The cat is black', 'question': 'What color is the cat'},
    {'context': 'Sarah works at Google', 'question': 'Where does Sarah work'},
]

for ex in test_examples:
    # Preprocessar
    ctx = [word2idx.get(w.lower(), 0) for w in ex['context'].split()]
    q = [word2idx.get(w.lower(), 0) for w in ex['question'].split()]

    ctx_pad = pad_sequences([ctx], maxlen=max_context_len, padding='post')
    q_pad = pad_sequences([q], maxlen=max_question_len, padding='post')

    # Predição
    pred = model.predict([ctx_pad, q_pad], verbose=0)
    answer_idx = np.argmax(pred[0])
    answer_word = idx2word[answer_idx]
    confidence = pred[0][answer_idx]

    print(f"\n  Context: {ex['context']}")
    print(f"  Question: {ex['question']}")
    print(f"  Answer: {answer_word} (confidence: {confidence:.2%})")

# ─── 6. VISUALIZAR ───
print("\n📊 Visualizando training...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Question Answering Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('qa_training.png', dpi=150)
print("✅ Training salvo: qa_training.png")

print("\n💡 QUESTION ANSWERING TYPES:")
print("  • Extractive: Extrair resposta do texto (span)")
print("  • Abstractive: Gerar resposta (summarização)")
print("  • Multiple Choice: Escolher entre opções")
print("  • Yes/No: Resposta binária")

print("\n🏆 DATASETS POPULARES:")
print("  • SQuAD: Stanford Question Answering (100k+ perguntas)")
print("  • Natural Questions: Google, perguntas reais")
print("  • TriviaQA: 95k perguntas de trivia")
print("  • RACE: Reading Comprehension (exames)")

print("\n🎯 ARQUITETURAS MODERNAS:")
print("  • BERT: Bidirectional Encoder (SOTA)")
print("  • RoBERTa: Robustly optimized BERT")
print("  • ALBERT: Lite BERT")
print("  • T5: Text-to-Text Transfer Transformer")

print("\n✅ QUESTION ANSWERING COMPLETO!")
