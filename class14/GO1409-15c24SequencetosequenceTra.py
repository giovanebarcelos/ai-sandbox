# GO1409-15c24SequencetosequenceTra
# ═══════════════════════════════════════════════════════════════════
# TRADUÇÃO INGLÊS-PORTUGUÊS COM SEQ2SEQ LSTM
# Arquitetura: Encoder-Decoder com Attention (simplificado)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.callbacks import EarlyStopping
import re

# ─── 1. CRIAR DATASET DE TRADUÇÃO ───
print("🌍 Criando dataset de tradução Inglês → Português...")

# Dataset sintético de frases
translation_pairs = [
    # Cumprimentos
    ("hello", "olá"),
    ("good morning", "bom dia"),
    ("good afternoon", "boa tarde"),
    ("good evening", "boa noite"),
    ("goodbye", "tchau"),
    ("see you later", "até logo"),
    ("how are you", "como vai você"),
    ("i am fine", "estou bem"),
    ("thank you", "obrigado"),
    ("you are welcome", "de nada"),

    # Básico
    ("yes", "sim"),
    ("no", "não"),
    ("maybe", "talvez"),
    ("please", "por favor"),
    ("sorry", "desculpe"),
    ("excuse me", "com licença"),

    # Cores
    ("the sky is blue", "o céu é azul"),
    ("the grass is green", "a grama é verde"),
    ("the sun is yellow", "o sol é amarelo"),
    ("the night is dark", "a noite é escura"),

    # Números/Tempo
    ("i have one cat", "eu tenho um gato"),
    ("i have two dogs", "eu tenho dois cachorros"),
    ("today is monday", "hoje é segunda"),
    ("tomorrow is tuesday", "amanhã é terça"),

    # Comida
    ("i like pizza", "eu gosto de pizza"),
    ("i love chocolate", "eu amo chocolate"),
    ("water is good", "água é bom"),
    ("coffee is hot", "café está quente"),
    ("the food is delicious", "a comida está deliciosa"),

    # Família
    ("my name is john", "meu nome é joão"),
    ("i have a brother", "eu tenho um irmão"),
    ("i have a sister", "eu tenho uma irmã"),
    ("my father works", "meu pai trabalha"),
    ("my mother cooks", "minha mãe cozinha"),

    # Ações
    ("i am walking", "estou caminhando"),
    ("i am running", "estou correndo"),
    ("i am reading", "estou lendo"),
    ("i am writing", "estou escrevendo"),
    ("i am sleeping", "estou dormindo"),
    ("i am eating", "estou comendo"),

    # Locais
    ("i live in brazil", "eu moro no brasil"),
    ("i work at home", "eu trabalho em casa"),
    ("the school is big", "a escola é grande"),
    ("the house is small", "a casa é pequena"),
    ("the city is beautiful", "a cidade é linda"),

    # Sentimentos
    ("i am happy", "estou feliz"),
    ("i am sad", "estou triste"),
    ("i am tired", "estou cansado"),
    ("i am hungry", "estou com fome"),
    ("i am thirsty", "estou com sede"),

    # Questões
    ("what is your name", "qual é seu nome"),
    ("where do you live", "onde você mora"),
    ("how old are you", "quantos anos você tem"),
    ("what do you do", "o que você faz"),

    # Frases completas
    ("the cat is on the table", "o gato está na mesa"),
    ("the dog is in the garden", "o cachorro está no jardim"),
    ("i want to learn portuguese", "eu quero aprender português"),
    ("i speak english", "eu falo inglês"),
    ("i love my family", "eu amo minha família"),
    ("i go to school every day", "eu vou para escola todo dia"),
    ("the weather is nice today", "o tempo está bom hoje"),
    ("i need to study more", "eu preciso estudar mais"),
]

# Adicionar tokens especiais
START_TOKEN = '<start>'
END_TOKEN = '<end>'

# Preparar pares
source_texts = []
target_texts = []
target_texts_inputs = []

for eng, por in translation_pairs:
    source_texts.append(eng.lower())
    target_texts.append(f"{START_TOKEN} {por.lower()} {END_TOKEN}")
    target_texts_inputs.append(f"{por.lower()} {END_TOKEN}")

print(f"  Total de pares: {len(source_texts)}")
print(f"\n  Exemplos:")
for i in range(3):
    print(f"    EN: {source_texts[i]}")
    print(f"    PT: {target_texts[i]}")
    print()

# ─── 2. TOKENIZAÇÃO ───
print("🔧 Tokenizando textos...")

# Tokenizer para inglês (source)
source_tokenizer = Tokenizer(filters='', lower=True)
source_tokenizer.fit_on_texts(source_texts)
source_vocab_size = len(source_tokenizer.word_index) + 1

# Tokenizer para português (target)
target_tokenizer = Tokenizer(filters='', lower=True)
target_tokenizer.fit_on_texts(target_texts + target_texts_inputs)
target_vocab_size = len(target_tokenizer.word_index) + 1

print(f"  Vocabulário inglês: {source_vocab_size} palavras")
print(f"  Vocabulário português: {target_vocab_size} palavras")

# Sequências
source_sequences = source_tokenizer.texts_to_sequences(source_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_sequences_inputs = target_tokenizer.texts_to_sequences(target_texts_inputs)

# Padding
max_source_len = max(len(seq) for seq in source_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

encoder_input_data = pad_sequences(source_sequences, maxlen=max_source_len, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')
decoder_target_data = pad_sequences(target_sequences_inputs, maxlen=max_target_len, padding='post')

print(f"  Max len (source): {max_source_len}")
print(f"  Max len (target): {max_target_len}")
print(f"  Encoder input shape: {encoder_input_data.shape}")
print(f"  Decoder input shape: {decoder_input_data.shape}")
print(f"  Decoder target shape: {decoder_target_data.shape}")

# ─── 3. CONSTRUIR MODELO SEQ2SEQ ───
print("\n🔨 Construindo modelo Seq2Seq...")

LATENT_DIM = 256
EMBEDDING_DIM = 128

# ENCODER
encoder_inputs = Input(shape=(max_source_len,))
encoder_embedding = Embedding(source_vocab_size, EMBEDDING_DIM)(encoder_inputs)
encoder_lstm = LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# DECODER
decoder_inputs = Input(shape=(max_target_len,))
decoder_embedding = Embedding(target_vocab_size, EMBEDDING_DIM)(decoder_inputs)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo completo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")
model.summary()

# ─── 4. TREINAR MODELO ───
print("\n🚀 Treinando modelo Seq2Seq...")

# Reshape target para (samples, timesteps, 1)
decoder_target_reshaped = np.expand_dims(decoder_target_data, -1)

callbacks = [EarlyStopping(patience=20, restore_best_weights=True, verbose=1)]

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_reshaped,
    batch_size=8,
    epochs=200,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# ─── 5. CONSTRUIR MODELOS DE INFERÊNCIA ───
print("\n🔧 Construindo modelos de inferência...")

# Encoder de inferência
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder de inferência
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

print("  ✓ Modelos de inferência prontos")

# ─── 6. FUNÇÃO DE TRADUÇÃO ───
def translate(input_seq):
    """Traduz sequência do inglês para português"""
    # Encode
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Token inicial
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index[START_TOKEN]

    # Decodificar
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Amostrar token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None

        for word, index in target_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word == END_TOKEN or len(decoded_sentence) > max_target_len:
            stop_condition = True
        elif sampled_word and sampled_word != START_TOKEN:
            decoded_sentence.append(sampled_word)

        # Atualizar target_seq
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Atualizar states
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# ─── 7. AVALIAR NO DATASET ───
print("\n📊 Avaliando traduções no dataset de treino...")

correct = 0
total = len(source_texts)

print("\n  Exemplos de traduções:")
for i in range(min(15, total)):
    input_seq = encoder_input_data[i:i+1]
    translated = translate(input_seq)
    expected = target_texts_inputs[i].replace(END_TOKEN, '').strip()

    match = '✓' if translated == expected else '✗'
    if translated == expected:
        correct += 1

    print(f"\n    {match} EN: {source_texts[i]}")
    print(f"      PT (pred): {translated}")
    print(f"      PT (real): {expected}")

accuracy = correct / total * 100
print(f"\n  Accuracy: {accuracy:.2f}% ({correct}/{total})")

# ─── 8. TESTAR COM NOVAS FRASES ───
print("\n🧪 Testando com novas frases...")

new_sentences = [
    "hello",
    "good morning",
    "i am happy",
    "the cat is on the table",
    "i love my family",
    "i want to learn portuguese"
]

print("\n  Traduções:")
for sentence in new_sentences:
    # Tokenizar e padear
    seq = source_tokenizer.texts_to_sequences([sentence.lower()])
    padded = pad_sequences(seq, maxlen=max_source_len, padding='post')

    # Traduzir
    translation = translate(padded)

    print(f"\n    EN: {sentence}")
    print(f"    PT: {translation}")

# ─── 9. VISUALIZAR HISTÓRICO ───
print("\n📈 Gerando visualizações...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Treino', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validação', linewidth=2)
axes[0].set_title('Histórico de Treinamento - Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Treino', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
axes[1].set_title('Histórico de Treinamento - Accuracy', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seq2seq_training.png', dpi=150)
print("  ✓ Histórico salvo: seq2seq_training.png")

# ─── 10. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ TRADUÇÃO SEQ2SEQ CONCLUÍDA!")
print("="*70)

print(f"\n📊 Estatísticas:")
print(f"  Pares de tradução: {total}")
print(f"  Vocabulário EN: {source_vocab_size} palavras")
print(f"  Vocabulário PT: {target_vocab_size} palavras")
print(f"  Max length (source): {max_source_len}")
print(f"  Max length (target): {max_target_len}")

print(f"\n🎯 Performance:")
print(f"  Accuracy no treino: {accuracy:.2f}%")
print(f"  Parâmetros: {model.count_params():,}")

print("\n📁 Arquivos gerados:")
print("  • seq2seq_training.png - Histórico de treinamento")

print("\n💡 Arquitetura:")
print("  • Encoder: Embedding → LSTM")
print("  • Decoder: Embedding → LSTM → Dense")
print("  • Latent dim: 256")

print("\n🔧 Melhorias possíveis:")
print("  • Adicionar Attention mechanism (Bahdanau ou Luong)")
print("  • Usar Bidirectional LSTM no encoder")
print("  • Implementar Teacher Forcing durante treino")
print("  • Testar com dataset maior (Tatoeba, OpenSubtitles)")
print("  • Usar Beam Search para inferência")
print("  • Implementar BLEU score para avaliação")
