# GO1422-36gMusicGenerationComLstmComposição
# ═══════════════════════════════════════════════════════════════════
# MUSIC GENERATION COM LSTM
# Gerar sequências musicais nota por nota
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
import matplotlib.pyplot as plt

print("🎵 MUSIC GENERATION COM LSTM")
print("=" * 70)

# ─── 1. CRIAR DATASET DE NOTAS MUSICAIS ───
print("\n🎼 Gerando dataset de notas musicais...")

# Notas da escala de Dó maior (C major)
notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']  # 1 oitava

# Criar sequência musical sintética (padrões conhecidos)
# Exemplo: escala, arpeggio, melodies
musical_sequences = [
    # Escalas
    'C D E F G A B C',
    'C B A G F E D C',
    'E F G A B C D E',

    # Arpeggios
    'C E G C',
    'C G E C',
    'D F A D',
    'G B D G',

    # Melodias simples
    'C C G G A A G',
    'F F E E D D C',
    'E E D D C',
    'C D E C D E F',
    'G A B C B A G',

    # Padrões rítmicos
    'C C C E E E G G G C',
    'D D F F A A D',
    'E G E G C',
]

# Repetir para aumentar dataset
musical_sequences = musical_sequences * 50

# Concatenar tudo
music_text = ' '.join(musical_sequences)
music_notes = music_text.split()

print(f"  Total de notas: {len(music_notes)}")
print(f"  Notas únicas: {len(set(music_notes))}")
print(f"  Vocab: {sorted(set(music_notes))}")

# ─── 2. PREPARAR DADOS ───
print("\n🔧 Preparando dados...")

# Criar vocabulário
vocab = sorted(set(music_notes))
note_to_int = {note: i for i, note in enumerate(vocab)}
int_to_note = {i: note for note, i in note_to_int.items()}

vocab_size = len(vocab)

print(f"  Vocab size: {vocab_size}")

# Converter notas para inteiros
notes_int = [note_to_int[note] for note in music_notes]

# Criar sequências (window)
seq_length = 10
X = []
y = []

for i in range(len(notes_int) - seq_length):
    X.append(notes_int[i:i+seq_length])
    y.append(notes_int[i+seq_length])

X = np.array(X)
y = np.array(y)

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ─── 3. CONSTRUIR MODELO ───
print("\n🏗️ Construindo modelo LSTM...")

model = Sequential([
    Embedding(vocab_size, 32, input_length=seq_length),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(vocab_size, activation='softmax')
], name='MusicGenerator')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando modelo...")

history = model.fit(
    X, y,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

print(f"  Final loss: {history.history['loss'][-1]:.4f}")
print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 5. GERAR MÚSICA ───
print("\n🎶 Gerando música...")

def generate_music(seed_sequence, num_notes=30, temperature=1.0):
    """
    Gera sequência musical
    temperature: controla aleatoriedade (0.5=conservador, 1.5=criativo)
    """
    generated = seed_sequence.copy()

    for _ in range(num_notes):
        # Preparar input
        x = np.array([generated[-seq_length:]])

        # Prever próxima nota
        predictions = model.predict(x, verbose=0)[0]

        # Aplicar temperature
        predictions = np.log(predictions + 1e-7) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample
        next_note = np.random.choice(vocab_size, p=predictions)
        generated.append(next_note)

    return generated

# Seed: iniciar com sequência conhecida
seed = [note_to_int['C'], note_to_int['D'], note_to_int['E'], note_to_int['F'],
        note_to_int['G'], note_to_int['A'], note_to_int['B'], note_to_int['C'],
        note_to_int['D'], note_to_int['E']]

print("\n  Seed: " + ' '.join([int_to_note[i] for i in seed]))

# Gerar com diferentes temperatures
temperatures = [0.5, 1.0, 1.5]

for temp in temperatures:
    generated_int = generate_music(seed, num_notes=20, temperature=temp)
    generated_notes = [int_to_note[i] for i in generated_int]

    print(f"\n  Temperature {temp}:")
    print(f"    {' '.join(generated_notes)}")

# ─── 6. VISUALIZAR ───
print("\n📈 Visualizando gerações...")

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

for idx, temp in enumerate(temperatures):
    generated_int = generate_music(seed, num_notes=50, temperature=temp)
    generated_notes = [int_to_note[i] for i in generated_int]

    # Plot piano roll
    note_indices = [note_to_int[note] for note in generated_notes]

    axes[idx].scatter(range(len(note_indices)), note_indices, s=100, alpha=0.7)
    axes[idx].plot(note_indices, alpha=0.3)
    axes[idx].set_yticks(range(vocab_size))
    axes[idx].set_yticklabels(vocab)
    axes[idx].set_xlabel('Timestep', fontsize=11)
    axes[idx].set_ylabel('Nota', fontsize=11)
    axes[idx].set_title(f'Temperature {temp} - Generated Melody', fontsize=12, fontweight='bold')
    axes[idx].grid(alpha=0.3)

plt.suptitle('Music Generation com LSTM', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('music_generation_lstm.png', dpi=150)
print("✅ Música salva: music_generation_lstm.png")

# ─── 7. ANÁLISE DE PROBABILIDADES ───
print("\n📊 Analisando probabilidades de transição...")

# Para uma sequência específica, ver distribuição de próxima nota
test_sequence = [note_to_int['C'], note_to_int['D'], note_to_int['E'], note_to_int['F'],
                note_to_int['G'], note_to_int['A'], note_to_int['B'], note_to_int['C'],
                note_to_int['D'], note_to_int['E']]

test_x = np.array([test_sequence])
probs = model.predict(test_x, verbose=0)[0]

plt.figure(figsize=(10, 6))
plt.bar(vocab, probs, color='steelblue', alpha=0.7)
plt.xlabel('Nota', fontsize=12)
plt.ylabel('Probabilidade', fontsize=12)
plt.title(f'Probabilidade da Próxima Nota após "{" ".join([int_to_note[i] for i in test_sequence])}"', 
         fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('music_generation_probs.png', dpi=150)
print("✅ Probabilidades salvas: music_generation_probs.png")

# Top 3 notas mais prováveis
top3_indices = np.argsort(probs)[-3:][::-1]
print("\n  Top 3 próximas notas mais prováveis:")
for idx in top3_indices:
    print(f"    {int_to_note[idx]}: {probs[idx]:.2%}")

print("\n💡 TEMPERATURE SAMPLING:")
print("  • Low (0.5): Conservador, repete padrões")
print("  • Medium (1.0): Balanceado")
print("  • High (1.5): Criativo, mais variação")

print("\n📚 EXTENSÕES POSSÍVEIS:")
print("  • Adicionar duração (semínima, colcheia, etc.)")
print("  • Polifonia (múltiplas notas simultâneas)")
print("  • Dinâmica (volume, forte/piano)")
print("  • MIDI: Exportar para formato MIDI real")

print("\n🎯 APLICAÇÕES:")
print("  • Composição assistida")
print("  • Música ambiente para games")
print("  • Educação musical")
print("  • Inspiração para compositores")

print("\n✅ MUSIC GENERATION COMPLETA!")
