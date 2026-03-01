# GO1524-19aNamedEntityRecognitionSystem
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

print("🏷️ Named Entity Recognition (NER) Demo\n")
print("="*70)

# Load pre-trained model
print("\n🔄 Loading spaCy model (en_core_web_sm)...\n")

try:
    nlp = spacy.load('en_core_web_sm')
    print("✅ Model loaded successfully!\n")
except:
    print("⚠️  Model not found. Install with:")
    print("   python -m spacy download en_core_web_sm")
    print("\nUsing simulated NER results for demo...\n")
    nlp = None

# Sample texts
texts = [
    "Apple Inc. is planning to open a new store in New York City next month. "
    "CEO Tim Cook announced the investment of $500 million in the project.",

    "Microsoft acquired GitHub for $7.5 billion in 2018. The deal was finalized "
    "in October and Satya Nadella praised the acquisition.",

    "The United Nations held a climate summit in Paris in December 2015. "
    "World leaders agreed to limit global warming to 2 degrees Celsius.",

    "Tesla CEO Elon Musk announced on Twitter that the company would invest "
    "$10 billion in a new factory in Texas next year.",

    "Amazon founder Jeff Bezos stepped down as CEO in July 2021. Andy Jassy "
    "took over the role at the Seattle-based company."
]

print("📝 Sample Texts:\n")
for i, text in enumerate(texts, 1):
    print(f"{i}. {text[:100]}...")
    print()

# Process texts and extract entities
print("="*70)
print("\n🔍 EXTRACTING ENTITIES...\n")

all_entities = []

if nlp:
    for text in texts:
        doc = nlp(text)

        for ent in doc.ents:
            all_entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
else:
    # Simulated entities for demo
    simulated = [
        ('Apple Inc.', 'ORG'), ('New York City', 'GPE'), ('next month', 'DATE'),
        ('Tim Cook', 'PERSON'), ('$500 million', 'MONEY'),
        ('Microsoft', 'ORG'), ('GitHub', 'ORG'), ('$7.5 billion', 'MONEY'),
        ('2018', 'DATE'), ('October', 'DATE'), ('Satya Nadella', 'PERSON'),
        ('United Nations', 'ORG'), ('Paris', 'GPE'), ('December 2015', 'DATE'),
        ('2 degrees Celsius', 'QUANTITY'),
        ('Tesla', 'ORG'), ('Elon Musk', 'PERSON'), ('Twitter', 'ORG'),
        ('$10 billion', 'MONEY'), ('Texas', 'GPE'), ('next year', 'DATE'),
        ('Amazon', 'ORG'), ('Jeff Bezos', 'PERSON'), ('July 2021', 'DATE'),
        ('Andy Jassy', 'PERSON'), ('Seattle', 'GPE')
    ]

    for text, label in simulated:
        all_entities.append({'text': text, 'label': label, 'start': 0, 'end': len(text)})

# Create DataFrame
df_entities = pd.DataFrame(all_entities)

print(f"✅ Extracted {len(all_entities)} entities\n")

# Entity type statistics
print("📊 ENTITY TYPE DISTRIBUTION:\n")

label_counts = Counter([e['label'] for e in all_entities])

for label, count in label_counts.most_common():
    print(f"   {label:15s}: {count:3d} entities")

print()

# Sample entities by type
print("="*70)
print("\n🏷️  SAMPLE ENTITIES BY TYPE:\n")

entity_types = ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE']

for ent_type in entity_types:
    entities_of_type = [e['text'] for e in all_entities if e['label'] == ent_type]

    if entities_of_type:
        print(f"{ent_type} (Person/Organization/Place/Money/Date):")
        sample = list(set(entities_of_type))[:5]
        print(f"   {', '.join(sample)}")
        print()

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Entity type distribution
ax = axes[0, 0]
labels = list(label_counts.keys())
counts = list(label_counts.values())

colors = plt.cm.Set3(range(len(labels)))
bars = ax.bar(labels, counts, color=colors, alpha=0.7)
ax.set_ylabel('Count')
ax.set_title('Entity Type Distribution')
ax.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            str(count), ha='center', va='bottom', fontweight='bold')

# 2. Entities per document
ax = axes[0, 1]

# Count entities per text
entities_per_doc = []
for text in texts:
    if nlp:
        doc = nlp(text)
        count = len(doc.ents)
    else:
        count = len([e for e in all_entities]) // len(texts)  # Approximate

    entities_per_doc.append(count)

ax.plot(range(1, len(texts)+1), entities_per_doc, 'o-', linewidth=2, 
       markersize=10, color='blue')
ax.set_xlabel('Document ID')
ax.set_ylabel('Number of Entities')
ax.set_title('Entities per Document')
ax.grid(alpha=0.3)

for i, count in enumerate(entities_per_doc, 1):
    ax.text(i, count + 0.3, str(count), ha='center', fontweight='bold')

# 3. Entity length distribution
ax = axes[1, 0]
entity_lengths = [len(e['text'].split()) for e in all_entities]

ax.hist(entity_lengths, bins=range(1, max(entity_lengths)+2), 
       color='lightgreen', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Words')
ax.set_ylabel('Frequency')
ax.set_title('Entity Length Distribution')
ax.grid(axis='y', alpha=0.3)

# 4. Top entities
ax = axes[1, 1]
ax.axis('off')

# Most common entities
entity_text_counts = Counter([e['text'] for e in all_entities])
top_entities = entity_text_counts.most_common(10)

table_data = []
for entity, count in top_entities:
    label = [e['label'] for e in all_entities if e['text'] == entity][0]
    table_data.append([entity[:20], label, count])

table = ax.table(cellText=table_data,
                colLabels=['Entity', 'Type', 'Count'],
                cellLoc='left',
                loc='center',
                colWidths=[0.5, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Top 10 Most Common Entities', pad=20, fontweight='bold')

plt.tight_layout()
plt.savefig('ner_analysis.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: ner_analysis.png")

print("\n✅ NER analysis completo!")
print("\n💡 ENTITY TYPES (spaCy):")
print("   - PERSON: People, including fictional")
print("   - NORP: Nationalities, religious/political groups")
print("   - ORG: Companies, agencies, institutions")
print("   - GPE: Countries, cities, states")
print("   - LOC: Non-GPE locations (mountains, bodies of water)")
print("   - PRODUCT: Objects, vehicles, foods, etc.")
print("   - DATE: Absolute or relative dates/periods")
print("   - TIME: Times smaller than a day")
print("   - MONEY: Monetary values")
print("   - QUANTITY: Measurements (weight, distance)")
print("\n💡 NER APPROACHES:")
print("   - Rule-based: Gazetteers, regex patterns")
print("   - Statistical: CRF, HMM (classical ML)")
print("   - Deep Learning: BiLSTM-CRF, Transformers (BERT)")
print("\n💡 APPLICATIONS:")
print("   - Information extraction from documents")
print("   - Question answering systems")
print("   - Knowledge graph construction")
print("   - Content recommendation")
