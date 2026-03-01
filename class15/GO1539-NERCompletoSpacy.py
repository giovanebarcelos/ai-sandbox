# GO1539-NERCompletoSpacy
import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Carregar modelo
nlp = spacy.load("en_core_web_sm")

# Textos de exemplo
texts = [
    "Apple Inc. is planning to open a new store in New York City next month. "
    "CEO Tim Cook announced the initiative during a press conference.",

    "Microsoft and Google are competing for cloud computing dominance. "
    "Satya Nadella met with Sundar Pichai in San Francisco last week.",

    "The United Nations held a summit in Geneva, Switzerland. "
    "Secretary-General António Guterres addressed climate change concerns.",
]

# Processar e extrair entidades
all_entities = []
for idx, text in enumerate(texts, 1):
    doc = nlp(text)
    print(f"\n{'='*70}")
    print(f"TEXTO {idx}:")
    print(f"{'='*70}")
    print(text)
    print(f"\n🏷️  Entidades encontradas:")

    for ent in doc.ents:
        print(f"   {ent.text:25s} | {ent.label_:15s} | {spacy.explain(ent.label_)}")
        all_entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'doc_id': idx
        })

    # Visualização inline (renderiza HTML)
    html = displacy.render(doc, style="ent", jupyter=False)
    with open(f'ner_visualization_{idx}.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ Visualização salva: ner_visualization_{idx}.html")

# Análise estatística
df_entities = pd.DataFrame(all_entities)

print(f"\n{'='*70}")
print("📊 ANÁLISE ESTATÍSTICA")
print(f"{'='*70}")

# Contar por tipo
entity_counts = df_entities['label'].value_counts()
print(f"\n🔢 Distribuição por tipo:")
print(entity_counts)

# Visualizar distribuição
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Contagem por tipo
axes[0, 0].barh(entity_counts.index, entity_counts.values, color='skyblue', alpha=0.8)
axes[0, 0].set_xlabel('Count', fontsize=11)
axes[0, 0].set_title('Entity Type Distribution', fontsize=13, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Entidades mais comuns
top_entities = df_entities['text'].value_counts().head(10)
axes[0, 1].barh(range(len(top_entities)), top_entities.values, color='coral', alpha=0.8)
axes[0, 1].set_yticks(range(len(top_entities)))
axes[0, 1].set_yticklabels(top_entities.index, fontsize=9)
axes[0, 1].set_xlabel('Frequency', fontsize=11)
axes[0, 1].set_title('Top 10 Most Common Entities', fontsize=13, fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Entidades por documento
doc_counts = df_entities.groupby(['doc_id', 'label']).size().unstack(fill_value=0)
doc_counts.plot(kind='bar', ax=axes[1, 0], alpha=0.8, stacked=False)
axes[1, 0].set_xlabel('Document ID', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Entities per Document by Type', fontsize=13, fontweight='bold')
axes[1, 0].legend(title='Entity Type', fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_xticklabels(doc_counts.index, rotation=0)

# 4. Comprimento das entidades
entity_lengths = df_entities['text'].str.len()
axes[1, 1].hist(entity_lengths, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Entity Length (characters)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Entity Length Distribution', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].axvline(entity_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {entity_lengths.mean():.1f}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('ner_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

# Extração de relações (simples)
print(f"\n🔗 EXTRAÇÃO DE RELAÇÕES SIMPLES:")
for idx, text in enumerate(texts, 1):
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

    if persons and orgs:
        print(f"\nDoc {idx}: {persons[0]} ↔ {orgs[0]}")
    if persons and locations:
        print(f"Doc {idx}: {persons[0]} → {locations[0]}")

print(f"\n✅ Análise NER completa!")
