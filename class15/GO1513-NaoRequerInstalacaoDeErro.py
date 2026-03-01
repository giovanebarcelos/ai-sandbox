# GO1513-NãoRequerInstalaçãoDeErro
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import seaborn as sns

class RegexEntityExtractor:
    """
    Extract entities from text using regex patterns

    Entities:
    - Emails
    - Phone numbers (BR format)
    - URLs
    - Dates
    - Money amounts
    - Credit cards
    - CPF/CNPJ
    - Hashtags & mentions
    """

    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_br': r'\(?\d{2}\)?\s?9?\d{4}-?\d{4}',
            'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
            'date_br': r'\d{1,2}/\d{1,2}/\d{4}',
            'money_br': r'R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?',
            'cpf': r'\d{3}\.\d{3}\.\d{3}-\d{2}',
            'cnpj': r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}',
            'hashtag': r'#\w+',
            'mention': r'@\w+',
            'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
        }

    def extract_entities(self, text: str, entity_type: str = None):
        """Extract specific entity type or all entities"""
        if entity_type:
            pattern = self.patterns.get(entity_type)
            if not pattern:
                raise ValueError(f"Unknown entity type: {entity_type}")
            return re.findall(pattern, text, re.IGNORECASE)

        # Extract all entities
        entities = {}
        for ent_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[ent_type] = matches

        return entities

    def clean_text(self, text: str):
        """Remove URLs, emails, and special characters"""
        # Remove URLs
        text = re.sub(self.patterns['url'], '[URL]', text)
        # Remove emails
        text = re.sub(self.patterns['email'], '[EMAIL]', text)
        # Remove phone numbers
        text = re.sub(self.patterns['phone_br'], '[PHONE]', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def validate_entities(self, text: str):
        """Validate and categorize extracted entities"""
        entities = self.extract_entities(text)

        validated = {}
        for ent_type, matches in entities.items():
            validated[ent_type] = {
                'count': len(matches),
                'unique': len(set(matches)),
                'samples': matches[:3]
            }

        return validated

# === DEMO ===

print("🔍 Advanced Regex Entity Extraction\n")
print("="*70)

# Sample text
sample_text = """
Contato: joao.silva@empresa.com.br ou (11) 98765-4321
Visite nosso site: https://www.minhaempresa.com.br
Pagamento: R$ 1.250,00 até 15/03/2024
CPF: 123.456.789-01
Siga-nos: @empresa_oficial #tecnologia #inovacao
Cartão: 4532 1234 5678 9010
CNPJ: 12.345.678/0001-90
"""

extractor = RegexEntityExtractor()

print("\n📝 Texto Original:")
print(sample_text)
print("\n" + "="*70)

# Extract all entities
entities = extractor.extract_entities(sample_text)

print("\n🔍 Entidades Extraídas:\n")
for ent_type, matches in entities.items():
    print(f"   {ent_type.upper()}:")
    for match in matches:
        print(f"      - {match}")
    print()

# Validate entities
validated = extractor.validate_entities(sample_text)

print("="*70)
print("\n📊 Estatísticas:\n")
for ent_type, stats in validated.items():
    print(f"   {ent_type}: {stats['count']} encontrado(s), {stats['unique']} único(s)")

# Clean text
cleaned = extractor.clean_text(sample_text)
print("\n" + "="*70)
print("\n🧹 Texto Limpo:")
print(cleaned)

# Multiple documents analysis
print("\n" + "="*70)
print("\n📚 Análise de Múltiplos Documentos\n")

documents = [
    "Envie email para suporte@empresa.com ou ligue (21) 3456-7890",
    "Acesse https://docs.python.org #python #programming",
    "Pagamento R$ 500,00 via PIX para CPF 987.654.321-00",
    "Siga @tech_news e @dev_brasil no Twitter #tech",
    "CNPJ: 98.765.432/0001-10 Telefone: (11) 99999-8888"
]

all_entities = {key: [] for key in extractor.patterns.keys()}

for doc in documents:
    ents = extractor.extract_entities(doc)
    for ent_type, matches in ents.items():
        all_entities[ent_type].extend(matches)

print("Resumo das Entidades em Todos os Documentos:\n")
for ent_type, matches in all_entities.items():
    if matches:
        print(f"   {ent_type}: {len(matches)} total, {len(set(matches))} únicos")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Entity counts
ax = axes[0]
ent_counts = {k: len(v) for k, v in all_entities.items() if v}
if ent_counts:
    bars = ax.bar(ent_counts.keys(), ent_counts.values(), color='skyblue', alpha=0.7)
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('Entity Distribution Across Documents')
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 2. Document coverage
ax = axes[1]
doc_entity_counts = []
for doc in documents:
    ents = extractor.extract_entities(doc)
    doc_entity_counts.append(sum(len(v) for v in ents.values()))

ax.bar(range(1, len(documents)+1), doc_entity_counts, color='lightcoral', alpha=0.7)
ax.set_xlabel('Document ID')
ax.set_ylabel('Total Entities')
ax.set_title('Entities per Document')
ax.set_xticks(range(1, len(documents)+1))

plt.tight_layout()
plt.savefig('regex_entity_extraction.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: regex_entity_extraction.png")

print("\n✅ Regex entity extraction completo!")
print("\n💡 USE CASES:")
print("   - Data anonymization (remove PII)")
print("   - Contact information extraction")
print("   - Social media analysis (hashtags, mentions)")
print("   - Financial data extraction")
print("   - Document preprocessing")
print("\n💡 BEST PRACTICES:")
print("   - Test patterns with multiple examples")
print("   - Handle regional variations (BR vs US phone formats)")
print("   - Use named groups for complex patterns")
print("   - Validate extracted data")
print("   - Consider using spaCy NER for entity recognition")
