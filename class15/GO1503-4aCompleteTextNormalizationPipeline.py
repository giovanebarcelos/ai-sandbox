# GO1503-4aCompleteTextNormalizationPipeline
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

class TextNormalizer:
    """
    Complete text normalization pipeline

    Steps:
    1. Unicode normalization
    2. Lowercase conversion
    3. URL/email removal
    4. Number handling
    5. Punctuation removal
    6. Stopword removal
    7. Whitespace normalization
    """

    def __init__(self, language='english', remove_stopwords=True):
        self.language = language
        self.remove_stopwords = remove_stopwords

        if remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()

    def normalize_unicode(self, text):
        """Normalize unicode characters (NFD)"""
        return unicodedata.normalize('NFD', text)

    def remove_accents(self, text):
        """Remove accents from characters"""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

    def lowercase(self, text):
        """Convert to lowercase"""
        return text.lower()

    def remove_urls(self, text):
        """Remove URLs"""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def remove_emails(self, text):
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    def remove_numbers(self, text, replace_with=''):
        """Remove or replace numbers"""
        return re.sub(r'\d+', replace_with, text)

    def remove_punctuation(self, text):
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_extra_whitespace(self, text):
        """Remove extra whitespace"""
        return ' '.join(text.split())

    def remove_stopwords_func(self, text):
        """Remove stopwords"""
        if not self.remove_stopwords:
            return text

        words = text.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        return ' '.join(filtered_words)

    def normalize(self, text, steps='all'):
        """
        Apply normalization pipeline

        steps: 'all' or list of step names
        """
        original = text

        if steps == 'all':
            text = self.remove_urls(text)
            text = self.remove_emails(text)
            text = self.lowercase(text)
            text = self.remove_accents(text)
            text = self.remove_numbers(text)
            text = self.remove_punctuation(text)
            text = self.remove_extra_whitespace(text)
            text = self.remove_stopwords_func(text)
        else:
            for step in steps:
                if hasattr(self, step):
                    text = getattr(self, step)(text)

        return text

    def analyze_normalization(self, text):
        """Show step-by-step normalization"""
        steps = [
            ('Original', text),
            ('Remove URLs', self.remove_urls(text)),
            ('Remove Emails', self.remove_emails(self.remove_urls(text))),
            ('Lowercase', self.lowercase(self.remove_emails(self.remove_urls(text)))),
        ]

        current = text
        for step_name, method in [('Remove URLs', self.remove_urls),
                                  ('Remove Emails', self.remove_emails),
                                  ('Lowercase', self.lowercase),
                                  ('Remove Accents', self.remove_accents),
                                  ('Remove Numbers', self.remove_numbers),
                                  ('Remove Punctuation', self.remove_punctuation),
                                  ('Remove Stopwords', self.remove_stopwords_func),
                                  ('Clean Whitespace', self.remove_extra_whitespace)]:
            current = method(current)
            steps.append((step_name, current))

        return steps

# === DEMO ===

print("🧹 Text Normalization Pipeline Demo\n")
print("="*70)

# Sample text
sample_text = """
Contact me at john.doe@email.com or visit https://www.example.com!
I have 3 cats and 2 dogs. Python is AMAZING!!!
Don't forget to check the website tomorrow at 10:00 AM.
"""

print("📝 Original Text:")
print(sample_text)
print("\n" + "="*70)

# Initialize normalizer
normalizer = TextNormalizer(language='english', remove_stopwords=True)

# Step-by-step analysis
print("\n🔍 STEP-BY-STEP NORMALIZATION:\n")

steps = normalizer.analyze_normalization(sample_text)

for i, (step_name, result) in enumerate(steps[:9]):
    print(f"{i}. {step_name}:")
    print(f"   \"{result[:80]}...\" (len={len(result)})\n")

# Final result
print("="*70)
print("\n✅ FINAL NORMALIZED TEXT:\n")
normalized = normalizer.normalize(sample_text)
print(f"\"{normalized}\"")

print(f"\n📊 Statistics:")
print(f"   Original length: {len(sample_text)} characters")
print(f"   Normalized length: {len(normalized)} characters")
print(f"   Reduction: {(1 - len(normalized)/len(sample_text))*100:.1f}%")

# Multiple documents
print("\n" + "="*70)
print("\n📚 Processing Multiple Documents\n")

documents = [
    "Visit https://example.com for more info! Email: info@example.com",
    "Python 3.9 is amazing!!! I love programming.",
    "Don't forget: meeting at 3:00 PM tomorrow.",
    "Check out www.python.org - it's the BEST resource!"
]

original_lengths = []
normalized_lengths = []

for i, doc in enumerate(documents, 1):
    normalized_doc = normalizer.normalize(doc)
    original_lengths.append(len(doc))
    normalized_lengths.append(len(normalized_doc))

    print(f"Doc {i}:")
    print(f"   Original:   \"{doc}\"")
    print(f"   Normalized: \"{normalized_doc}\"")
    print()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Length comparison
ax = axes[0]
x = range(1, len(documents) + 1)
width = 0.35

ax.bar([i - width/2 for i in x], original_lengths, width, label='Original', alpha=0.8, color='steelblue')
ax.bar([i + width/2 for i in x], normalized_lengths, width, label='Normalized', alpha=0.8, color='lightcoral')

ax.set_xlabel('Document')
ax.set_ylabel('Length (characters)')
ax.set_title('Text Length: Original vs Normalized')
ax.set_xticks(x)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. Reduction percentage
ax = axes[1]
reductions = [(1 - norm/orig) * 100 for orig, norm in zip(original_lengths, normalized_lengths)]

ax.bar(x, reductions, color='green', alpha=0.7)
ax.set_xlabel('Document')
ax.set_ylabel('Reduction (%)')
ax.set_title('Text Reduction After Normalization')
ax.set_xticks(x)
ax.axhline(y=sum(reductions)/len(reductions), color='r', linestyle='--', 
          label=f'Average: {sum(reductions)/len(reductions):.1f}%')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('text_normalization.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: text_normalization.png")

print("\n✅ Text normalization completo!")
print("\n💡 NORMALIZATION TIPS:")
print("   - Apply steps in correct order (URLs before lowercase)")
print("   - Consider domain-specific requirements")
print("   - Keep numbers if relevant (dates, prices)")
print("   - Stopwords depend on task (keep for translation)")
print("   - Test on sample data before full pipeline")
print("\n💡 WHEN TO USE:")
print("   - Before feature extraction (BoW, TF-IDF)")
print("   - Before training ML models")
print("   - For text comparison and deduplication")
print("   - Search and retrieval systems")
