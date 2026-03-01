# GO1528-23aTopicModelingWithLda
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("📊 Topic Modeling with LDA Demo\n")
print("="*70)

# Load sample dataset
print("\n📂 Loading 20 Newsgroups dataset (subset)...\n")

categories = ['rec.sport.baseball', 'rec.sport.hockey', 
              'comp.graphics', 'comp.sys.ibm.pc.hardware',
              'sci.med', 'sci.space']

newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Sample documents
documents = newsgroups.data[:500]
print(f"✅ Loaded {len(documents)} documents from {len(categories)} categories\n")

# Preprocessing & Feature Extraction
print("🔄 Extracting features with CountVectorizer...\n")

vectorizer = CountVectorizer(
    max_features=1000,
    max_df=0.95,
    min_df=2,
    stop_words='english'
)

doc_term_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"📊 Document-Term Matrix: {doc_term_matrix.shape}")
print(f"   Documents: {doc_term_matrix.shape[0]}")
print(f"   Vocabulary: {doc_term_matrix.shape[1]}\n")

# Train LDA model
print("="*70)
print("\n🧠 Training LDA Model...\n")

n_topics = 6
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=25,
    learning_method='online',
    random_state=42,
    n_jobs=-1
)

doc_topic_dist = lda.fit_transform(doc_term_matrix)

print("✅ LDA training complete!\n")
print(f"   Topics: {n_topics}")
print(f"   Perplexity: {lda.perplexity(doc_term_matrix):.2f}")
print(f"   Log-likelihood: {lda.score(doc_term_matrix):.2f}\n")

# Display topics
print("="*70)
print("\n🏷️  DISCOVERED TOPICS:\n")

def display_topics(model, feature_names, n_top_words=10):
    """Display top words for each topic"""
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

        print(f"Topic {topic_idx}:")
        print(f"   {', '.join(top_words)}")
        print()

    return topics

topics = display_topics(lda, feature_names, n_top_words=10)

# Assign topic labels (manual interpretation)
topic_labels = [
    'Baseball',
    'Computer Hardware', 
    'Medical Science',
    'Ice Hockey',
    'Computer Graphics',
    'Space Science'
]

print("="*70)
print("\n📑 DOCUMENT-TOPIC DISTRIBUTION (Sample):\n")

# Show topic distribution for first 5 documents
for i in range(5):
    print(f"Document {i}:")
    doc_topics = doc_topic_dist[i]
    dominant_topic = doc_topics.argmax()

    print(f"   Dominant Topic: {dominant_topic} ({topic_labels[dominant_topic]})")
    print(f"   Distribution: {doc_topics}")
    print()

print("="*70)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Topic word distributions (heatmap)
ax = axes[0, 0]
top_n = 8
topic_word_matrix = np.zeros((n_topics, top_n))

for i, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-top_n:][::-1]
    topic_word_matrix[i] = topic[top_indices]

im = ax.imshow(topic_word_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(top_n))
ax.set_yticks(range(n_topics))
ax.set_xticklabels([f'W{i+1}' for i in range(top_n)])
ax.set_yticklabels([f'T{i}' for i in range(n_topics)])
ax.set_xlabel('Top Words')
ax.set_ylabel('Topics')
ax.set_title('Topic-Word Distribution Heatmap')
plt.colorbar(im, ax=ax)

# 2. Document count per dominant topic
ax = axes[0, 1]
dominant_topics = doc_topic_dist.argmax(axis=1)
topic_counts = np.bincount(dominant_topics, minlength=n_topics)

bars = ax.bar(range(n_topics), topic_counts, color='skyblue', alpha=0.7)
ax.set_xlabel('Topic')
ax.set_ylabel('Number of Documents')
ax.set_title('Documents per Dominant Topic')
ax.set_xticks(range(n_topics))
ax.set_xticklabels([topic_labels[i] for i in range(n_topics)], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, topic_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            str(count), ha='center', va='bottom', fontweight='bold')

# 3. Topic diversity (entropy)
ax = axes[1, 0]
topic_entropies = []

for topic in lda.components_:
    # Normalize to probabilities
    prob = topic / topic.sum()
    # Calculate entropy
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    topic_entropies.append(entropy)

ax.bar(range(n_topics), topic_entropies, color='lightgreen', alpha=0.7)
ax.set_xlabel('Topic')
ax.set_ylabel('Entropy (Diversity)')
ax.set_title('Topic Word Distribution Diversity')
ax.set_xticks(range(n_topics))
ax.set_xticklabels([f'T{i}' for i in range(n_topics)])
ax.grid(axis='y', alpha=0.3)

# 4. Average topic coherence per document
ax = axes[1, 1]
doc_coherence = []

for i in range(50):  # Sample 50 documents
    doc_dist = doc_topic_dist[i]
    # Coherence = concentration on fewer topics (inverse entropy)
    coherence = -np.sum(doc_dist * np.log(doc_dist + 1e-10))
    doc_coherence.append(coherence)

ax.hist(doc_coherence, bins=20, color='salmon', alpha=0.7, edgecolor='black')
ax.set_xlabel('Coherence Score')
ax.set_ylabel('Number of Documents')
ax.set_title('Document Topic Coherence Distribution')
ax.axvline(x=np.mean(doc_coherence), color='red', linestyle='--', 
          label=f'Mean: {np.mean(doc_coherence):.2f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('topic_modeling_lda.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: topic_modeling_lda.png")

print("\n✅ Topic modeling demo completo!")
print("\n💡 LDA CONCEPTS:")
print("   - Topic: Probability distribution over words")
print("   - Document: Mixture of topics")
print("   - α (alpha): Document-topic density (higher = more topics per doc)")
print("   - β (beta): Topic-word density (higher = more words per topic)")
print("\n💡 CHOOSING NUM_TOPICS:")
print("   - Too few: Topics too broad, mix multiple concepts")
print("   - Too many: Topics too specific, redundant")
print("   - Methods: Perplexity, coherence score, domain knowledge")
print("\n💡 APPLICATIONS:")
print("   - Document clustering & organization")
print("   - Recommendation systems (content-based)")
print("   - Trend analysis in social media")
print("   - Academic literature review")
