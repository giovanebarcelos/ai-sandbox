# GO1541-38cAvançadoTopicModelingComLda
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis.lda_model
import warnings
warnings.filterwarnings('ignore')

# Corpus de documentos
documents = [
    "Machine learning algorithms learn patterns from data to make predictions",
    "Deep learning uses neural networks with multiple layers for complex tasks",
    "Python is a popular programming language for data science and AI",
    "Natural language processing helps computers understand human language",
    "Computer vision enables machines to interpret and analyze visual information",
    "Reinforcement learning trains agents through rewards and penalties",
    "Data preprocessing is crucial for building accurate machine learning models",
    "TensorFlow and PyTorch are popular frameworks for deep learning",
    "Supervised learning uses labeled data to train predictive models",
    "Unsupervised learning discovers hidden patterns in unlabeled data",
    "The stock market showed significant volatility this week",
    "Economic indicators suggest potential recession in coming months",
    "Federal Reserve raised interest rates to combat inflation",
    "GDP growth exceeded expectations in the last quarter",
    "Unemployment rates reached historic lows this year",
    "Climate change poses serious risks to global ecosystems",
    "Renewable energy sources are becoming more cost-effective",
    "Carbon emissions must be reduced to prevent catastrophic warming",
    "Electric vehicles are gaining market share rapidly",
    "Solar and wind power installations are breaking records",
]

# Vetorizar documentos
vectorizer = CountVectorizer(max_features=100, stop_words='english', max_df=0.8, min_df=2)
doc_term_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("="*70)
print("TOPIC MODELING - LDA (Latent Dirichlet Allocation)")
print("="*70)
print(f"\nDocumentos: {len(documents)}")
print(f"Vocabulário: {len(feature_names)} termos")
print(f"Document-Term Matrix: {doc_term_matrix.shape}")

# Treinar LDA
n_topics = 3
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,
    learning_method='online',
    random_state=42,
    n_jobs=-1
)

doc_topic_matrix = lda.fit_transform(doc_term_matrix)

# Exibir tópicos
print(f"\n🔍 TÓPICOS DESCOBERTOS (Top 10 palavras por tópico):\n")

def display_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        topics.append((top_words, top_weights))

        print(f"Tópico {topic_idx + 1}:")
        print(f"  Palavras: {', '.join(top_words)}")
        print()

    return topics

topics_info = display_topics(lda, feature_names, n_top_words=10)

# Visualizações
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Heatmap: Document-Topic Distribution
axes[0, 0].imshow(doc_topic_matrix.T, cmap='YlOrRd', aspect='auto')
axes[0, 0].set_xlabel('Document Index', fontsize=11)
axes[0, 0].set_ylabel('Topic', fontsize=11)
axes[0, 0].set_title('Document-Topic Distribution Heatmap', fontsize=13, fontweight='bold')
axes[0, 0].set_yticks(range(n_topics))
axes[0, 0].set_yticklabels([f'Topic {i+1}' for i in range(n_topics)])
cbar = plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
cbar.set_label('Probability', fontsize=10)

# 2. Top words per topic (barras)
for topic_idx, (words, weights) in enumerate(topics_info):
    if topic_idx == 0:  # Plotar apenas primeiro tópico no subplot
        y_pos = np.arange(len(words))
        axes[0, 1].barh(y_pos, weights, color='skyblue', alpha=0.8)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(words, fontsize=9)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('Weight', fontsize=11)
        axes[0, 1].set_title(f'Topic 1 - Top Words', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Distribuição de tópicos dominantes por documento
dominant_topics = doc_topic_matrix.argmax(axis=1)
topic_counts = np.bincount(dominant_topics, minlength=n_topics)

axes[1, 0].bar(range(n_topics), topic_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
               alpha=0.8, edgecolor='black', linewidth=2)
axes[1, 0].set_xlabel('Topic', fontsize=11)
axes[1, 0].set_ylabel('Number of Documents', fontsize=11)
axes[1, 0].set_title('Dominant Topic Distribution', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(range(n_topics))
axes[1, 0].set_xticklabels([f'Topic {i+1}' for i in range(n_topics)])
axes[1, 0].grid(axis='y', alpha=0.3)

for i, count in enumerate(topic_counts):
    axes[1, 0].text(i, count, str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Box plot: Topic probability distributions
topic_data = [doc_topic_matrix[:, i] for i in range(n_topics)]
bp = axes[1, 1].boxplot(topic_data, labels=[f'T{i+1}' for i in range(n_topics)],
                        patch_artist=True, showmeans=True)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1, 1].set_xlabel('Topic', fontsize=11)
axes[1, 1].set_ylabel('Probability', fontsize=11)
axes[1, 1].set_title('Topic Probability Distribution (Box Plot)', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('topic_modeling_lda.png', dpi=300, bbox_inches='tight')
plt.show()

# Classificar documentos por tópico
print("="*70)
print("DOCUMENTOS CLASSIFICADOS POR TÓPICO")
print("="*70)

for topic_idx in range(n_topics):
    print(f"\n📂 TÓPICO {topic_idx + 1}:")
    # Documentos onde este tópico é dominante
    topic_docs = np.where(dominant_topics == topic_idx)[0]
    for doc_idx in topic_docs:
        prob = doc_topic_matrix[doc_idx, topic_idx]
        print(f"   Doc {doc_idx:2d} ({prob:.3f}): {documents[doc_idx][:60]}...")

# Perplexidade (menor = melhor)
perplexity = lda.perplexity(doc_term_matrix)
print(f"\n📊 MÉTRICAS:")
print(f"   Perplexity: {perplexity:.2f} (menor = melhor)")
print(f"   Log-likelihood: {lda.score(doc_term_matrix):.2f}")

print(f"\n✅ Topic Modeling completo!")
