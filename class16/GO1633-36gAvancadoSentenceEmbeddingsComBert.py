# GO1633-36gAvançadoSentenceEmbeddingsComBert
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class BertSentenceEmbeddings:
    """Gerar e analisar embeddings de sentenças com BERT"""

    def __init__(self, model_name='bert-base-uncased'):
        print(f"🔄 Carregando {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        print("✅ Modelo carregado!")

    def get_sentence_embedding(self, sentence, pooling='cls'):
        """
        Obter embedding de sentença
        pooling: 'cls', 'mean', 'max'
        """
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        if pooling == 'cls':
            # Usar token [CLS]
            embedding = last_hidden_state[0, 0, :].numpy()
        elif pooling == 'mean':
            # Média de todos tokens (exceto padding)
            mask = inputs['attention_mask'][0].numpy()
            embedding = (last_hidden_state[0] * mask.reshape(-1, 1)).sum(dim=0) / mask.sum()
            embedding = embedding.numpy()
        elif pooling == 'max':
            # Max pooling
            embedding = last_hidden_state[0].max(dim=0)[0].numpy()
        else:
            raise ValueError(f"Pooling '{pooling}' não suportado")

        return embedding

    def compute_similarity_matrix(self, sentences, pooling='cls'):
        """Calcular matriz de similaridade entre sentenças"""

        print(f"🔄 Gerando embeddings ({pooling} pooling)...")
        embeddings = []
        for sent in sentences:
            emb = self.get_sentence_embedding(sent, pooling=pooling)
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix, embeddings

    def visualize_similarity(self, sentences, pooling='cls'):
        """Visualizar similaridade entre sentenças"""

        sim_matrix, embeddings = self.compute_similarity_matrix(sentences, pooling)

        # Plot heatmap
        plt.figure(figsize=(12, 10))

        # Nomes curtos para labels
        labels = [f"S{i+1}" for i in range(len(sentences))]

        sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Cosine Similarity'},
                   vmin=0, vmax=1)

        plt.title(f'Sentence Similarity Matrix (BERT {pooling.upper()} pooling)', fontsize=14)
        plt.xlabel('Sentence')
        plt.ylabel('Sentence')

        # Adicionar legendas
        legend_text = "\n".join([f"{labels[i]}: {sent[:60]}..." for i, sent in enumerate(sentences)])
        plt.figtext(0.5, -0.05, legend_text, ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig(f'bert_similarity_{pooling}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return sim_matrix, embeddings

    def visualize_embeddings_pca(self, sentences, pooling='cls'):
        """Visualizar embeddings em 2D usando PCA"""

        _, embeddings = self.compute_similarity_matrix(sentences, pooling)

        # PCA para 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))

        # Plot pontos
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            s=200, c=range(len(sentences)), cmap='tab10', 
                            alpha=0.7, edgecolors='black', linewidth=2)

        # Anotar
        for i, (x, y) in enumerate(embeddings_2d):
            plt.annotate(f'S{i+1}', (x, y), fontsize=12, fontweight='bold',
                        ha='center', va='center')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title(f'BERT Sentence Embeddings - PCA Visualization ({pooling.upper()} pooling)\n' +
                 f'Total explained variance: {pca.explained_variance_ratio_.sum():.1%}', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)

        # Legenda
        legend_labels = [f"S{i+1}: {sent[:40]}..." for i, sent in enumerate(sentences)]
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=plt.cm.tab10(i/10), 
                                      markersize=10, label=legend_labels[i])
                          for i in range(len(sentences))],
                  loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

        plt.tight_layout()
        plt.savefig(f'bert_embeddings_pca_{pooling}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compare_pooling_strategies(self, sentences):
        """Comparar diferentes estratégias de pooling"""

        pooling_methods = ['cls', 'mean', 'max']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, pooling in enumerate(pooling_methods):
            sim_matrix, _ = self.compute_similarity_matrix(sentences, pooling)

            labels = [f"S{i+1}" for i in range(len(sentences))]

            sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx], vmin=0, vmax=1, cbar=True)
            axes[idx].set_title(f'{pooling.upper()} Pooling', fontsize=12)

        plt.suptitle('Comparison of Pooling Strategies', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('bert_pooling_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def find_most_similar(self, query, candidates, pooling='cls', top_k=3):
        """Encontrar sentenças mais similares à query"""

        # Embedding da query
        query_emb = self.get_sentence_embedding(query, pooling).reshape(1, -1)

        # Embeddings dos candidatos
        candidate_embs = []
        for cand in candidates:
            emb = self.get_sentence_embedding(cand, pooling)
            candidate_embs.append(emb)
        candidate_embs = np.array(candidate_embs)

        # Similaridades
        similarities = cosine_similarity(query_emb, candidate_embs)[0]

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        print(f"\n🔍 Query: \"{query}\"\n")
        print(f"Top-{top_k} most similar sentences:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"\n{rank}. Similarity: {similarities[idx]:.4f}")
            print(f"   \"{candidates[idx]}\"")

        # Visualizar
        plt.figure(figsize=(12, 6))
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_sims = similarities[sorted_indices]

        colors = ['green' if i < top_k else 'gray' for i in range(len(candidates))]
        bars = plt.barh(range(len(candidates)), sorted_sims, color=colors, alpha=0.7)

        plt.yticks(range(len(candidates)), 
                  [f"C{sorted_indices[i]+1}" for i in range(len(candidates))],
                  fontsize=10)
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.title(f'Similarity to Query: "{query[:50]}..."\n(Green = Top-{top_k})', fontsize=14)
        plt.xlim(0, 1)
        plt.grid(axis='x', alpha=0.3)

        for i, sim in enumerate(sorted_sims):
            plt.text(sim, i, f' {sim:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('bert_similarity_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()

# Inicializar sistema


if __name__ == "__main__":
    embedder = BertSentenceEmbeddings()

    # Teste 1: Similaridade entre sentenças relacionadas
    print("\n" + "="*80)
    print("TESTE 1: Similaridade Entre Sentenças")
    print("="*80)

    sentences_tech = [
        "Artificial intelligence is transforming the world.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "The weather is nice today.",
        "I love eating pizza.",
        "Neural networks mimic the human brain.",
    ]

    sim_matrix, embeddings = embedder.visualize_similarity(sentences_tech, pooling='cls')
    embedder.visualize_embeddings_pca(sentences_tech, pooling='cls')

    # Teste 2: Comparar métodos de pooling
    print("\n" + "="*80)
    print("TESTE 2: Comparação de Métodos de Pooling")
    print("="*80)

    embedder.compare_pooling_strategies(sentences_tech)

    # Teste 3: Busca semântica
    print("\n" + "="*80)
    print("TESTE 3: Busca Semântica")
    print("="*80)

    query = "What is deep learning?"
    candidates = [
        "Deep learning is a type of machine learning based on neural networks.",
        "I went to the store yesterday.",
        "Neural networks have multiple layers.",
        "The capital of France is Paris.",
        "Machine learning algorithms learn from data.",
        "Pizza is my favorite food.",
        "Artificial neural networks are inspired by the brain.",
    ]

    embedder.find_most_similar(query, candidates, pooling='mean', top_k=3)

    # Teste 4: Detecção de paráfrases
    print("\n" + "="*80)
    print("TESTE 4: Detecção de Paráfrases")
    print("="*80)

    paraphrase_pairs = [
        "The cat is on the mat.",
        "A cat sits on the mat.",
        "The dog is in the park.",
        "On the mat, there is a cat.",
        "I enjoy reading books.",
    ]

    sim_para, _ = embedder.visualize_similarity(paraphrase_pairs, pooling='mean')

    print("\n✅ Análise completa!")
    print("\n📊 APLICAÇÕES:")
    print("   - Busca semântica em documentos")
    print("   - Detecção de duplicatas/paráfrases")
    print("   - Agrupamento de textos similares")
    print("   - Recomendação de conteúdo")
    print("   - Q&A systems")
