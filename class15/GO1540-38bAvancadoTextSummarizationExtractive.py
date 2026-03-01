# GO1540-38bAvançadoTextSummarizationExtractive
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize


if __name__ == "__main__":
    nltk.download('punkt', quiet=True)

    class TextSummarizer:
        """Resumidor de texto usando algoritmos extractive"""

        def __init__(self, method='tfidf'):
            self.method = method

        def tfidf_summarize(self, text, num_sentences=3):
            """Resumo baseado em TF-IDF"""
            sentences = sent_tokenize(text)

            # TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calcular score de cada sentença (soma dos TF-IDF)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

            # Top sentenças
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Manter ordem original

            summary = ' '.join([sentences[i] for i in top_indices])

            return summary, sentence_scores, sentences

        def textrank_summarize(self, text, num_sentences=3, damping=0.85):
            """Resumo usando TextRank (similar ao PageRank)"""
            sentences = sent_tokenize(text)

            # Criar matriz de similaridade
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Criar grafo
            graph = nx.from_numpy_array(similarity_matrix)

            # PageRank
            scores = nx.pagerank(graph, alpha=damping)

            # Top sentenças
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), 
                                     reverse=True)
            top_indices = sorted([idx for _, idx, _ in ranked_sentences[:num_sentences]])

            summary = ' '.join([sentences[i] for i in top_indices])
            sentence_scores = np.array([scores[i] for i in range(len(sentences))])

            return summary, sentence_scores, sentences, similarity_matrix

        def visualize_comparison(self, text, num_sentences=3):
            """Comparar métodos de resumo"""

            # TF-IDF
            summary_tfidf, scores_tfidf, sentences = self.tfidf_summarize(text, num_sentences)

            # TextRank
            summary_textrank, scores_textrank, _, sim_matrix = self.textrank_summarize(text, num_sentences)

            # Visualização
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Scores TF-IDF
            axes[0, 0].barh(range(len(scores_tfidf)), scores_tfidf, color='skyblue', alpha=0.8)
            axes[0, 0].set_yticks(range(len(sentences)))
            axes[0, 0].set_yticklabels([f"S{i+1}" for i in range(len(sentences))], fontsize=9)
            axes[0, 0].set_xlabel('TF-IDF Score', fontsize=11)
            axes[0, 0].set_title('TF-IDF Sentence Scores', fontsize=13, fontweight='bold')
            axes[0, 0].invert_yaxis()
            axes[0, 0].grid(axis='x', alpha=0.3)

            # Marcar sentenças selecionadas
            top_tfidf = scores_tfidf.argsort()[-num_sentences:]
            for idx in top_tfidf:
                axes[0, 0].get_children()[idx].set_color('red')
                axes[0, 0].get_children()[idx].set_alpha(1.0)

            # 2. Scores TextRank
            axes[0, 1].barh(range(len(scores_textrank)), scores_textrank, color='coral', alpha=0.8)
            axes[0, 1].set_yticks(range(len(sentences)))
            axes[0, 1].set_yticklabels([f"S{i+1}" for i in range(len(sentences))], fontsize=9)
            axes[0, 1].set_xlabel('PageRank Score', fontsize=11)
            axes[0, 1].set_title('TextRank Sentence Scores', fontsize=13, fontweight='bold')
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(axis='x', alpha=0.3)

            # Marcar sentenças selecionadas
            top_textrank = scores_textrank.argsort()[-num_sentences:]
            for idx in top_textrank:
                axes[0, 1].get_children()[idx].set_color('red')
                axes[0, 1].get_children()[idx].set_alpha(1.0)

            # 3. Matriz de similaridade
            im = axes[1, 0].imshow(sim_matrix, cmap='YlOrRd', aspect='auto')
            axes[1, 0].set_xticks(range(len(sentences)))
            axes[1, 0].set_yticks(range(len(sentences)))
            axes[1, 0].set_xticklabels([f"S{i+1}" for i in range(len(sentences))], fontsize=9)
            axes[1, 0].set_yticklabels([f"S{i+1}" for i in range(len(sentences))], fontsize=9)
            axes[1, 0].set_title('Sentence Similarity Matrix (TextRank)', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 0], label='Cosine Similarity')

            # 4. Comparação de comprimento
            original_len = len(text.split())
            tfidf_len = len(summary_tfidf.split())
            textrank_len = len(summary_textrank.split())

            methods = ['Original', 'TF-IDF\nSummary', 'TextRank\nSummary']
            lengths = [original_len, tfidf_len, textrank_len]
            colors_comp = ['gray', 'skyblue', 'coral']

            bars = axes[1, 1].bar(methods, lengths, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
            axes[1, 1].set_ylabel('Word Count', fontsize=11)
            axes[1, 1].set_title('Summary Length Comparison', fontsize=13, fontweight='bold')
            axes[1, 1].grid(axis='y', alpha=0.3)

            for bar, length in zip(bars, lengths):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{length}\n({length/original_len*100:.0f}%)',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig('text_summarization_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

            return summary_tfidf, summary_textrank

    # Texto de exemplo (artigo sobre IA)
    article = """
    Artificial intelligence has made remarkable progress in recent years, transforming industries 
    and reshaping how we interact with technology. Machine learning algorithms can now recognize 
    images, understand speech, and even generate human-like text. Deep learning, a subset of 
    machine learning based on neural networks, has been particularly successful. Companies like 
    Google, Microsoft, and OpenAI are investing billions in AI research. The technology promises 
    to revolutionize healthcare by improving disease diagnosis and drug discovery. Self-driving 
    cars, powered by AI, are being tested on roads worldwide. However, AI also raises important 
    ethical questions about privacy, bias, and job displacement. Researchers are working on making 
    AI systems more transparent and fair. The future of AI will likely involve closer collaboration 
    between humans and machines, augmenting rather than replacing human capabilities. Governments 
    are beginning to regulate AI to ensure it benefits society. Education systems are adapting to 
    prepare students for an AI-driven world. Despite challenges, the potential benefits of AI for 
    humanity are enormous, from solving climate change to advancing scientific discovery.
    """

    print("="*70)
    print("TEXT SUMMARIZATION - COMPARAÇÃO DE MÉTODOS")
    print("="*70)

    # Criar summarizer
    summarizer = TextSummarizer()

    # Comparar métodos
    summary_tfidf, summary_textrank = summarizer.visualize_comparison(article, num_sentences=3)

    print(f"\n📄 TEXTO ORIGINAL ({len(article.split())} palavras):")
    print(article)

    print(f"\n📝 RESUMO TF-IDF ({len(summary_tfidf.split())} palavras):")
    print(summary_tfidf)

    print(f"\n📝 RESUMO TEXTRANK ({len(summary_textrank.split())} palavras):")
    print(summary_textrank)

    # Métricas de compressão
    compression_tfidf = len(summary_tfidf.split()) / len(article.split())
    compression_textrank = len(summary_textrank.split()) / len(article.split())

    print(f"\n📊 MÉTRICAS:")
    print(f"   TF-IDF compression: {compression_tfidf:.1%}")
    print(f"   TextRank compression: {compression_textrank:.1%}")

    print(f"\n✅ Análise completa!")
