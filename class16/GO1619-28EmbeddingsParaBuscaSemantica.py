# GO1619-28EmbeddingsParaBuscaSemântica
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(["Python é ótimo", "Eu amo programar"])
    similarity = cosine_similarity(embeddings[0], embeddings[1])
