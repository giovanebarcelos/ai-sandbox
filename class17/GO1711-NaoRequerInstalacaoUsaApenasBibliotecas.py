# GO1711-NãoRequerInstalaçãoUsaApenasBibliotecas
# Complete RAG pipeline demonstration
from dataclasses import dataclass
from typing import List
import time

@dataclass
class Document:
    text: str
    metadata: dict

class SimpleRAG:
    def __init__(self):
        self.docs = []

    def ingest(self, docs: List[Document]):
        """Ingest documents"""
        self.docs.extend(docs)
        print(f"✅ Ingested {len(docs)} documents")

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Simple keyword retrieval"""
        scores = []
        for doc in self.docs:
            # Simple scoring: count matching words
            query_words = set(query.lower().split())
            doc_words = set(doc.text.lower().split())
            score = len(query_words & doc_words)
            scores.append((score, doc))

        # Sort and return top k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scores[:k]]

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer from context"""
        context = "\n".join([d.text for d in context_docs])
        answer = f"Based on the context, {query[:50]}... [simulated answer]"
        return answer

    def query(self, question: str) -> dict:
        """Full RAG pipeline"""
        start = time.time()

        # Retrieve
        docs = self.retrieve(question, k=3)
        retrieval_time = time.time() - start

        # Generate
        answer = self.generate(question, docs)
        total_time = time.time() - start

        return {
            'question': question,
            'answer': answer,
            'sources': [d.metadata for d in docs],
            'retrieval_time': retrieval_time,
            'total_time': total_time
        }

# Demo


if __name__ == "__main__":
    rag = SimpleRAG()

    # Ingest
    docs = [
        Document("Machine learning is a subset of AI", {'source': 'ml.pdf'}),
        Document("Deep learning uses neural networks", {'source': 'dl.pdf'}),
        Document("Transformers revolutionized NLP", {'source': 'transformers.pdf'}),
    ]
    rag.ingest(docs)

    # Query
    result = rag.query("What is machine learning?")
    print(f"\n🔍 Query: {result['question']}")
    print(f"💬 Answer: {result['answer']}")
    print(f"📚 Sources: {result['sources']}")
    print(f"⏱️  Time: {result['total_time']:.3f}s")
