"""
GO1711 - Pipeline RAG Completo (Simulado)
==========================================
Demonstra o pipeline completo de RAG (Retrieval-Augmented Generation)
sem dependências externas — apenas biblioteca padrão do Python.

Pipeline RAG:
  1. Ingestão: documentos são armazenados com metadados
  2. Recuperação: busca por palavra-chave (simula busca vetorial)
  3. Geração: resposta é gerada a partir do contexto recuperado
  4. Métricas: latência de cada etapa é registrada

Em produção:
  - Ingestão: embeddings com sentence-transformers + FAISS
  - Recuperação: busca por similaridade coseno
  - Geração: LLM (Ollama, OpenAI, Claude)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Document:
    """Representa um documento com texto e metadados."""
    text: str
    metadata: Dict


class SimpleRAG:
    """
    Pipeline RAG simplificado que usa busca por palavra-chave.
    Demonstra o fluxo sem dependências externas.
    """

    def __init__(self):
        # Repositório de documentos em memória
        self.docs: List[Document] = []

    def ingest(self, docs: List[Document]) -> None:
        """
        Etapa 1: Ingestão — armazena documentos para futura recuperação.
        Em produção: calcularia embeddings e salvaria em vector store.
        """
        self.docs.extend(docs)
        print(f"  Ingeridos: {len(docs)} documentos (total: {len(self.docs)})")

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Etapa 2: Recuperação — encontra documentos relevantes para a query.
        Usa contagem de palavras em comum (substitui similaridade vetorial).
        """
        scores = []
        query_words = set(query.lower().split())
        for doc in self.docs:
            doc_words = set(doc.text.lower().split())
            # Score = quantas palavras da query aparecem no documento
            score = len(query_words & doc_words)
            scores.append((score, doc))

        # Ordenar por relevância (maior score = mais relevante)
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scores[:k]]

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Etapa 3: Geração — gera resposta a partir do contexto recuperado.
        Simula o que um LLM faria com os documentos como contexto.
        """
        if not context_docs:
            return "Nenhum contexto relevante encontrado."

        # Em produção: passaria context para LLM como system prompt
        context_preview = context_docs[0].text[:80]
        return (f"Com base no contexto recuperado: '{context_preview}...', "
                f"a resposta para '{query}' está no documento acima.")

    def query(self, question: str) -> dict:
        """
        Pipeline completo: recupera documentos + gera resposta.
        Mede latência de cada etapa.
        """
        t0 = time.time()

        # Recuperação
        docs = self.retrieve(question, k=3)
        t_retrieval = time.time() - t0

        # Geração
        answer = self.generate(question, docs)
        t_total = time.time() - t0

        return {
            'question': question,
            'answer': answer,
            'sources': [d.metadata for d in docs],
            'retrieval_time_ms': round(t_retrieval * 1000, 2),
            'total_time_ms': round(t_total * 1000, 2),
            'n_docs_retrieved': len(docs),
        }


# ─────────────────────────────────────────────────────────────────
# CORPUS DE DEMONSTRAÇÃO
# ─────────────────────────────────────────────────────────────────

CORPUS_IA = [
    Document(
        "Machine learning é um subcampo da inteligência artificial que "
        "permite que sistemas aprendam a partir de dados sem serem explicitamente programados.",
        {'source': 'intro_ia.pdf', 'topico': 'machine learning', 'pagina': 1}
    ),
    Document(
        "Deep learning usa redes neurais com múltiplas camadas para aprender "
        "representações hierárquicas dos dados.",
        {'source': 'deep_learning.pdf', 'topico': 'redes neurais', 'pagina': 5}
    ),
    Document(
        "Transformers revolucionaram o processamento de linguagem natural (NLP) "
        "com o mecanismo de atenção, permitindo paralelização no treinamento.",
        {'source': 'transformers.pdf', 'topico': 'transformers', 'pagina': 2}
    ),
    Document(
        "RAG (Retrieval-Augmented Generation) combina recuperação de documentos "
        "com geração de texto por LLMs para reduzir alucinações.",
        {'source': 'rag_guide.pdf', 'topico': 'rag', 'pagina': 1}
    ),
    Document(
        "Reinforcement learning treina agentes através de recompensas e punições, "
        "aprendendo política ótima por tentativa e erro.",
        {'source': 'rl_overview.pdf', 'topico': 'reinforcement learning', 'pagina': 3}
    ),
]


if __name__ == "__main__":
    print("=" * 60)
    print("GO1711 - PIPELINE RAG COMPLETO (SIMULADO)")
    print("=" * 60)

    print("\nETAPAS DO PIPELINE:")
    print("  1. Ingestao   → documentos armazenados")
    print("  2. Recuperacao → documentos relevantes encontrados")
    print("  3. Geracao    → resposta gerada com contexto")

    print()
    print("─" * 60)
    print("INGESTAO:")
    print("─" * 60)
    rag = SimpleRAG()
    rag.ingest(CORPUS_IA)

    print()
    print("─" * 60)
    print("CONSULTAS:")
    print("─" * 60)

    perguntas = [
        "O que é machine learning?",
        "Como funcionam os transformers?",
        "O que é RAG e para que serve?",
        "Como redes neurais aprendem?",
    ]

    for pergunta in perguntas:
        result = rag.query(pergunta)
        print(f"\n  Pergunta: {result['question']}")
        print(f"  Resposta: {result['answer'][:100]}...")
        print(f"  Fontes:   {[s['source'] for s in result['sources']]}")
        print(f"  Latencia: recuperacao={result['retrieval_time_ms']}ms, "
              f"total={result['total_time_ms']}ms")

    print()
    print("─" * 60)
    print("LIMITACOES DESTA SIMULACAO:")
    print("─" * 60)
    print("  - Busca por palavras (sem semântica): 'carro' != 'veículo'")
    print("  - LLM simulado: resposta não usa o contexto de verdade")
    print("  - Em producao: usar FAISS + sentence-transformers + Ollama/OpenAI")
