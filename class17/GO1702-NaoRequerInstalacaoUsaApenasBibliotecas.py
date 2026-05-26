"""
GO1702 - Metadados em Vector Stores
====================================
Demonstra como adicionar textos com metadados estruturados em um vector store.
Metadados permitem filtrar documentos por fonte, data, página etc.
Requer apenas bibliotecas padrão (simulação sem dependências externas).

Conceito: quando ingerimos documentos em um RAG, cada chunk carrega metadados
para rastreabilidade (fonte, data, página). Isso habilita filtros precisos
na recuperação, ex: "buscar só em manual.pdf de 2025-01".
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import math


@dataclass
class Documento:
    texto: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vetor: List[float] = field(default_factory=list)


class VectorStoreSimulado:
    """
    Simula um vector store com suporte a metadados e filtragem.
    Em produção seria ChromaDB, FAISS ou Pinecone.
    """

    def __init__(self):
        self.documentos: List[Documento] = []

    def _vetorizar_simples(self, texto: str) -> List[float]:
        """
        Embedding simplificado para demonstração.
        Em produção: sentence-transformers, OpenAI embeddings, etc.
        """
        palavras = texto.lower().split()
        # Representação hash-based: 8 dimensões
        vetor = [0.0] * 8
        for palavra in palavras:
            for i, char in enumerate(palavra[:4]):
                vetor[i % 8] += ord(char) / 1000.0
        # Normalizar (norma L2)
        norma = math.sqrt(sum(v ** 2 for v in vetor)) or 1.0
        return [v / norma for v in vetor]

    def add_texts(self, texts: List[str], metadatas: List[Dict]) -> None:
        """
        Ingere textos com metadados associados.
        Cada texto recebe um vetor (embedding) para busca semântica.
        """
        for texto, meta in zip(texts, metadatas):
            doc = Documento(
                texto=texto,
                metadata=meta,
                vetor=self._vetorizar_simples(texto),
            )
            self.documentos.append(doc)
        print(f"  [{self.__class__.__name__}] {len(texts)} documento(s) adicionado(s). "
              f"Total: {len(self.documentos)}")

    def similarity_search(
        self,
        query: str,
        filter: Dict[str, Any] = None,
        k: int = 3,
    ) -> List[Documento]:
        """
        Busca os k documentos mais próximos da query.
        filter: dict de metadados que TODOS os documentos retornados devem ter.
        """
        query_vec = self._vetorizar_simples(query)

        # Aplicar filtro por metadados ANTES da busca semântica
        candidatos = self.documentos
        if filter:
            candidatos = [
                d for d in candidatos
                if all(d.metadata.get(k) == v for k, v in filter.items())
            ]

        # Calcular similaridade de cosseno
        def cosseno(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot  # vetores já normalizados

        resultados = sorted(candidatos, key=lambda d: cosseno(query_vec, d.vetor), reverse=True)
        return resultados[:k]


# ─────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1702 - METADADOS EM VECTOR STORES")
    print("=" * 60)

    vectorstore = VectorStoreSimulado()

    # Ingerir documentos com metadados estruturados
    # Metadados permitem filtrar por fonte, data e página posteriormente
    chunks = [
        "Funcionários têm direito a 30 dias de férias por ano.",
        "As férias devem ser solicitadas com 30 dias de antecedência.",
        "Férias podem ser parceladas em até 3 períodos distintos.",
        "O pagamento das férias é feito antes do início do período.",
        "O produto X tem garantia de 12 meses contra defeito de fabricação.",
    ]

    vectorstore.add_texts(
        texts=chunks,
        metadatas=[
            {"source": "manual.pdf", "page": 1, "date": "2025-01"},
            {"source": "manual.pdf", "page": 2, "date": "2025-01"},
            {"source": "manual.pdf", "page": 2, "date": "2025-01"},
            {"source": "manual.pdf", "page": 3, "date": "2025-01"},
            {"source": "produto.pdf", "page": 1, "date": "2025-03"},
        ],
    )

    print()
    print("Busca sem filtro: 'política de férias'")
    resultados = vectorstore.similarity_search("política de férias", k=3)
    for i, doc in enumerate(resultados, 1):
        print(f"  {i}. [{doc.metadata}]")
        print(f"     '{doc.texto[:60]}...'")

    print()
    print("Busca COM filtro source=manual.pdf, date=2025-01:")
    resultados_filtrados = vectorstore.similarity_search(
        "política de férias",
        filter={"source": "manual.pdf", "date": "2025-01"},
        k=3,
    )
    for i, doc in enumerate(resultados_filtrados, 1):
        print(f"  {i}. [{doc.metadata}]")
        print(f"     '{doc.texto[:60]}...'")

    print()
    print("  Metadados sao essenciais para RAG de producao:")
    print("  - Rastreabilidade: saber de qual documento veio cada trecho")
    print("  - Filtragem temporal: buscar so documentos recentes")
    print("  - Controle de acesso: filtrar por usuario/departamento")
