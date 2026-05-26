"""
GO1703 - Busca com Filtro de Metadados em Vector Store
=======================================================
Demonstra como usar filtros de metadados para restringir buscas no RAG.
Requer apenas bibliotecas padrão (simulação sem dependências externas).

Conceito: similarity_search com 'filter' permite combinar busca semântica
(encontrar textos semanticamente próximos) com filtragem estruturada
(garantir que só documentos de uma fonte/data específica sejam retornados).
Isso é chamado de "filtered nearest neighbor search".
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Documento:
    texto: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vetor: List[float] = field(default_factory=list)


def vetorizar(texto: str) -> List[float]:
    """Embedding simplificado baseado em frequência de caracteres."""
    vetor = [0.0] * 8
    for palavra in texto.lower().split():
        for i, char in enumerate(palavra[:4]):
            vetor[i % 8] += ord(char) / 1000.0
    norma = math.sqrt(sum(v ** 2 for v in vetor)) or 1.0
    return [v / norma for v in vetor]


def cosseno(a: List[float], b: List[float]) -> float:
    """Similaridade de cosseno entre dois vetores normalizados."""
    return sum(x * y for x, y in zip(a, b))


class VectorStoreComFiltro:
    """Vector store que suporta filtragem por metadados antes da busca vetorial."""

    def __init__(self):
        self.documentos: List[Documento] = []

    def add(self, texto: str, meta: Dict) -> None:
        self.documentos.append(Documento(texto=texto, metadata=meta, vetor=vetorizar(texto)))

    def similarity_search(
        self,
        query: str,
        filter: Dict[str, Any] = None,
        k: int = 3,
    ) -> List[Documento]:
        """
        Filtra primeiro por metadados, depois ordena por similaridade.
        Sem filtro: busca em todos os documentos.
        Com filtro: restringe o espaço de busca antes de medir similaridade.
        """
        query_vec = vetorizar(query)

        candidatos = self.documentos
        if filter:
            # Manter apenas documentos que satisfazem TODOS os critérios do filtro
            candidatos = [
                d for d in candidatos
                if all(d.metadata.get(chave) == valor for chave, valor in filter.items())
            ]

        ordenados = sorted(candidatos, key=lambda d: cosseno(query_vec, d.vetor), reverse=True)
        return ordenados[:k]


# ─────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1703 - BUSCA COM FILTRO DE METADADOS")
    print("=" * 60)

    vs = VectorStoreComFiltro()

    # Documentos de múltiplas fontes e datas
    dados = [
        ("Política de férias: 30 dias anuais para CLT.",
         {"source": "manual.pdf", "date": "2025-01"}),
        ("Férias devem ser solicitadas com antecedência.",
         {"source": "manual.pdf", "date": "2025-01"}),
        ("Novo benefício de vale-cultura a partir de 2025-03.",
         {"source": "comunicado.pdf", "date": "2025-03"}),
        ("Atualização da política de férias: parcelamento em 3x.",
         {"source": "manual.pdf", "date": "2025-06"}),
        ("Horas extras compensadas em banco de horas.",
         {"source": "manual.pdf", "date": "2025-01"}),
    ]
    for texto, meta in dados:
        vs.add(texto, meta)

    print(f"\n  Total de documentos no store: {len(vs.documentos)}")

    # Sem filtro: retorna o mais similar independente da fonte/data
    print("\n[SEM FILTRO] Query: 'política de férias'")
    for doc in vs.similarity_search("política de férias", k=3):
        print(f"  {doc.metadata} -> '{doc.texto[:55]}...'")

    # Com filtro: restringe a manual.pdf de 2025-01 APENAS
    print("\n[COM FILTRO source=manual.pdf, date=2025-01] Query: 'política de férias'")
    results = vs.similarity_search(
        "política de férias",
        filter={"source": "manual.pdf", "date": "2025-01"},
        k=3,
    )
    for doc in results:
        print(f"  {doc.metadata} -> '{doc.texto[:55]}...'")

    print()
    print("  Por que filtrar?")
    print("  - Evita retornar documentos desatualizados")
    print("  - Garante que a resposta do LLM usa a versao correta")
    print("  - Aumenta precisao sem custo de re-rankear todos os docs")
