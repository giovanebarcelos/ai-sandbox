#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1741-Tutorial04Busca
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 4: BUSCA POR SIMILARIDADE
=================================
Carrega o vector store criado na Parte 3 e realiza buscas semânticas
sobre os documentos. Demonstra conceitos de Top-K, threshold e
similaridade por cosseno.

Funciona em: Windows 10/11 | Linux
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════

MODELO_EMBED = "nomic-embed-text"
PASTA_DB = Path("./chroma_db")

# ═══════════════════════════════════════════════════════════════════


def separador(titulo=""):
    print()
    print("=" * 60)
    if titulo:
        print(f"  {titulo}")
        print("=" * 60)


def carregar_vector_store():
    """Carrega o vector store ChromaDB existente."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    if not PASTA_DB.exists():
        print(f"\n  ⚠️  Banco de dados não encontrado em {PASTA_DB}")
        print("  Execute primeiro a Parte 3 (GO1740-Tutorial03Embeddings.py)")
        return None

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)
    vector_store = Chroma(
        persist_directory=str(PASTA_DB),
        embedding_function=embeddings,
    )

    print(f"  ✅ Vector store carregado: {PASTA_DB.resolve()}")
    return vector_store


def buscar(vector_store, pergunta, k=5, score_threshold=0.0):
    """Realiza uma busca por similaridade e retorna documentos com scores."""
    docs_com_scores = vector_store.similarity_search_with_relevance_scores(
        pergunta,
        k=k,
        score_threshold=score_threshold,
    )
    return docs_com_scores


def exibir_resultados(pergunta, resultados):
    """Exibe os resultados da busca de forma visual."""
    print(f"\n  🔍 Pergunta: \"{pergunta}\"")
    print(f"  {'─' * 50}")

    if not resultados:
        print("  ⚠️  Nenhum resultado encontrado.")
        return

    print(f"\n  📊 Top-{len(resultados)} resultados:")
    print()

    for i, (doc, score) in enumerate(resultados, 1):
        arquivo = doc.metadata.get("arquivo", "desconhecido")
        pagina = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", i)
        texto = doc.page_content[:200].replace("\n", " ").strip()

        # Barra de score visual
        barra = "█" * int(score * 20) + "░" * (20 - int(score * 20))

        print(f"  ┌── #{i}  Score: {score:.3f}  {barra}")
        print(f"  │  📁 {arquivo}" + (f" (pág. {pagina})" if pagina is not None else ""))
        print(f"  │  💬 \"{texto}...\"")
        print(f"  └──")

    print(f"  ⏱  Busca concluída em tempo real")


def comparar_k_values(vector_store):
    """Compara resultados com diferentes valores de K."""
    perguntas_teste = [
        "O que é uma rede neural?",
        "Como funciona o aprendizado supervisionado?",
        "O que é Ollama?",
    ]

    for pergunta in perguntas_teste:
        separador(f"PERGUNTA: {pergunta}")

        for k in [1, 3, 5]:
            resultados = buscar(vector_store, pergunta, k=k)

            if resultados:
                scores = [s for _, s in resultados]
                score_medio = sum(scores) / len(scores)
                print(f"\n  K={k}: score médio = {score_medio:.3f}")
            else:
                print(f"\n  K={k}: sem resultados")


def modo_interativo(vector_store):
    """Modo interativo onde o usuário digita perguntas."""
    print()
    print("  💬 MODO INTERATIVO DE BUSCA")
    print("  Digite suas perguntas (ou 'sair' para encerrar)")
    print()

    while True:
        try:
            pergunta = input("  🧑 Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "quit", "exit", "q"):
            break

        resultados = buscar(vector_store, pergunta, k=3)
        exibir_resultados(pergunta, resultados)


def main():
    """Função principal."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  TUTORIAL AULA 17 - PARTE 4: BUSCA          ║")
    print("║  Busca por Similaridade Semântica           ║")
    print("╚══════════════════════════════════════════════╝")

    # 1. Carregar vector store
    separador("1. CARREGANDO VECTOR STORE")
    vector_store = carregar_vector_store()

    if vector_store is None:
        return

    # 2. Busca simples
    separador("2. BUSCA SIMPLES (TOP-3)")
    pergunta_exemplo = "O que é Inteligência Artificial?"
    resultados = buscar(vector_store, pergunta_exemplo, k=3)
    exibir_resultados(pergunta_exemplo, resultados)

    # 3. Comparar K values
    separador("3. COMPARAÇÃO: K=1 vs K=3 vs K=5")
    comparar_k_values(vector_store)

    # 4. Modo interativo
    separador("4. MODO INTERATIVO")
    modo_interativo(vector_store)

    print()
    print("  ✅ Busca finalizada!")
    print()
    print("  Próximo passo: python GO1742-Tutorial05ChatRAG.py")


if __name__ == "__main__":
    main()
