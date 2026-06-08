#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1744-Tutorial07Completo
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 7: PIPELINE RAG COMPLETO
=================================
Pipeline completo integrado que executa todas as etapas:
  1. Setup e verificação
  2. Ingestão de documentos
  3. Chunking e embeddings
  4. Vector store (ChromaDB)
  5. Chat RAG interativo
  6. Relatório final com estatísticas

Funciona em: Windows 10/11 | Linux

Uso:
  python GO1744-Tutorial07Completo.py          # Pipeline completo
  python GO1744-Tutorial07Completo.py --chat   # Apenas chat (se já indexado)
  python GO1744-Tutorial07Completo.py --reindex # Reindexar documentos
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════

MODELO_LLM = "llama3.2:3b"
MODELO_EMBED = "nomic-embed-text"
PASTA_DOCS = Path("./docs")
PASTA_DB = Path("./chroma_db")
ARQUIVO_HISTORICO = Path("./logs/historico_completo.json")
ARQUIVO_RELATORIO = Path("./logs/relatorio_final.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ═══════════════════════════════════════════════════════════════════


def log(msg):
    """Log com timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def separador(titulo=""):
    print()
    print("=" * 60)
    if titulo:
        print(f"  {titulo}")
        print("=" * 60)


def verificar_ollama():
    """Verifica se Ollama está rodando."""
    try:
        import ollama
        modelos = ollama.list()
        nomes = [m["name"] for m in modelos.get("models", [])]
        log(f"✅ Ollama disponível. Modelos: {nomes[:3]}")
        return True
    except Exception:
        log("⚠️  Ollama não disponível (modo fallback)")
        return False


# ═══════════════════════════════════════════════════════════════════
# ETAPA 1: INGESTÃO
# ═══════════════════════════════════════════════════════════════════

def etapa_ingestao():
    """Carrega documentos da pasta docs."""
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, Docx2txtLoader,
    )

    log("📂 Verificando documentos...")

    if not PASTA_DOCS.exists():
        PASTA_DOCS.mkdir(parents=True)
        log(f"   📁 Pasta {PASTA_DOCS}/ criada")
        return []

    documentos = []
    for ext, loader_cls in [
        (".pdf", PyPDFLoader),
        (".txt", TextLoader),
        (".md", TextLoader),
    ]:
        for arquivo in sorted(PASTA_DOCS.glob(f"*{ext}")):
            try:
                if ext in (".txt", ".md"):
                    try:
                        loader = loader_cls(str(arquivo), encoding="utf-8")
                        docs = loader.load()
                    except UnicodeDecodeError:
                        loader = loader_cls(str(arquivo), encoding="latin-1")
                        docs = loader.load()
                else:
                    loader = loader_cls(str(arquivo))
                    docs = loader.load()

                for doc in docs:
                    doc.metadata["arquivo"] = arquivo.name
                documentos.extend(docs)
                log(f"   ✅ {arquivo.name}: {len(docs)} trecho(s)")
            except Exception as e:
                log(f"   ⚠️  {arquivo.name}: {e}")

    # DOCX
    for arquivo in sorted(PASTA_DOCS.glob("*.docx")):
        try:
            loader = Docx2txtLoader(str(arquivo))
            docs = loader.load()
            for doc in docs:
                doc.metadata["arquivo"] = arquivo.name
            documentos.extend(docs)
            log(f"   ✅ {arquivo.name}: {len(docs)} trecho(s)")
        except Exception as e:
            log(f"   ⚠️  {arquivo.name}: {e}")

    log(f"📦 Total: {len(documentos)} trechos de documentos")
    return documentos


# ═══════════════════════════════════════════════════════════════════
# ETAPA 2: CHUNKING
# ═══════════════════════════════════════════════════════════════════

def etapa_chunking(documentos):
    """Divide documentos em chunks."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    if not documentos:
        log("⚠️  Nenhum documento para chunking")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documentos)
    log(f"🔪 Chunking: {len(documentos)} docs → {len(chunks)} chunks")
    log(f"   Tamanho: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars")

    return chunks


# ═══════════════════════════════════════════════════════════════════
# ETAPA 3: VECTOR STORE
# ═══════════════════════════════════════════════════════════════════

def etapa_vector_store(chunks):
    """Cria/atualiza o vector store."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    import ollama

    log(f"🧮 Gerando embeddings...")
    log(f"   Modelo: {MODELO_EMBED}")

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)

    # Se banco já existe e temos chunks novos, adiciona
    if PASTA_DB.exists() and chunks:
        vector_store = Chroma(
            persist_directory=str(PASTA_DB),
            embedding_function=embeddings,
        )
        vector_store.add_documents(chunks)
        log(f"   ➕ Adicionados {len(chunks)} novos chunks ao banco existente")
    elif chunks:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(PASTA_DB),
        )
        log(f"   ✅ Novo banco criado com {len(chunks)} chunks")
    elif PASTA_DB.exists():
        vector_store = Chroma(
            persist_directory=str(PASTA_DB),
            embedding_function=embeddings,
        )
        log(f"   ✅ Banco existente carregado")
    else:
        log("❌ Nenhum banco disponível")
        return None

    try:
        vector_store.persist()
    except Exception:
        pass

    return vector_store


# ═══════════════════════════════════════════════════════════════════
# ETAPA 4: CHAT RAG
# ═══════════════════════════════════════════════════════════════════

def etapa_chat(vector_store, usar_ollama):
    """Modo chat interativo."""
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║       PIPELINE RAG - CHAT INTERATIVO        ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Modelo: {MODELO_LLM:<33s}║")
    print(f"  ║  Documentos: {len(list(PASTA_DOCS.glob('*'))):<3d} arquivos{' ' * 24}║")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║  Digite sua pergunta ou 'sair'              ║")
    print("  ╚══════════════════════════════════════════════╝")

    historico = []
    total_perguntas = 0
    tempos_resposta = []

    while True:
        try:
            pergunta = input("\n  🧑 Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "quit", "exit", "q"):
            break

        # Busca
        import time
        inicio = time.time()

        docs = vector_store.similarity_search_with_relevance_scores(
            pergunta, k=3
        )

        if not docs:
            print("\n  ⚠️  Nenhum documento relevante encontrado.")
            continue

        if usar_ollama:
            # Constrói prompt
            contexto = ""
            for i, (doc, score) in enumerate(docs, 1):
                fonte = doc.metadata.get("arquivo", "desconhecido")
                contexto += f"\nDocumento {i} [Fonte: {fonte}]\n"
                contexto += doc.page_content.strip() + "\n"

            prompt = f"""Você é um assistente que responde com base EXCLUSIVAMENTE nos documentos abaixo.

CONTEXTO:
{contexto}

REGRAS:
- Responda APENAS com base no contexto acima
- Se não souber, diga "Não encontrei essa informação nos documentos"
- Cite a fonte (nome do arquivo) de cada informação
- Responda em português

PERGUNTA: {pergunta}

RESPOSTA:"""

            print("  ⏳", end="", flush=True)
            import ollama
            response = ollama.generate(
                model=MODELO_LLM,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.3},
            )
            resposta = response["response"].strip()
        else:
            resposta = "[Modo fallback] Documentos encontrados, mas Ollama não está disponível para gerar resposta."
            for i, (doc, score) in enumerate(docs[:2], 1):
                arquivo = doc.metadata.get("arquivo", "?")
                resposta += f"\n  {i}. {arquivo} (relevância: {score:.1%})"

        fim = time.time()
        tempo = fim - inicio
        tempos_resposta.append(tempo)
        total_perguntas += 1

        # Exibe
        print(f"\r  🤖 Assistente:")
        print(f"  {'─' * 50}")
        print(f"  {resposta}")
        print(f"  {'─' * 50}")

        # Fontes
        fontes = list(set(
            (doc.metadata.get("arquivo", "?"), f"{score:.1%}")
            for doc, score in docs
        ))
        if fontes:
            print(f"\n  📚 Fontes:")
            for nome, score in fontes:
                print(f"     • {nome} (relevância: {score})")

        print(f"  ⏱  {tempo:.1f}s")

        # Histórico
        historico.append({"role": "user", "content": pergunta, "tempo": tempo})
        historico.append({"role": "assistant", "content": resposta})

    # Salva histórico
    if historico:
        Path("./logs").mkdir(parents=True, exist_ok=True)
        with open(ARQUIVO_HISTORICO, "w", encoding="utf-8") as f:
            json.dump(historico, f, ensure_ascii=False, indent=2)
        log(f"💾 Histórico salvo: {ARQUIVO_HISTORICO}")

    return total_perguntas, tempos_resposta


# ═══════════════════════════════════════════════════════════════════
# ETAPA 5: RELATÓRIO
# ═══════════════════════════════════════════════════════════════════

def gerar_relatorio(num_docs, num_chunks, total_perguntas, tempos):
    """Gera relatório final com estatísticas."""
    relatorio = {
        "data": datetime.now().isoformat(),
        "modelo_llm": MODELO_LLM,
        "modelo_embed": MODELO_EMBED,
        "documentos": num_docs,
        "chunks": num_chunks,
        "total_perguntas": total_perguntas,
        "tempo_medio_resposta": (
            round(sum(tempos) / len(tempos), 1) if tempos else 0
        ),
        "tempo_total": round(sum(tempos), 1) if tempos else 0,
    }

    # Salva relatório
    Path("./logs").mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    # Exibe
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║        PIPELINE RAG - RELATÓRIO FINAL        ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  📂 Documentos processados:  {num_docs:<3d}                 ║")
    print(f"║  📦 Total de chunks:         {num_chunks:<3d}                 ║")
    print(f"║  💾 Vector store:            ChromaDB           ║")
    print(f"║  🦙 Modelo LLM:              {MODELO_LLM:<20s}║")
    print(f"║  🧮 Modelo Embed:            {MODELO_EMBED:<20s}║")
    print(f"║{' ' * 50}║")
    print(f"║  ⏱ Tempo médio resposta:     {relatorio['tempo_medio_resposta']:<3.1f}s{' ' * 24}║")
    print(f"║  📊 Total de perguntas:      {total_perguntas:<3d}{' ' * 24}║")
    print(f"║{' ' * 50}║")
    print(f"║  📁 Relatório salvo em:       logs/            ║")
    print("╚══════════════════════════════════════════════╝")

    return relatorio


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def pipeline_completo():
    """Executa o pipeline completo."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║    TUTORIAL AULA 17 - PARTE 7: PIPELINE     ║")
    print("║         RAG Completo                        ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Modelo LLM:  {MODELO_LLM}")
    print(f"  Modelo Emb:  {MODELO_EMBED}")
    print(f"  Pasta docs:  {PASTA_DOCS.resolve()}")
    print(f"  Pasta db:    {PASTA_DB.resolve()}")

    # 0. Verificar Ollama
    usar_ollama = verificar_ollama()

    # 1. Ingestão
    separador("ETAPA 1: INGESTÃO DE DOCUMENTOS")
    documentos = etapa_ingestao()

    # 2. Chunking
    separador("ETAPA 2: CHUNKING")
    chunks = etapa_chunking(documentos)

    # 3. Vector Store
    separador("ETAPA 3: VECTOR STORE (ChromaDB)")
    vector_store = etapa_vector_store(chunks)

    if vector_store is None:
        log("❌ Não foi possível criar/carregar o vector store")
        return

    # 4. Chat
    separador("ETAPA 4: CHAT RAG")
    total_perguntas, tempos = etapa_chat(vector_store, usar_ollama)

    # 5. Relatório
    separador("RELATÓRIO FINAL")
    gerar_relatorio(
        num_docs=len(documentos),
        num_chunks=len(chunks),
        total_perguntas=total_perguntas,
        tempos=tempos,
    )

    print()
    log("🎉 Pipeline concluído!")
    print()
    print("  💡 Para usar a interface web:")
    print("     streamlit run GO1743-Tutorial06Interface.py")
    print()
    print("  💡 Para apenas conversar (se já indexou antes):")
    print("     python GO1744-Tutorial07Completo.py --chat")


def apenas_chat():
    """Apenas carrega banco existente e entra no chat."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║        PIPELINE RAG - MODO CHAT             ║")
    print("╚══════════════════════════════════════════════╝")

    if not PASTA_DB.exists():
        log(f"❌ Banco não encontrado em {PASTA_DB}")
        log("   Execute primeiro: python GO1744-Tutorial07Completo.py")
        return

    usar_ollama = verificar_ollama()

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)
    vector_store = Chroma(
        persist_directory=str(PASTA_DB),
        embedding_function=embeddings,
    )
    log(f"✅ Banco carregado: {PASTA_DB.resolve()}")

    total_perguntas, tempos = etapa_chat(vector_store, usar_ollama)

    if total_perguntas > 0:
        gerar_relatorio(
            num_docs=0,
            num_chunks=0,
            total_perguntas=total_perguntas,
            tempos=tempos,
        )


def reindexar():
    """Remove banco existente e reindexa do zero."""
    import shutil

    if PASTA_DB.exists():
        shutil.rmtree(PASTA_DB)
        log(f"♻️  Banco antigo removido: {PASTA_DB}")

    pipeline_completo()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG Completo - Tutorial Aula 17"
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Apenas modo chat (se já indexou antes)"
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Reindexar documentos do zero"
    )
    args = parser.parse_args()

    if args.chat:
        apenas_chat()
    elif args.reindex:
        reindexar()
    else:
        pipeline_completo()


if __name__ == "__main__":
    main()
