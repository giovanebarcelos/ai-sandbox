#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1740-Tutorial03Embeddings
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 3: EMBEDDINGS E VECTOR STORE
=====================================
Divide documentos em chunks, gera embeddings com Ollama e armazena
no ChromaDB (banco de dados vetorial).

Funciona em: Windows 10/11 | Linux
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES (altere aqui se necessário)
# ═══════════════════════════════════════════════════════════════════

MODELO_EMBED = "nomic-embed-text"
PASTA_DOCS = Path("./docs")
PASTA_DB = Path("./chroma_db")

# Configurações de chunking
CHUNK_SIZE = 500       # Tamanho de cada chunk em caracteres
CHUNK_OVERLAP = 50     # Sobreposição entre chunks

# ═══════════════════════════════════════════════════════════════════


def separador(titulo=""):
    """Imprime um separador visual."""
    print()
    print("=" * 60)
    if titulo:
        print(f"  {titulo}")
        print("=" * 60)


def carregar_documentos():
    """Carrega todos os documentos da pasta docs."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
    )

    documentos = []
    erros = []

    for ext, loader_cls in [
        (".pdf", PyPDFLoader),
        (".txt", TextLoader),
        (".md", TextLoader),
    ]:
        for arquivo in sorted(PASTA_DOCS.glob(f"*{ext}")):
            try:
                if ext == ".txt" or ext == ".md":
                    try:
                        loader = loader_cls(str(arquivo), encoding="utf-8")
                        docs = loader.load()
                    except UnicodeDecodeError:
                        loader = loader_cls(str(arquivo), encoding="latin-1")
                        docs = loader.load()
                else:
                    loader = loader_cls(str(arquivo))
                    docs = loader.load()

                # Adiciona metadados úteis
                for doc in docs:
                    doc.metadata["arquivo"] = arquivo.name
                    doc.metadata["extensao"] = ext
                    doc.metadata["tipo"] = "PDF" if ext == ".pdf" else "Texto"

                documentos.extend(docs)
                print(f"     ✅ {arquivo.name}: {len(docs)} trecho(s)")
            except Exception as e:
                erros.append((arquivo.name, str(e)))
                print(f"     ⚠️  {arquivo.name}: erro ao carregar ({e})")

    # Tenta carregar DOCX separadamente (dependência opcional)
    for arquivo in sorted(PASTA_DOCS.glob("*.docx")):
        try:
            loader = Docx2txtLoader(str(arquivo))
            docs = loader.load()
            for doc in docs:
                doc.metadata["arquivo"] = arquivo.name
                doc.metadata["extensao"] = ".docx"
                doc.metadata["tipo"] = "Documento Word"
            documentos.extend(docs)
            print(f"     ✅ {arquivo.name}: {len(docs)} trecho(s)")
        except Exception as e:
            erros.append((arquivo.name, str(e)))
            print(f"     ⚠️  {arquivo.name}: erro ({e})")

    if erros:
        print(f"\n  ⚠️  Erros ao carregar {len(erros)} arquivo(s):")
        for nome, erro in erros:
            print(f"      - {nome}: {erro}")

    return documentos


def chunking(documentos):
    """Divide documentos em chunks usando RecursiveCharacterTextSplitter."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documentos)

    print(f"\n  📦 Total de chunks criados: {len(chunks)}")
    print(f"  📐 Tamanho do chunk: {CHUNK_SIZE} caracteres")
    print(f"  🔗 Overlap entre chunks: {CHUNK_OVERLAP} caracteres")

    # Estatísticas dos chunks
    if chunks:
        tamanhos = [len(c.page_content) for c in chunks]
        print(f"\n  📊 Estatísticas dos chunks:")
        print(f"     - Menor:    {min(tamanhos)} caracteres")
        print(f"     - Maior:    {max(tamanhos)} caracteres")
        print(f"     - Média:    {sum(tamanhos) / len(tamanhos):.0f} caracteres")

    return chunks


def criar_vector_store(chunks):
    """Cria um vector store ChromaDB com embeddings do Ollama."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    print(f"\n  🧮 Conectando ao Ollama para embeddings...")
    print(f"     Modelo: {MODELO_EMBED}")

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)

    # Remove diretório antigo se existir
    if PASTA_DB.exists():
        import shutil
        shutil.rmtree(PASTA_DB)
        print(f"     ♻️  Banco anterior removido")

    print(f"     ⏳ Gerando embeddings e indexando... (pode levar alguns minutos)")
    print(f"     📊 Total de chunks: {len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PASTA_DB),
    )

    # O ChromaDB moderno salva automaticamente, mas vamos garantir
    try:
        vector_store.persist()
    except Exception:
        pass  # ChromaDB >= 0.4 não precisa mais de persist()

    print(f"     ✅ Vector store criado: {PASTA_DB.resolve()}")
    print(f"     💾 {len(chunks)} vetores indexados com {len(chunks[0].page_content) if chunks else 0} caracteres cada")

    return vector_store


def visualizar_chunks(chunks, n=5):
    """Mostra exemplos dos chunks criados."""
    print(f"\n  📄 PRIMEIROS {min(n, len(chunks))} CHUNKS:")
    print(f"  {'─' * 50}")

    for i, chunk in enumerate(chunks[:n]):
        arquivo = chunk.metadata.get("arquivo", "desconhecido")
        pagina = chunk.metadata.get("page", "?")
        texto = chunk.page_content[:120].replace("\n", " ")

        print(f"\n  Chunk #{i + 1}")
        print(f"     📁 {arquivo} (pág. {pagina})" if pagina != "?" else f"     📁 {arquivo}")
        print(f"     💬 \"{texto}...\"")

    if len(chunks) > n:
        print(f"\n     ... e mais {len(chunks) - n} chunk(s)")


def verificar_ollama():
    """Verifica se Ollama está acessível."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def main():
    """Função principal."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  TUTORIAL AULA 17 - PARTE 3: EMBEDDINGS     ║")
    print("║  Chunking + Vector Store                    ║")
    print("╚══════════════════════════════════════════════╝")

    # 0. Verificar Ollama
    if not verificar_ollama():
        print("\n  ⚠️  Ollama não está acessível!")
        print("  Certifique-se de que 'ollama serve' está rodando.")
        return

    # 1. Carregar documentos
    separador("1. CARREGANDO DOCUMENTOS")
    if not PASTA_DOCS.exists():
        print(f"\n  📁 Pasta {PASTA_DOCS}/ não encontrada.")
        print("  Execute primeiro a Parte 1 (Setup) ou crie a pasta manualmente.")
        return

    documentos = carregar_documentos()

    if not documentos:
        print("\n  ⚠️  Nenhum documento carregado.")
        print("  Coloque arquivos .txt, .md, .pdf ou .docx na pasta ./docs/")
        return

    # 2. Chunking
    separador("2. DIVIDINDO EM CHUNKS")
    chunks = chunking(documentos)

    if not chunks:
        print("\n  ⚠️  Nenhum chunk criado.")
        return

    # 3. Visualizar chunks
    separador("3. AMOSTRA DOS CHUNKS")
    visualizar_chunks(chunks)

    # 4. Criar vector store
    separador("4. CRIANDO VECTOR STORE (ChromaDB)")
    vector_store = criar_vector_store(chunks)

    # 5. Teste rápido: verificar contagem
    separador("5. VERIFICAÇÃO")
    print(f"  ✅ Vector store pronto em: {PASTA_DB.resolve()}")
    print(f"  📊 Total de chunks indexados: {len(chunks)}")
    print(f"  🧮 Modelo de embedding: {MODELO_EMBED}")
    print()
    print("  Próximo passo: python GO1741-Tutorial04Busca.py")
    print()
    print("  💡 Dica: Experimente alterar CHUNK_SIZE no início do arquivo")
    print("     para 200 ou 1000 e veja como muda o resultado!")


if __name__ == "__main__":
    main()
