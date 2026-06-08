#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1739-Tutorial02Ingestao
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 2: INGESTÃO DE DOCUMENTOS
=================================
Carrega documentos de diferentes formatos (PDF, TXT, DOCX) usando
os carregadores do LangChain. Mostra estatísticas dos documentos.

Funciona em: Windows 10/11 | Linux
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════

PASTA_DOCS = Path("./docs")

# ═══════════════════════════════════════════════════════════════════


def separador(titulo=""):
    """Imprime um separador visual."""
    print()
    print("=" * 60)
    if titulo:
        print(f"  {titulo}")
        print("=" * 60)


def listar_arquivos(pasta):
    """Lista todos os arquivos compatíveis na pasta docs."""
    extensoes = {
        ".pdf": "📄 PDF",
        ".txt": "📄 TXT",
        ".md":  "📄 Markdown",
        ".docx": "📄 DOCX",
        ".csv": "📄 CSV",
        ".json": "📄 JSON",
    }

    arquivos = []
    for ext, icone in extensoes.items():
        for f in sorted(pasta.glob(f"*{ext}")):
            arquivos.append((f, icone))

    if not arquivos:
        print("  ⚠️  Nenhum arquivo compatível encontrado em", pasta)
        print("  Coloque seus PDFs, DOCs ou TXTs na pasta ./docs/")
        return []

    print(f"  📂 Documentos encontrados: {len(arquivos)}")
    print()
    for f, icone in arquivos:
        tamanho = f.stat().st_size
        if tamanho < 1024:
            tam_str = f"{tamanho} B"
        elif tamanho < 1024 * 1024:
            tam_str = f"{tamanho / 1024:.0f} KB"
        else:
            tam_str = f"{tamanho / (1024*1024):.1f} MB"
        print(f"     {icone} {f.name:<40s} {tam_str:>8s}")

    return arquivos


def carregar_pdf(caminho):
    """Carrega um arquivo PDF usando PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(caminho))
        documentos = loader.load()
        print(f"     ✅ PDF carregado: {len(documentos)} página(s)")
        return documentos
    except Exception as e:
        print(f"     ❌ Erro ao carregar PDF: {e}")
        return []


def carregar_txt(caminho):
    """Carrega um arquivo TXT usando TextLoader."""
    try:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(caminho), encoding="utf-8")
        # Tenta utf-8 primeiro, se falhar tenta latin-1
        try:
            documentos = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(str(caminho), encoding="latin-1")
            documentos = loader.load()
        print(f"     ✅ TXT carregado: {len(documentos)} documento(s)")
        return documentos
    except Exception as e:
        print(f"     ❌ Erro ao carregar TXT: {e}")
        return []


def carregar_docx(caminho):
    """Carrega um arquivo DOCX usando Docx2txtLoader."""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(caminho))
        documentos = loader.load()
        print(f"     ✅ DOCX carregado: {len(documentos)} documento(s)")
        return documentos
    except Exception as e:
        print(f"     ❌ Erro ao carregar DOCX: {e}")
        return []


def carregar_markdown(caminho):
    """Carrega um arquivo Markdown usando UnstructuredMarkdownLoader."""
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(str(caminho))
        documentos = loader.load()
        print(f"     ✅ Markdown carregado: {len(documentos)} documento(s)")
        return documentos
    except Exception as e:
        print(f"     ❌ Erro ao carregar Markdown: {e}")
        return []


def inspecionar_documento(doc):
    """Mostra informações básicas de um documento carregado."""
    print(f"     Conteúdo: {len(doc.page_content)} caracteres")
    if doc.metadata:
        # Mostra apenas metadados relevantes
        meta_visivel = {k: v for k, v in doc.metadata.items()
                        if k not in ("source",)}
        if meta_visivel:
            print(f"     Metadados: {meta_visivel}")


def estatisticas_documentos(todos_docs):
    """Gera estatísticas sobre os documentos carregados."""
    if not todos_docs:
        print("\n  ⚠️  Nenhum documento carregado.")
        return

    total_caracteres = sum(len(d.page_content) for d in todos_docs)
    # Estimativa de tokens (~4 caracteres por token em português)
    total_tokens_estimado = total_caracteres // 4

    print()
    print("  📊 ESTATÍSTICAS DOS DOCUMENTOS")
    print(f"  {'─' * 50}")
    print(f"  Total de documentos carregados:  {len(todos_docs)}")
    print(f"  Total de caracteres:             {total_caracteres:,}")
    print(f"  Total estimado de tokens:        {total_tokens_estimado:,}")

    # Maior e menor documento
    if todos_docs:
        maior = max(todos_docs, key=lambda d: len(d.page_content))
        menor = min(todos_docs, key=lambda d: len(d.page_content))
        print(f"  Maior documento:                  {len(maior.page_content):,} chars")
        print(f"  Menor documento:                  {len(menor.page_content):,} chars")


def main():
    """Função principal da ingestão."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  TUTORIAL AULA 17 - PARTE 2: INGESTÃO       ║")
    print("║  Carregamento de Documentos                 ║")
    print("╚══════════════════════════════════════════════╝")

    # Verifica se a pasta docs existe
    if not PASTA_DOCS.exists():
        print(f"\n  📁 Criando pasta {PASTA_DOCS}/ ...")
        PASTA_DOCS.mkdir(parents=True, exist_ok=True)
        print(f"  Pasta criada. Coloque seus documentos nela e execute novamente.")
        return

    # 1. Listar arquivos disponíveis
    separador("1. ARQUIVOS DISPONÍVEIS")
    arquivos = listar_arquivos(PASTA_DOCS)

    if not arquivos:
        return

    # 2. Carregar cada arquivo
    separador("2. CARREGANDO DOCUMENTOS")

    todos_documentos = []
    carregadores = {
        ".pdf": carregar_pdf,
        ".txt": carregar_txt,
        ".md": carregar_markdown,
        ".docx": carregar_docx,
    }

    for caminho, icone in arquivos:
        ext = caminho.suffix.lower()
        carregador = carregadores.get(ext)

        if carregador:
            print(f"\n  {icone} {caminho.name}")
            docs = carregador(caminho)
            todos_documentos.extend(docs)
            for doc in docs[:1]:  # inspeciona apenas o primeiro trecho
                inspecionar_documento(doc)
        else:
            print(f"\n  {icone} {caminho.name}")
            print(f"     ⚠️  Formato não suportado: {ext}")

    # 3. Estatísticas
    separador("3. ESTATÍSTICAS")
    estatisticas_documentos(todos_documentos)

    # 4. Resumo
    print()
    print(f"  ✅ Ingestão concluída! {len(todos_documentos)} documentos carregados.")
    print()
    print("  Próximo passo: python GO1740-Tutorial03Embeddings.py")


if __name__ == "__main__":
    main()
