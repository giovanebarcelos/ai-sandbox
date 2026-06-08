#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1743-Tutorial06Interface
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 6: INTERFACE STREAMLIT
==============================
Interface web interativa para o chatbot RAG com:
  - Upload de documentos (PDF, DOCX, TXT)
  - Chat com histórico
  - Exibição de fontes consultadas
  - Botão para limpar conversa

Funciona em: Windows 10/11 | Linux

Para executar:
  streamlit run GO1743-Tutorial06Interface.py
"""

import os
import sys
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Streamlit deve ser importado primeiro
import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════

MODELO_LLM = "llama3.2:3b"
MODELO_EMBED = "nomic-embed-text"
PASTA_DB = Path("./chroma_db")

# ═══════════════════════════════════════════════════════════════════


def verificar_ollama():
    """Verifica se Ollama está acessível."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


@st.cache_resource
def inicializar_vector_store():
    """Inicializa (ou cria) o vector store em cache."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)

    if PASTA_DB.exists():
        return Chroma(
            persist_directory=str(PASTA_DB),
            embedding_function=embeddings,
        )
    return None


def processar_documento_upload(arquivo_upload):
    """Processa um arquivo enviado pelo usuário."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Salva em temp e carrega
    with tempfile.NamedTemporaryFile(delete=False, suffix=arquivo_upload.name) as tmp:
        tmp.write(arquivo_upload.getvalue())
        caminho_tmp = tmp.name

    try:
        ext = Path(arquivo_upload.name).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(caminho_tmp)
            documentos = loader.load()
        elif ext == ".txt":
            try:
                loader = TextLoader(caminho_tmp, encoding="utf-8")
                documentos = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(caminho_tmp, encoding="latin-1")
                documentos = loader.load()
        elif ext == ".md":
            loader = TextLoader(caminho_tmp, encoding="utf-8")
            documentos = loader.load()
        else:
            return None, f"Formato {ext} não suportado. Use PDF, TXT ou MD."

        # Adiciona metadados do nome original
        for doc in documentos:
            doc.metadata["arquivo"] = arquivo_upload.name

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documentos)

        # Indexar no ChromaDB
        vector_store = inicializar_vector_store()
        if vector_store is None:
            from langchain_community.embeddings import OllamaEmbeddings
            from langchain_community.vectorstores import Chroma
            embeddings = OllamaEmbeddings(model=MODELO_EMBED)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(PASTA_DB),
            )
        else:
            vector_store.add_documents(chunks)

        # Persistir
        try:
            vector_store.persist()
        except Exception:
            pass

        os.unlink(caminho_tmp)

        return len(chunks), f"✅ {len(chunks)} chunk(s) indexados de '{arquivo_upload.name}'"

    except Exception as e:
        os.unlink(caminho_tmp)
        return None, f"❌ Erro ao processar: {e}"


def buscar_resposta(pergunta, vector_store):
    """Busca contexto e gera resposta."""
    import ollama

    # 1. Buscar contexto
    docs = vector_store.similarity_search_with_relevance_scores(
        pergunta, k=3
    )

    if not docs:
        return "Não encontrei informações sobre isso nos documentos.", []

    # 2. Construir prompt
    contexto = ""
    for i, (doc, score) in enumerate(docs, 1):
        fonte = doc.metadata.get("arquivo", "desconhecido")
        contexto += f"\nDocumento {i} [Fonte: {fonte} | Relevância: {score:.1%}]\n"
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

    # 3. Gerar resposta
    response = ollama.generate(
        model=MODELO_LLM,
        prompt=prompt,
        stream=False,
        options={"temperature": 0.3},
    )

    resposta = response["response"].strip()

    # 4. Extrair fontes
    fontes = list(set(
        (doc.metadata.get("arquivo", "?"), f"{score:.1%}")
        for doc, score in docs
    ))

    return resposta, fontes


def pagina_principal():
    """Renderiza a página principal do Streamlit."""
    st.set_page_config(
        page_title="📚 Tutor IA - Chat com Documentos",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.title("📚 Tutor IA")
        st.markdown("---")

        # Status do Ollama
        ollama_ok = verificar_ollama()
        if ollama_ok:
            st.success(f"🟢 Ollama conectado ({MODELO_LLM})")
        else:
            st.error("🔴 Ollama não disponível")
            st.info("Execute 'ollama serve' em outro terminal")

        st.markdown("---")
        st.subheader("📁 Upload de Documentos")

        uploaded_files = st.file_uploader(
            "Escolha arquivos (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for arquivo in uploaded_files:
                with st.spinner(f"Processando {arquivo.name}..."):
                    num_chunks, msg = processar_documento_upload(arquivo)
                    if num_chunks:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown("---")
        st.caption(f"📂 Banco: `{PASTA_DB.resolve()}`")

        if st.button("🧹 Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()

        with st.expander("⚙️ Configurações"):
            novo_modelo = st.text_input("Modelo LLM", value=MODELO_LLM)
            globals()["MODELO_LLM"] = novo_modelo

    # Área principal: chat
    st.title("💬 Chat com seus Documentos")
    st.markdown("Faça perguntas sobre os documentos que você enviou.")

    # Inicializar histórico
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # Exibir histórico
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "fontes" in msg and msg["fontes"]:
                with st.expander("📚 Fontes consultadas"):
                    for nome, score in msg["fontes"]:
                        st.write(f"- {nome} (relevância: {score})")

    # Input do usuário
    pergunta = st.chat_input("Digite sua pergunta sobre os documentos...")

    if pergunta:
        # Adiciona pergunta ao histórico
        st.session_state.mensagens.append({"role": "user", "content": pergunta})

        with st.chat_message("user"):
            st.markdown(pergunta)

        # Gera resposta
        with st.chat_message("assistant"):
            vector_store = inicializar_vector_store()

            if vector_store is None:
                resposta = "⚠️ Nenhum documento indexado. Faça upload de documentos primeiro!"
                fontes = []
                st.warning(resposta)
            elif not ollama_ok:
                resposta = "⚠️ Ollama não está rodando. Execute 'ollama serve' em outro terminal."
                fontes = []
                st.error(resposta)
            else:
                with st.spinner("🔍 Buscando nos documentos..."):
                    resposta, fontes = buscar_resposta(pergunta, vector_store)
                st.markdown(resposta)

                if fontes:
                    with st.expander("📚 Fontes consultadas"):
                        for nome, score in fontes:
                            st.write(f"- 📄 {nome} (relevância: {score})")

            # Adiciona resposta ao histórico
            st.session_state.mensagens.append({
                "role": "assistant",
                "content": resposta,
                "fontes": fontes,
            })


if __name__ == "__main__":
    pagina_principal()
