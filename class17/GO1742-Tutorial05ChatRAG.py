#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1742-Tutorial05ChatRAG
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 5: CHAT RAG COM CONTEXTO
=================================
Combina busca + LLM em um pipeline completo de RAG:
  1. Recebe pergunta do usuário
  2. Busca chunks relevantes no vector store
  3. Constrói prompt aumentado (contexto + pergunta)
  4. Envia para o Ollama LLM
  5. Exibe resposta com fontes

Funciona em: Windows 10/11 | Linux
"""

import os
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════

MODELO_LLM = "llama3.2:3b"      # Troque para qualquer modelo Ollama
MODELO_EMBED = "nomic-embed-text"
PASTA_DB = Path("./chroma_db")
ARQUIVO_HISTORICO = Path("./logs/historico.json")

# ═══════════════════════════════════════════════════════════════════


def separador(titulo=""):
    print()
    print("=" * 60)
    if titulo:
        print(f"  {titulo}")
        print("=" * 60)


def carregar_vector_store():
    """Carrega o vector store ChromaDB."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    if not PASTA_DB.exists():
        print(f"\n  ⚠️  Banco não encontrado em {PASTA_DB}")
        print("  Execute a Parte 3 primeiro (GO1740).")
        return None

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)
    vector_store = Chroma(
        persist_directory=str(PASTA_DB),
        embedding_function=embeddings,
    )
    return vector_store


def buscar_contexto(vector_store, pergunta, k=3):
    """Busca chunks relevantes para a pergunta."""
    docs = vector_store.similarity_search_with_relevance_scores(
        pergunta, k=k
    )
    return docs


def construir_prompt(pergunta, contexto_docs, historico=None):
    """Constrói o prompt aumentado com contexto."""
    # Formata o contexto
    contexto_texto = ""
    for i, (doc, score) in enumerate(contexto_docs, 1):
        arquivo = doc.metadata.get("arquivo", "desconhecido")
        fonte = f"[Fonte: {arquivo} | Relevância: {score:.1%}]"
        contexto_texto += f"\nDocumento {i} {fonte}\n"
        contexto_texto += doc.page_content.strip()
        contexto_texto += "\n"

    # Histórico (últimas 3 trocas)
    historico_texto = ""
    if historico:
        for turno in historico[-6:]:  # últimas 6 mensagens = 3 trocas
            papel = turno.get("role", "")
            conteudo = turno.get("content", "")
            if papel == "user":
                historico_texto += f"\nUsuário: {conteudo}"
            elif papel == "assistant":
                historico_texto += f"\nAssistente: {conteudo}"

    # Prompt completo
    system_prompt = """Você é um assistente especializado em responder perguntas com base EXCLUSIVAMENTE nos documentos fornecidos no CONTEXTO abaixo.

REGRAS:
1. Responda APENAS com base nas informações do CONTEXTO
2. Se o contexto não tiver informação suficiente, diga "Não encontrei essa informação nos documentos disponíveis"
3. Sempre cite a FONTE do documento que usou (nome do arquivo)
4. Seja claro, didático e responda em português
5. Use formatação markdown quando apropriado (negrito, listas)"""

    if historico_texto:
        prompt = f"""{system_prompt}

CONTEXTO:
{contexto_texto}

HISTÓRICO DA CONVERSA:
{historico_texto}

PERGUNTA DO USUÁRIO: {pergunta}

RESPOSTA:"""
    else:
        prompt = f"""{system_prompt}

CONTEXTO:
{contexto_texto}

PERGUNTA DO USUÁRIO: {pergunta}

RESPOSTA:"""

    return prompt


def gerar_resposta_ollama(prompt):
    """Gera resposta usando Ollama."""
    import ollama

    response = ollama.generate(
        model=MODELO_LLM,
        prompt=prompt,
        stream=False,
        options={
            "temperature": 0.3,      # Baixo para respostas mais factuais
            "top_p": 0.9,
            "top_k": 40,
        }
    )
    return response["response"].strip()


def gerar_resposta_fallback(pergunta, contexto_docs):
    """Fallback caso Ollama não esteja disponível."""
    resposta = "⚠️  Modo DEMO (Ollama não disponível)\n\n"
    resposta += f"Pergunta: {pergunta}\n\n"
    resposta += "Documentos relevantes encontrados:\n\n"

    for i, (doc, score) in enumerate(contexto_docs, 1):
        arquivo = doc.metadata.get("arquivo", "desconhecido")
        texto = doc.page_content[:300].replace("\n", " ").strip()
        resposta += f"{i}. [{arquivo}] (relevância: {score:.1%})\n"
        resposta += f"   \"{texto}...\"\n\n"

    resposta += "\n💡 Instale o Ollama e execute 'ollama serve' para respostas reais!"
    return resposta


def verificar_ollama():
    """Verifica se Ollama está rodando."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def exibir_resposta(pergunta, resposta, fontes):
    """Exibe a resposta com formatação."""
    print(f"\n  🧑 Você: {pergunta}")
    print(f"\n  🤖 Assistente:")
    print(f"  {'─' * 50}")
    print(f"  {resposta}")
    print(f"  {'─' * 50}")

    if fontes:
        print(f"\n  📚 Fontes consultadas:")
        for nome, score in fontes:
            print(f"     • {nome} (relevância: {score:.1%})")


def salvar_historico(mensagens):
    """Salva o histórico em JSON."""
    ARQUIVO_HISTORICO.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_HISTORICO, "w", encoding="utf-8") as f:
        json.dump(mensagens, f, ensure_ascii=False, indent=2)


def comparar_com_sem_rag(vector_store):
    """Compara resposta COM e SEM contexto RAG."""
    import ollama

    pergunta = "O que é Inteligência Artificial?"

    print(f"\n  🔬 COMPARAÇÃO: COM RAG vs SEM RAG")
    print(f"  Pergunta: \"{pergunta}\"")
    print()

    # --- SEM RAG ---
    print("  ─── SEM RAG (LLM puro, sem contexto) ───")
    prompt_sem = f"Responda em português: {pergunta}"
    try:
        resp_sem = ollama.generate(
            model=MODELO_LLM,
            prompt=prompt_sem,
            stream=False,
        )["response"].strip()
        print(f"  {resp_sem[:400]}")
    except Exception as e:
        print(f"  ⚠️  Erro: {e}")
    print()

    # --- COM RAG ---
    print("  ─── COM RAG (LLM + contexto dos documentos) ───")
    docs = buscar_contexto(vector_store, pergunta, k=3)
    prompt_com = construir_prompt(pergunta, docs)
    try:
        resp_com = ollama.generate(
            model=MODELO_LLM,
            prompt=prompt_com,
            stream=False,
        )["response"].strip()
        print(f"  {resp_com[:400]}")
    except Exception as e:
        print(f"  ⚠️  Erro: {e}")

    print()
    print("  ✅ Compare as duas respostas acima.")
    print("     A resposta COM RAG deve ser mais precisa e citar fontes.")


def modo_interativo(vector_store, usar_ollama):
    """Modo de chat interativo."""
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║        CHAT RAG - SESSÃO INTERATIVA         ║")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║  Digite sua pergunta (ou 'sair' p/ encerrar)║")
    print("  ╚══════════════════════════════════════════════╝")

    historico = []

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

        # Busca contexto
        docs = buscar_contexto(vector_store, pergunta, k=3)

        if not docs:
            print("\n  ⚠️  Nenhum documento relevante encontrado.")
            continue

        # Gera resposta
        print("  ⏳ Pensando...", end="", flush=True)

        if usar_ollama:
            prompt = construir_prompt(pergunta, docs, historico)
            resposta = gerar_resposta_ollama(prompt)
        else:
            resposta = gerar_resposta_fallback(pergunta, docs)

        print("\r" + " " * 20 + "\r", end="")

        # Extrai fontes
        fontes = list(set(
            (doc.metadata.get("arquivo", "desconhecido"), score)
            for doc, score in docs
        ))

        # Exibe
        exibir_resposta(pergunta, resposta, fontes)

        # Atualiza histórico
        historico.append({"role": "user", "content": pergunta})
        historico.append({"role": "assistant", "content": resposta})

    # Salva histórico
    if historico:
        salvar_historico(historico)
        print(f"\n  💾 Histórico salvo em: {ARQUIVO_HISTORICO}")

    print("\n  👋 Até logo!")


def main():
    """Função principal."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  TUTORIAL AULA 17 - PARTE 5: CHAT RAG      ║")
    print("║  Geração Aumentada por Recuperação          ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Modelo LLM: {MODELO_LLM}")
    print(f"  Modelo Embed: {MODELO_EMBED}")

    # 0. Verificar dependências
    usar_ollama = verificar_ollama()
    if usar_ollama:
        print("  ✅ Ollama disponível — respostas reais do LLM")
    else:
        print("  ⚠️  Ollama não disponível — modo fallback (demo)")

    # 1. Carregar vector store
    separador("1. CARREGANDO BASE DE CONHECIMENTO")
    vector_store = carregar_vector_store()
    if vector_store is None:
        return

    # 2. Demonstração: busca simples
    separador("2. DEMONSTRAÇÃO: BUSCA DE CONTEXTO")
    pergunta_demo = "O que é uma rede neural?"
    docs = buscar_contexto(vector_store, pergunta_demo, k=3)
    print(f"\n  Pergunta: \"{pergunta_demo}\"")
    print(f"  📦 {len(docs)} documento(s) relevante(s) encontrado(s)")
    for i, (doc, score) in enumerate(docs, 1):
        arquivo = doc.metadata.get("arquivo", "?")
        print(f"     {i}. {arquivo} (score: {score:.3f})")

    # 3. Comparação RAG vs SEM RAG
    if usar_ollama:
        separador("3. COMPARAÇÃO: COM RAG vs SEM RAG")
        comparar_com_sem_rag(vector_store)

    # 4. Modo interativo
    separador("4. CHAT INTERATIVO")
    modo_interativo(vector_store, usar_ollama)

    print()
    print("  Próximo passo: python GO1743-Tutorial06Interface.py")


if __name__ == "__main__":
    main()
