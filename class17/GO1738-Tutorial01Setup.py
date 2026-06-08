#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1738-Tutorial01Setup
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PARTE 1: SETUP DO AMBIENTE
============================
Verifica se todos os componentes necessários estão instalados:
  1. Python 3.8+
  2. Ollama rodando e acessível
  3. Modelos LLM e de embedding baixados
  4. Bibliotecas Python (chromadb, langchain, etc.)
  5. Estrutura de pastas do projeto

Funciona em: Windows 10/11 | Linux
"""

import sys
import subprocess
import os
from pathlib import Path
import platform

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES (altere aqui se necessário)
# ═══════════════════════════════════════════════════════════════════

# Modelo LLM para o tutorial (troque para qualquer modelo Ollama instalado)
MODELO_LLM = "llama3.2:3b"

# Modelo de embedding
MODELO_EMBED = "nomic-embed-text"

# Pasta onde os documentos serão armazenados
PASTA_DOCS = Path("./docs")
PASTA_DB = Path("./chroma_db")
PASTA_LOGS = Path("./logs")

# ═══════════════════════════════════════════════════════════════════

def separador(titulo=""):
    """Imprime um separador visual."""
    print()
    print("=" * 60)
    if titulo:
        print(f"  {titulo}")
        print("=" * 60)


def verificar_python():
    """Verifica versão do Python (mínimo 3.8)."""
    print(f"  Python: {sys.version.split()[0]}", end=" ")
    if sys.version_info >= (3, 8):
        print("✅")
        return True
    else:
        print("❌ (mínimo 3.8)")
        return False


def verificar_ollama():
    """Verifica se Ollama está instalado e rodando."""
    try:
        # Testa se o comando ollama existe
        resultado = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if resultado.returncode == 0:
            versao = resultado.stdout.strip()
            print(f"  Ollama: {versao} ✅")

            # Testa se o servidor está respondendo
            try:
                import ollama
                ollama.list()
                print(f"  Ollama API: respondendo ✅")
                return True
            except Exception:
                print(f"  Ollama API: ⚠️  instalado mas servidor não está rodando")
                print(f"             Execute 'ollama serve' em outro terminal")
                return False
        else:
            print(f"  Ollama: ❌ não encontrado")
            return False
    except FileNotFoundError:
        print(f"  Ollama: ❌ não encontrado")
        print(f"           Instale em: https://ollama.com")
        return False
    except subprocess.TimeoutExpired:
        print(f"  Ollama: ⚠️  timeout ao verificar")
        return False


def verificar_modelo_ollama(nome_modelo, tipo="LLM"):
    """Verifica se um modelo específico está baixado no Ollama."""
    try:
        import ollama
        modelos = ollama.list()
        nomes = []
        for m in modelos.get("models", []):
            nome = m.get("name", "")
            # Ollama pode retornar "nome:tag" ou apenas "nome"
            nomes.append(nome.split(":")[0])

        modelo_base = nome_modelo.split(":")[0]
        if modelo_base in nomes or any(nome_modelo in n for n in nomes):
            print(f"  Modelo {tipo} ({nome_modelo}): ✅")
            return True
        else:
            print(f"  Modelo {tipo} ({nome_modelo}): ⚠️  não encontrado")
            print(f"           Execute: ollama pull {nome_modelo}")
            return False
    except Exception:
        print(f"  Modelo {tipo} ({nome_modelo}): ⚠️  não foi possível verificar")
        return False


def verificar_biblioteca(nome):
    """Verifica se uma biblioteca Python está instalada."""
    try:
        __import__(nome.replace("-", "_").split("[")[0])
        print(f"  {nome}: ✅")
        return True
    except ImportError:
        print(f"  {nome}: ❌")
        print(f"           Execute: pip install {nome}")
        return False


def criar_pastas():
    """Cria a estrutura de pastas do projeto."""
    todas_ok = True
    for pasta in [PASTA_DOCS, PASTA_DB, PASTA_LOGS]:
        try:
            pasta.mkdir(parents=True, exist_ok=True)
            print(f"  📁 {pasta}/: criada ✅")
        except Exception as e:
            print(f"  📁 {pasta}/: erro ❌ ({e})")
            todas_ok = False
    return todas_ok


def criar_documento_exemplo():
    """Cria um documento de exemplo na pasta docs se estiver vazia."""
    arquivos = list(PASTA_DOCS.glob("*"))
    if len(arquivos) > 0:
        print(f"  📄 Documentos existentes: {len(arquivos)} arquivos (ok)")
        return True

    # Cria um arquivo de exemplo
    exemplo = PASTA_DOCS / "exemplo_introducao_ia.txt"
    conteudo = """INTRODUÇÃO À INTELIGÊNCIA ARTIFICIAL

A Inteligência Artificial (IA) é um ramo da ciência da computação que
busca criar sistemas capazes de realizar tarefas que normalmente exigem
inteligência humana. Estas tarefas incluem aprendizado, raciocínio,
percepção visual, reconhecimento de fala e tomada de decisões.

HISTÓRIA DA IA

A IA como campo de estudo formal começou em 1956, durante a Conferência
de Dartmouth, onde John McCarthy, Marvin Minsky, Nathaniel Rochester e
Claude Shannon propuseram o termo "Inteligência Artificial".

Desde então, a IA passou por diversos períodos de otimismo (chamados
de "verões da IA") e decepção (chamados de "invernos da IA").

PRINCIPAIS ÁREAS DA IA

1. Machine Learning (Aprendizado de Máquina)
   - Aprendizado Supervisionado: classificação e regressão
   - Aprendizado Não-Supervisionado: clustering e associação
   - Aprendizado por Reforço: aprendizado por tentativa e erro

2. Redes Neurais Artificiais
   - MLP (Multilayer Perceptron)
   - CNN (Redes Convolucionais) para imagens
   - RNN/LSTM para sequências temporais
   - Transformers para NLP

3. Processamento de Linguagem Natural (NLP)
   - Tokenização e análise sintática
   - Análise de sentimentos
   - Tradução automática
   - Geração de texto

OLLAMA E MODELOS LOCAIS

Com ferramentas como o Ollama, é possível rodar modelos de linguagem
diretamente no seu computador, sem depender de serviços em nuvem.
Isso garante privacidade dos dados, funcionamento offline e sem custos
de API.

Modelos como Llama 3, Mistral e Phi-3 podem ser executados localmente
com hardware modesto (8-16 GB de RAM).

RAG (RETRIEVAL-AUGMENTED GENERATION)

RAG é uma técnica que combina recuperação de informação com geração
de texto. O processo funciona assim:

1. Documentos são divididos em pedaços (chunks)
2. Cada chunk é convertido em um vetor (embedding)
3. Quando uma pergunta é feita, ela também é convertida em vetor
4. O sistema busca os chunks mais similares à pergunta
5. O LLM gera uma resposta baseada nos chunks encontrados
"""
    try:
        with open(exemplo, "w", encoding="utf-8") as f:
            f.write(conteudo)
        print(f"  📄 {exemplo.name}: criado ✅ (pasta docs estava vazia)")
        return True
    except Exception as e:
        print(f"  📄 {exemplo.name}: erro ❌ ({e})")
        return False


def gerar_relatorio(resultados):
    """Gera um relatório visual do ambiente."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║         RELATÓRIO DO AMBIENTE               ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  SO:              {platform.system()} {platform.release()}".ljust(52) + "║")
    print(f"║  Python:          {resultados.get('python', '?')}".ljust(52) + "║")
    print(f"║  Ollama:          {resultados.get('ollama', '?')}".ljust(52) + "║")
    print(f"║  Modelo LLM:      {resultados.get('modelo_llm', '?')}".ljust(52) + "║")
    print(f"║  Modelo Embed:    {resultados.get('modelo_embed', '?')}".ljust(52) + "║")
    print(f"║  ChromaDB:        {resultados.get('chromadb', '?')}".ljust(52) + "║")
    print(f"║  LangChain:       {resultados.get('langchain', '?')}".ljust(52) + "║")
    print(f"║  Streamlit:       {resultados.get('streamlit', '?')}".ljust(52) + "║")
    print(f"║  Pastas:          {resultados.get('pastas', '?')}".ljust(52) + "║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Verificação geral
    todos_ok = all(
        v and "❌" not in str(v) and "⚠️" not in str(v)
        for v in resultados.values()
    )

    if todos_ok:
        print("🎉 AMBIENTE PRONTO! Você pode prosseguir para a Parte 2.")
        print()
        print("Próximo passo: python GO1739-Tutorial02Ingestao.py")
    else:
        print("⚠️  Alguns componentes precisam de atenção.")
        print("   Verifique os itens marcados com ❌ ou ⚠️  acima.")
        print()
        print("Comandos úteis:")
        print("  pip install ollama chromadb langchain langchain-community pypdf python-docx streamlit")
        print("  ollama pull llama3.2:3b")
        print("  ollama pull nomic-embed-text")

    return todos_ok


def main():
    """Função principal do setup."""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  TUTORIAL AULA 17 - PARTE 1: SETUP          ║")
    print("║  Verificação do Ambiente                    ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Sistema: {platform.system()} {platform.release()}")
    print(f"  Pasta de documentos: {PASTA_DOCS.resolve()}")
    print()

    resultados = {}

    # 1. Python
    separador("1. VERIFICANDO PYTHON")
    resultados["python"] = "✅" if verificar_python() else "❌"

    # 2. Ollama
    separador("2. VERIFICANDO OLLAMA")
    resultados["ollama"] = "✅" if verificar_ollama() else "❌"

    # 3. Modelos
    separador("3. VERIFICANDO MODELOS")
    modelos_llm_ok = verificar_modelo_ollama(MODELO_LLM, "LLM")
    modelos_embed_ok = verificar_modelo_ollama(MODELO_EMBED, "Embedding")
    resultados["modelo_llm"] = "✅" if modelos_llm_ok else "⚠️"
    resultados["modelo_embed"] = "✅" if modelos_embed_ok else "⚠️"

    # 4. Bibliotecas Python
    separador("4. VERIFICANDO BIBLIOTECAS")
    resultados["chromadb"] = "✅" if verificar_biblioteca("chromadb") else "❌"
    resultados["langchain"] = "✅" if verificar_biblioteca("langchain") else "❌"
    resultados["streamlit"] = "✅" if verificar_biblioteca("streamlit") else "❌"

    # Também verifica dependências de carregamento de documentos
    verificar_biblioteca("pypdf")
    verificar_biblioteca("docx2txt")

    # 5. Pastas
    separador("5. CRIANDO PASTAS DO PROJETO")
    pastas_ok = criar_pastas()
    resultados["pastas"] = "✅" if pastas_ok else "❌"

    # 6. Documento de exemplo
    separador("6. VERIFICANDO DOCUMENTOS")
    criar_documento_exemplo()

    # 7. Relatório final
    separador("RELATÓRIO FINAL")
    gerar_relatorio(resultados)


if __name__ == "__main__":
    main()
