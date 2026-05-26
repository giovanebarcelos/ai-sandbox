#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1706-19ProjetoChatbotRagComOllama
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

PROJETO: Chatbot RAG com Base de Conhecimento
==============================================
Implementa o pipeline completo de RAG:
  1. Ingestão: documentos → chunks → embeddings → vector store
  2. Retrieval: pergunta → embedding → busca por similaridade
  3. Augmentation: contexto + pergunta → prompt estruturado
  4. Generation: LLM gera resposta baseada no contexto

MODO DE OPERAÇÃO:
  - COM Ollama instalado: usa llama3.2 como LLM e nomic-embed-text como embedder
  - SEM Ollama (fallback): usa TF-IDF para embeddings + template de resposta local
  - Visualizações sempre disponíveis independente do modo

Instalação recomendada: pip install chromadb ollama scikit-learn matplotlib
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 1. DETECTAR MODO DE OPERAÇÃO (Ollama disponível ou não)
# ═══════════════════════════════════════════════════════════════════

def verificar_ollama():
    """
    Verifica se Ollama está disponível e acessível.
    Retorna True se conectado, False caso contrário.
    """
    try:
        import ollama
        # Teste de ping: listar modelos disponíveis
        modelos = ollama.list()
        nomes = [m['name'] for m in modelos.get('models', [])]
        print(f"✅ Ollama disponível! Modelos: {nomes[:3]}")
        return True
    except Exception as e:
        print(f"⚠️  Ollama não disponível: {e}")
        print("   Modo fallback ativado (TF-IDF + resposta template)")
        return False


OLLAMA_DISPONIVEL = verificar_ollama()

print("\n" + "=" * 70)
print("CHATBOT RAG - BASE DE CONHECIMENTO SOBRE FAPA/IA")
print(f"Modo: {'OLLAMA (LLM Real)' if OLLAMA_DISPONIVEL else 'FALLBACK (TF-IDF + Template)'}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 2. BASE DE CONHECIMENTO (documentos internos da disciplina)
# ═══════════════════════════════════════════════════════════════════

# Estes seriam os documentos reais carregados de PDFs/arquivos.
# Para a demo, usamos uma base de conhecimento fictícia sobre o curso.
BASE_CONHECIMENTO = [
    # --- Sobre o Curso ---
    {
        "texto": "A disciplina de Inteligência Artificial da FAPA cobre 20 aulas "
                 "incluindo Machine Learning, Redes Neurais, NLP, LLMs e Reinforcement Learning.",
        "fonte": "programa_curso.pdf",
        "tema": "curso"
    },
    {
        "texto": "As avaliações consistem em provas teóricas (40%) e projetos práticos (60%). "
                 "O projeto final deve demonstrar aplicação real de algum conceito de IA.",
        "fonte": "programa_curso.pdf",
        "tema": "avaliacao"
    },
    {
        "texto": "A frequência mínima exigida é de 75% das aulas. Faltas devem ser justificadas "
                 "em até 72 horas após a aula perdida.",
        "fonte": "regimento_fapa.pdf",
        "tema": "frequencia"
    },

    # --- Machine Learning ---
    {
        "texto": "Algoritmos de Machine Learning incluem Regressão Linear, Árvores de Decisão, "
                 "Random Forest, SVM e KNN. A escolha depende do problema (classificação vs regressão).",
        "fonte": "aula05_classificacao.pdf",
        "tema": "machine_learning"
    },
    {
        "texto": "Validação cruzada (k-fold) é usada para estimar a performance do modelo "
                 "de forma robusta. O valor típico de k é 5 ou 10 folds.",
        "fonte": "aula06_validacao.pdf",
        "tema": "machine_learning"
    },
    {
        "texto": "Métricas de avaliação para classificação: Acurácia, Precisão, Recall e F1-Score. "
                 "Para dados desbalanceados, F1-Score é mais adequado que Acurácia.",
        "fonte": "aula06_validacao.pdf",
        "tema": "machine_learning"
    },

    # --- Redes Neurais ---
    {
        "texto": "Redes neurais artificiais são compostas por camadas de neurônios artificiais. "
                 "O algoritmo backpropagation ajusta os pesos usando gradiente descendente.",
        "fonte": "aula09_redes_neurais.pdf",
        "tema": "deep_learning"
    },
    {
        "texto": "CNNs (Convolutional Neural Networks) são usadas para visão computacional. "
                 "A convolução detecta padrões espaciais como bordas e texturas nas imagens.",
        "fonte": "aula12_cnn.pdf",
        "tema": "deep_learning"
    },
    {
        "texto": "RNNs e LSTMs processam sequências de dados (texto, áudio, séries temporais). "
                 "O LSTM resolve o problema do gradiente desvanecente com células de memória.",
        "fonte": "aula14_rnn_lstm.pdf",
        "tema": "deep_learning"
    },

    # --- LLMs e RAG ---
    {
        "texto": "Transformers são a arquitetura base dos LLMs modernos. "
                 "O mecanismo de atenção permite ao modelo focar em partes relevantes do contexto.",
        "fonte": "aula16_transformers.pdf",
        "tema": "llm"
    },
    {
        "texto": "RAG (Retrieval-Augmented Generation) reduz alucinações do LLM ao fornecer "
                 "contexto relevante recuperado de documentos reais.",
        "fonte": "aula17_ollama_rag.pdf",
        "tema": "rag"
    },
    {
        "texto": "Ollama permite executar LLMs localmente sem internet. Modelos disponíveis: "
                 "llama3.2 (3B), mistral (7B), phi-3 (3.8B). Instalar via ollama.com.",
        "fonte": "aula17_ollama_rag.pdf",
        "tema": "ollama"
    },
    {
        "texto": "Embeddings convertem texto em vetores numéricos que capturam significado semântico. "
                 "Textos semanticamente similares ficam próximos no espaço vetorial.",
        "fonte": "aula17_ollama_rag.pdf",
        "tema": "embeddings"
    },

    # --- Ferramentas ---
    {
        "texto": "LangChain é um framework Python para construir aplicações com LLMs. "
                 "Oferece chains, agents, memory e integrações com vector stores.",
        "fonte": "ferramentas_ia.pdf",
        "tema": "ferramentas"
    },
    {
        "texto": "ChromaDB é um vector store open-source ideal para projetos RAG. "
                 "Instalação: pip install chromadb. Suporta embeddings locais e remotos.",
        "fonte": "ferramentas_ia.pdf",
        "tema": "ferramentas"
    },
]

print(f"\n📚 Base de conhecimento: {len(BASE_CONHECIMENTO)} documentos")
print(f"   Temas: {set(d['tema'] for d in BASE_CONHECIMENTO)}")

# ═══════════════════════════════════════════════════════════════════
# 3. SISTEMA DE EMBEDDINGS (Ollama ou TF-IDF fallback)
# ═══════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Abstração sobre o motor de embeddings.
    Usa Ollama (nomic-embed-text) se disponível, senão TF-IDF como fallback.
    """

    def __init__(self, usar_ollama=False):
        self.usar_ollama = usar_ollama
        self._vectorizer = None  # Usado apenas no modo fallback

        if usar_ollama:
            import ollama
            self._ollama = ollama
            print("🧠 Embeddings: nomic-embed-text (Ollama) - 768 dimensões")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Treinar vectorizer na base de conhecimento
            textos = [d['texto'] for d in BASE_CONHECIMENTO]
            self._vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1
            )
            self._vectorizer.fit(textos)
            print("🧠 Embeddings: TF-IDF (fallback) - 200 dimensões")

    def encode(self, texto):
        """Converte texto em vetor numérico."""
        if self.usar_ollama:
            resp = self._ollama.embeddings(
                model='nomic-embed-text',
                prompt=texto
            )
            return np.array(resp['embedding'])
        else:
            vec = self._vectorizer.transform([texto]).toarray()[0]
            return vec


# ═══════════════════════════════════════════════════════════════════
# 4. VECTOR STORE PARA RAG
# ═══════════════════════════════════════════════════════════════════

class RAGVectorStore:
    """
    Vector store otimizado para RAG.
    Armazena embeddings de todos os documentos da base de conhecimento.
    """

    def __init__(self, embedding_engine):
        self.engine = embedding_engine
        self.documentos = []
        self.embeddings = []
        self.metadados = []

    def indexar(self, documentos):
        """
        Fase de INDEXAÇÃO: converte todos os documentos em embeddings.
        Executada apenas uma vez (offline), o resultado é persistido.
        """
        print(f"\n📥 Indexando {len(documentos)} documentos...")
        for i, doc in enumerate(documentos):
            emb = self.engine.encode(doc['texto'])
            self.documentos.append(doc['texto'])
            self.embeddings.append(emb)
            self.metadados.append({'fonte': doc['fonte'], 'tema': doc['tema']})

            # Progresso
            if (i + 1) % 5 == 0 or (i + 1) == len(documentos):
                print(f"   [{i+1:2d}/{len(documentos)}] ✓ '{doc['fonte']}'")

        self.embeddings = np.array(self.embeddings)
        print(f"✅ Indexação concluída! Shape: {self.embeddings.shape}")

    def recuperar(self, query, k=3, min_score=0.05):
        """
        Fase de RETRIEVAL: busca os k chunks mais relevantes para a query.

        Parâmetros:
            query: pergunta do usuário
            k: número de documentos a recuperar
            min_score: score mínimo de relevância (filtra ruído)

        Retorna:
            Lista de (documento, score, metadados)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Converter query em vetor (usando o mesmo modelo de embeddings)
        query_emb = self.engine.encode(query).reshape(1, -1)

        # Calcular similaridade com todos os documentos
        scores = cosine_similarity(query_emb, self.embeddings)[0]

        # Ranking: top-k por score decrescente
        indices_ordenados = np.argsort(scores)[::-1][:k]

        resultados = []
        for idx in indices_ordenados:
            if scores[idx] >= min_score:
                resultados.append({
                    'texto': self.documentos[idx],
                    'score': float(scores[idx]),
                    'fonte': self.metadados[idx]['fonte'],
                    'tema': self.metadados[idx]['tema'],
                    'indice': int(idx)
                })

        return resultados


# ═══════════════════════════════════════════════════════════════════
# 5. GERADOR DE RESPOSTAS (LLM ou Template)
# ═══════════════════════════════════════════════════════════════════

class RespostaGerador:
    """
    Gera resposta final a partir do contexto recuperado.
    Com Ollama: usa LLM real. Sem Ollama: usa template inteligente.
    """

    def __init__(self, usar_ollama=False, modelo='llama3.2'):
        self.usar_ollama = usar_ollama
        self.modelo = modelo
        if usar_ollama:
            import ollama
            self._ollama = ollama
            print(f"🤖 Gerador: LLM Ollama ({modelo})")
        else:
            print("🤖 Gerador: Template inteligente (fallback)")

    def gerar(self, pergunta, contexto_docs):
        """
        Fase de GENERATION: gera resposta aumentada com contexto.

        O prompt segue o padrão RAG:
          - Contexto: trechos recuperados (ground truth)
          - Instrução: responder com base apenas no contexto
          - Pergunta: query original do usuário
        """
        # Montar contexto a partir dos documentos recuperados
        contexto = ""
        for i, doc in enumerate(contexto_docs, 1):
            contexto += f"\n[Fonte {i}: {doc['fonte']}]\n{doc['texto']}\n"

        if self.usar_ollama:
            # Prompt estruturado para RAG com LLM real
            prompt = f"""Você é um assistente especializado no curso de IA da FAPA.
Use APENAS as informações do CONTEXTO abaixo para responder.
Se a informação não estiver no contexto, diga "Não tenho essa informação nos documentos disponíveis."

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA (baseada no contexto acima):"""

            response = self._ollama.generate(
                model=self.modelo,
                prompt=prompt,
                options={'temperature': 0.3}  # Temperatura baixa = mais preciso
            )
            return response['response']

        else:
            # Modo fallback: extrair a resposta mais relevante do contexto
            if not contexto_docs:
                return "Não encontrei informações relevantes na base de conhecimento."

            # Usar o documento mais similar como base da resposta
            doc_principal = contexto_docs[0]
            fontes = list(set(d['fonte'] for d in contexto_docs))

            resposta = (
                f"Com base nos documentos disponíveis:\n\n"
                f"{doc_principal['texto']}\n\n"
            )

            # Adicionar complemento se houver mais documentos relevantes
            if len(contexto_docs) > 1:
                resposta += f"Informações complementares: {contexto_docs[1]['texto']}\n\n"

            resposta += f"[Fontes: {', '.join(fontes)}]"
            return resposta


# ═══════════════════════════════════════════════════════════════════
# 6. PIPELINE RAG COMPLETO
# ═══════════════════════════════════════════════════════════════════

class ChatbotRAG:
    """
    Pipeline RAG completo: Indexação → Retrieval → Augmentation → Generation

    Uso:
        chatbot = ChatbotRAG()
        chatbot.indexar_documentos(BASE_CONHECIMENTO)
        resposta = chatbot.responder("Qual a frequência mínima do curso?")
    """

    def __init__(self, usar_ollama=OLLAMA_DISPONIVEL):
        self.embedding_engine = EmbeddingEngine(usar_ollama=usar_ollama)
        self.vector_store = RAGVectorStore(self.embedding_engine)
        self.gerador = RespostaGerador(usar_ollama=usar_ollama)
        self.historico = []  # Memória de conversação

    def indexar_documentos(self, documentos):
        """Fase offline: indexar todos os documentos."""
        self.vector_store.indexar(documentos)

    def responder(self, pergunta, k=3, verbose=True):
        """
        Responde uma pergunta usando RAG.

        Pipeline:
          1. Embed(pergunta) → vetor da query
          2. TopK(vetor, vector_store) → documentos relevantes
          3. Prompt(documentos, pergunta) → prompt aumentado
          4. LLM(prompt) → resposta fundamentada
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 PERGUNTA: {pergunta}")
            print(f"{'='*60}")

        # RETRIEVAL: Buscar documentos relevantes
        docs_recuperados = self.vector_store.recuperar(pergunta, k=k)

        if verbose:
            print(f"\n📚 CONTEXTO RECUPERADO ({len(docs_recuperados)} chunks):")
            for i, doc in enumerate(docs_recuperados, 1):
                print(f"   {i}. [{doc['score']:.3f}] [{doc['tema']:15s}] "
                      f"{doc['texto'][:70]}...")

        # GENERATION: Gerar resposta com LLM
        resposta = self.gerador.gerar(pergunta, docs_recuperados)

        if verbose:
            print(f"\n💬 RESPOSTA:\n{resposta}")

        # Armazenar no histórico
        self.historico.append({
            'pergunta': pergunta,
            'docs_recuperados': docs_recuperados,
            'resposta': resposta
        })

        return resposta, docs_recuperados


# ═══════════════════════════════════════════════════════════════════
# 7. EXECUTAR O CHATBOT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("INICIANDO CHATBOT RAG")
print("=" * 70)

# Instanciar e indexar documentos
chatbot = ChatbotRAG()
chatbot.indexar_documentos(BASE_CONHECIMENTO)

# Perguntas de teste que exercitam diferentes partes da base de conhecimento
perguntas_teste = [
    "Qual é a frequência mínima do curso?",
    "Como funciona o algoritmo de Redes Neurais Convolucionais?",
    "Para que serve o RAG e como ele reduz alucinações?",
    "Quais ferramentas posso usar para construir aplicações com LLMs?",
    "Como são distribuídas as notas do curso?",
]

todos_scores = []  # Para visualização posterior

for pergunta in perguntas_teste:
    resposta, docs = chatbot.responder(pergunta, k=3, verbose=True)
    todos_scores.append({
        'pergunta': pergunta,
        'docs': docs
    })

# ═══════════════════════════════════════════════════════════════════
# 8. VISUALIZAÇÕES DO PIPELINE RAG
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GERANDO VISUALIZAÇÕES DO PIPELINE RAG")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Chatbot RAG - Análise do Pipeline de Retrieval', fontsize=14, fontweight='bold')

# --- Gráfico 1: Scores de recuperação por pergunta ---
ax1 = axes[0, 0]

# Pegar os scores das perguntas de teste
labels_curtos = [f"P{i+1}" for i in range(len(todos_scores))]
scores_top1 = [q['docs'][0]['score'] if q['docs'] else 0 for q in todos_scores]
scores_top2 = [q['docs'][1]['score'] if len(q['docs']) > 1 else 0 for q in todos_scores]
scores_top3 = [q['docs'][2]['score'] if len(q['docs']) > 2 else 0 for q in todos_scores]

x = np.arange(len(labels_curtos))
w = 0.25
bars1 = ax1.bar(x - w, scores_top1, w, label='Top-1', color='#2196F3', alpha=0.85)
bars2 = ax1.bar(x, scores_top2, w, label='Top-2', color='#4CAF50', alpha=0.85)
bars3 = ax1.bar(x + w, scores_top3, w, label='Top-3', color='#FF9800', alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(labels_curtos)
ax1.set_ylabel('Cosine Similarity Score')
ax1.set_ylim(0, 1.1)
ax1.set_title('Scores de Retrieval por Pergunta\n(maior score = maior relevância)')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.6, label='Threshold mín. (0.3)')

# Adicionar legendas das perguntas
legenda_perguntas = '\n'.join([f"P{i+1}: {p[:45]}..." for i, p in enumerate(perguntas_teste)])
ax1.text(0.02, 0.98, legenda_perguntas, transform=ax1.transAxes,
         fontsize=6.5, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# --- Gráfico 2: Temas mais recuperados ---
ax2 = axes[0, 1]

contagem_temas = defaultdict(int)
for q_info in todos_scores:
    for doc in q_info['docs']:
        contagem_temas[doc['tema']] += 1

temas_lista = list(contagem_temas.keys())
contagens_lista = list(contagem_temas.values())

# Ordenar por contagem
ordem = np.argsort(contagens_lista)[::-1]
temas_ord = [temas_lista[i] for i in ordem]
contagens_ord = [contagens_lista[i] for i in ordem]

cores_temas = plt.cm.Set3(np.linspace(0, 1, len(temas_ord)))
bars = ax2.barh(temas_ord, contagens_ord, color=cores_temas, alpha=0.85)
ax2.set_xlabel('Frequência de Recuperação')
ax2.set_title('Temas Mais Recuperados\n(frequência nos top-3 de todas as queries)')
ax2.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, contagens_ord):
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
             str(val), va='center', fontweight='bold')

# --- Gráfico 3: Heatmap de relevância ---
ax3 = axes[1, 0]

# Calcular scores de cada pergunta para cada documento
from sklearn.metrics.pairwise import cosine_similarity
emb_store = chatbot.vector_store
n_perguntas = len(perguntas_teste)
n_docs = len(emb_store.documentos)

# Criar matriz pergunta × documento
scores_completos = np.zeros((n_perguntas, n_docs))
for i, pergunta in enumerate(perguntas_teste):
    q_emb = chatbot.embedding_engine.encode(pergunta).reshape(1, -1)
    sim = cosine_similarity(q_emb, emb_store.embeddings)[0]
    scores_completos[i] = sim

im = ax3.imshow(scores_completos, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.8)
plt.colorbar(im, ax=ax3, label='Cosine Similarity')

ax3.set_yticks(range(n_perguntas))
ax3.set_yticklabels([f"P{i+1}" for i in range(n_perguntas)], fontsize=9)
ax3.set_xlabel('Índice do Documento')
ax3.set_title('Heatmap: Relevância Query × Documento\n(mais amarelo = mais relevante)')

# Marcar os top-3 de cada query
for i in range(n_perguntas):
    top3 = np.argsort(scores_completos[i])[-3:]
    for j in top3:
        ax3.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='blue', linewidth=2))

# --- Gráfico 4: Diagrama do Pipeline RAG ---
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Pipeline RAG - Fluxo de Dados')

# Setas e boxes do pipeline
etapas = [
    (5, 9.0, 'Pergunta do Usuário', '#E3F2FD', '❓'),
    (5, 7.5, 'Embedding da Query\n(nomic-embed-text)', '#FFF9C4', '🧮'),
    (5, 6.0, 'Busca por\nSimilaridade Cosine', '#E8F5E9', '🔍'),
    (5, 4.5, 'Top-K Chunks\nRelevantes', '#FFF3E0', '📚'),
    (5, 3.0, 'Prompt Aumentado\n(contexto + pergunta)', '#FCE4EC', '📝'),
    (5, 1.5, 'LLM Gera Resposta\n(llama3.2 / template)', '#EDE7F6', '🤖'),
]

for (x, y, txt, cor, emoji) in etapas:
    box = mpatches.FancyBboxPatch(
        (x - 2.5, y - 0.55), 5, 1.1,
        boxstyle="round,pad=0.1", facecolor=cor, edgecolor='gray', linewidth=1.5
    )
    ax4.add_patch(box)
    ax4.text(x, y + 0.05, f"{emoji} {txt}", ha='center', va='center',
             fontsize=8.5, fontweight='bold')

# Setas entre etapas
for i in range(len(etapas) - 1):
    y_start = etapas[i][1] - 0.55
    y_end = etapas[i + 1][1] + 0.55
    ax4.annotate('', xy=(5, y_end), xytext=(5, y_start),
                 arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

# Labels das fases
fases = [
    (8.5, 8.7, 'RETRIEVAL', '#2196F3'),
    (8.5, 4.5, 'AUGMENTATION', '#4CAF50'),
    (8.5, 1.5, 'GENERATION', '#9C27B0'),
]
for (x, y, label, cor) in fases:
    ax4.text(x, y, label, ha='center', va='center', fontsize=8,
             fontweight='bold', color=cor, rotation=90)

plt.tight_layout()
plt.savefig('rag_chatbot_analise.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: rag_chatbot_analise.png")

# ═══════════════════════════════════════════════════════════════════
# 9. INTERFACE INTERATIVA (opcional - descomentar para usar)
# ═══════════════════════════════════════════════════════════════════

def iniciar_chat_interativo():
    """
    Inicia um loop de chat interativo no terminal.
    Digitar 'sair' ou 'exit' encerra o chat.
    """
    print("\n" + "=" * 60)
    print("CHATBOT RAG INTERATIVO (digite 'sair' para encerrar)")
    print("=" * 60)

    while True:
        try:
            pergunta = input("\n🙋 Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not pergunta:
            continue

        if pergunta.lower() in ('sair', 'exit', 'quit'):
            print("👋 Encerrando chatbot. Até logo!")
            break

        resposta, _ = chatbot.responder(pergunta, k=3, verbose=True)

# Para iniciar o chat interativo manualmente, chamar:
# iniciar_chat_interativo()

print("\n" + "=" * 70)
print("DEMONSTRAÇÃO CONCLUÍDA!")
print("=" * 70)
print(f"\n📊 Estatísticas:")
print(f"   Documentos indexados: {len(BASE_CONHECIMENTO)}")
print(f"   Perguntas respondidas: {len(chatbot.historico)}")
print(f"   Modo: {'Ollama (LLM real)' if OLLAMA_DISPONIVEL else 'Fallback TF-IDF'}")

print("""
💡 DICAS PARA PRODUÇÃO:
   1. Instalar Ollama: curl -fsSL https://ollama.com/install.sh | sh
   2. Baixar modelo: ollama pull llama3.2
   3. Baixar embedder: ollama pull nomic-embed-text
   4. Usar ChromaDB com persist_directory para salvar os embeddings
   5. Implementar reranking com CrossEncoder para melhor precisão
""")
