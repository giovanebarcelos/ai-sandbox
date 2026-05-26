"""
GO1713 - Interface Web para Chatbot RAG com Streamlit
======================================================
Demonstra a estrutura de uma interface web RAG usando Streamlit.
O Streamlit transforma scripts Python em aplicações web sem HTML/JS.

Instalação:
    pip install streamlit

Execução:
    streamlit run GO1713-27StreamlitUi.py

Conceito: Streamlit usa session_state para manter o histórico da conversa
entre re-renders (a cada mensagem, a página "re-executa" de cima para baixo,
mas session_state persiste). O padrão chat_input + chat_message é o idioma
padrão para chatbots em Streamlit >= 1.23.
"""

import sys
import subprocess


def verificar_streamlit():
    """Verifica se Streamlit está instalado e orienta o usuário."""
    try:
        import streamlit
        return True
    except ImportError:
        print("Streamlit não está instalado.")
        print("Para instalar: pip install streamlit")
        return False


def mostrar_codigo_completo():
    """Exibe o código da aplicação para estudo."""
    codigo = '''
# ─────────────────────────────────────────────────────────────
# Estrutura de um Chatbot RAG com Streamlit
# ─────────────────────────────────────────────────────────────

import streamlit as st

# Configurar página (deve ser a 1ª chamada Streamlit)
st.set_page_config(
    page_title="Chatbot RAG",
    page_icon="🤖",
    layout="wide"
)

# Título e descrição
st.title("Chatbot RAG")
st.caption("Powered by Ollama + FAISS")

# ─── Estado da sessão ────────────────────────────────────────
# session_state persiste entre re-renders (cada input re-executa o script)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Exibir histórico ────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ─── Input do usuário ────────────────────────────────────────
if prompt := st.chat_input("Sua pergunta"):
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Exibir mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.write(prompt)

    # Gerar resposta via RAG
    with st.chat_message("assistant"):
        with st.spinner("Buscando na base de documentos..."):
            # Em produção: substituir pela chamada real ao RAG
            response = rag_query(prompt)
            st.write(response)

    # Salvar resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Forçar re-render para mostrar nova mensagem
    st.rerun()

# ─── Sidebar com controles ───────────────────────────────────
with st.sidebar:
    st.header("Configurações")
    model = st.selectbox("Modelo", ["llama3.2", "mistral", "phi3"])
    top_k = st.slider("Documentos recuperados", 1, 10, 3)
    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()
'''
    print(codigo)


def demo_sem_streamlit():
    """
    Simula o fluxo do chatbot RAG no terminal.
    Demonstra a lógica sem precisar do Streamlit instalado.
    """
    print("=" * 60)
    print("GO1713 - INTERFACE STREAMLIT PARA CHATBOT RAG")
    print("=" * 60)
    print()
    print("Streamlit nao instalado. Simulando fluxo no terminal.")
    print("Para a interface web real: pip install streamlit")
    print("                          streamlit run GO1713-27StreamlitUi.py")
    print()

    # Simular a base de conhecimento (RAG simplificado)
    base = {
        "retorno": "Nossa política permite devoluções em até 30 dias.",
        "frete": "Frete grátis para compras acima de R$ 150.",
        "garantia": "Produtos têm garantia de 12 meses contra defeitos.",
        "entrega": "Prazo de entrega: 5 a 10 dias úteis.",
        "pagamento": "Parcelamos em até 12x sem juros no cartão.",
    }

    def rag_query(pergunta: str) -> str:
        pergunta_lower = pergunta.lower()
        for chave, resposta in base.items():
            if chave in pergunta_lower:
                return f"[RAG] {resposta}"
        return "[RAG] Não encontrei essa informação na base de documentos."

    # Simular sessão de chat (session_state equivalente)
    historico = []

    perguntas_demo = [
        "Qual a política de retorno?",
        "Como funciona o frete?",
        "Tem garantia?",
        "Boa tarde!",
    ]

    print("─" * 60)
    print("Simulacao de sessao de chat:")
    print("─" * 60)

    for pergunta in perguntas_demo:
        historico.append({"role": "user", "content": pergunta})
        resposta = rag_query(pergunta)
        historico.append({"role": "assistant", "content": resposta})

        print(f"\nUsuario: {pergunta}")
        print(f"Chatbot: {resposta}")

    print()
    print(f"Historico total: {len(historico)} mensagens")
    print()

    mostrar_codigo_completo()


# ─────────────────────────────────────────────────────────────
# APLICAÇÃO STREAMLIT (executada com 'streamlit run ...')
# ─────────────────────────────────────────────────────────────

def rag_query(pergunta: str) -> str:
    """RAG simplificado — substituir pelo sistema real em produção."""
    base = {
        "retorno": "Nossa política permite devoluções em até 30 dias.",
        "frete": "Frete grátis para compras acima de R$ 150.",
        "garantia": "Produtos têm garantia de 12 meses.",
        "entrega": "Prazo de entrega: 5 a 10 dias úteis.",
    }
    for chave, resposta in base.items():
        if chave in pergunta.lower():
            return f"Com base nos nossos documentos: {resposta}"
    return "Não encontrei essa informação na base. Tente reformular a pergunta."


if __name__ == "__main__":
    # Quando executado diretamente (python GO1713...) → demo no terminal
    demo_sem_streamlit()
else:
    # Quando executado via 'streamlit run GO1713...' → interface web
    try:
        import streamlit as st

        st.set_page_config(page_title="Chatbot RAG", page_icon="🤖")
        st.title("Chatbot RAG")
        st.caption("Simulação com base de conhecimento interna")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Sua pergunta"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            response = rag_query(prompt)
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    except ImportError:
        pass
