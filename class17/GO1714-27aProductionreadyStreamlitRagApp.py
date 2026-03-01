# GO1714-27aProductionreadyStreamlitRagApp
import streamlit as st
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class StreamlitRAGApp:
    """
    Aplicação Streamlit completa para RAG com:
    - Chat interface
    - Document upload
    - Source citation
    - Analytics dashboard
    - Configuration panel
    """

    def __init__(self, rag_system):
        self.rag = rag_system
        self.init_session_state()

    def init_session_state(self):
        """Inicializa estado da sessão"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = 0
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'temperature': 0.7,
                'k_documents': 3,
                'model': 'llama3.2'
            }

    def render_sidebar(self):
        """Renderiza sidebar com configurações"""
        with st.sidebar:
            st.title("⚙️ Configurações")

            # Model selection
            st.session_state.settings['model'] = st.selectbox(
                "Modelo LLM",
                ['llama3.2', 'mistral', 'phi3'],
                index=0
            )

            # Temperature
            st.session_state.settings['temperature'] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Maior = mais criativo"
            )

            # Number of documents
            st.session_state.settings['k_documents'] = st.slider(
                "Documentos para contexto",
                min_value=1,
                max_value=10,
                value=3,
                help="Quantos docs recuperar"
            )

            st.divider()

            # Document upload
            st.subheader("📁 Upload Documentos")
            uploaded_files = st.file_uploader(
                "Adicionar documentos",
                accept_multiple_files=True,
                type=['txt', 'pdf', 'md']
            )

            if uploaded_files and st.button("Processar Documentos"):
                with st.spinner("Processando..."):
                    count = self.process_uploads(uploaded_files)
                    st.success(f"✅ {count} documentos adicionados!")
                    st.session_state.documents_loaded += count

            # Statistics
            st.divider()
            st.subheader("📊 Estatísticas")
            st.metric("Documentos", st.session_state.documents_loaded)
            st.metric("Perguntas", len(st.session_state.messages) // 2)

    def process_uploads(self, files) -> int:
        """Processa arquivos enviados"""
        count = 0
        for file in files:
            # Read file content
            content = file.read().decode('utf-8')

            # Add to RAG system (simulado)
            # self.rag.add_documents([{
            #     'text': content,
            #     'source': file.name
            # }])
            count += 1
            time.sleep(0.5)  # Simulate processing

        return count

    def render_chat_interface(self):
        """Renderiza interface de chat"""
        st.title("💬 RAG Chatbot")
        st.caption("Powered by Ollama + ChromaDB")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Fontes"):
                        for source in message["sources"]:
                            st.caption(f"• {source}")

        # Chat input
        if prompt := st.chat_input("Faça sua pergunta..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    # Simulate RAG query
                    response_text = self.generate_response(prompt)
                    sources = ["doc1.pdf", "doc2.md", "doc3.txt"]

                    st.markdown(response_text)

                    # Show sources
                    with st.expander("📚 Fontes"):
                        for source in sources:
                            st.caption(f"• {source}")

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources,
                    "timestamp": datetime.now()
                })

                # Track query
                st.session_state.query_history.append({
                    'question': prompt,
                    'answer_length': len(response_text),
                    'sources_count': len(sources),
                    'timestamp': datetime.now()
                })

        # Clear chat button
        if len(st.session_state.messages) > 0:
            if st.button("🗑️ Limpar Chat"):
                st.session_state.messages = []
                st.rerun()

    def generate_response(self, query: str) -> str:
        """Gera resposta (simulado)"""
        # Simulate processing time
        time.sleep(1)

        # Simulated response
        responses = [
            f"Baseado nos documentos, posso responder sobre '{query}'. Esta é uma resposta exemplo que demonstra como o sistema RAG funciona integrando recuperação de documentos com geração de texto.",
            f"Sobre '{query}': encontrei informações relevantes em múltiplos documentos. O sistema RAG combina busca semântica com LLMs para fornecer respostas contextualizadas e precisas.",
            f"Sua pergunta sobre '{query}' é interessante. Nos documentos disponíveis, há informações que indicam que este tópico é importante e requer atenção cuidadosa aos detalhes fornecidos."
        ]

        import random
        return random.choice(responses)

    def render_analytics_tab(self):
        """Renderiza dashboard de analytics"""
        st.header("📊 Analytics Dashboard")

        if not st.session_state.query_history:
            st.info("Nenhuma query ainda. Comece fazendo perguntas!")
            return

        # Create dataframe
        df = pd.DataFrame(st.session_state.query_history)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", len(df))

        with col2:
            avg_length = df['answer_length'].mean()
            st.metric("Avg Answer Length", f"{avg_length:.0f} chars")

        with col3:
            avg_sources = df['sources_count'].mean()
            st.metric("Avg Sources", f"{avg_sources:.1f}")

        with col4:
            if len(df) > 1:
                time_diff = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 60
                qpm = len(df) / max(time_diff, 1)
                st.metric("Queries/Min", f"{qpm:.2f}")
            else:
                st.metric("Queries/Min", "N/A")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Answer Length Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df['answer_length'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Answer Length (chars)')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.subheader("Sources Used")
            fig, ax = plt.subplots(figsize=(8, 4))
            sources_counts = df['sources_count'].value_counts().sort_index()
            ax.bar(sources_counts.index, sources_counts.values, color='lightgreen', alpha=0.7)
            ax.set_xlabel('Number of Sources')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Query timeline
        st.subheader("Query Timeline")
        df_timeline = df.set_index('timestamp').resample('1min').size()
        st.line_chart(df_timeline)

        # Recent queries table
        st.subheader("Recent Queries")
        recent = df[['question', 'answer_length', 'sources_count']].tail(10)
        st.dataframe(recent, use_container_width=True)

    def run(self):
        """Executa aplicação"""
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Sidebar
        self.render_sidebar()

        # Main content - tabs
        tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

        with tab1:
            self.render_chat_interface()

        with tab2:
            self.render_analytics_tab()

# === EXECUTAR APP ===

# Para rodar: streamlit run app.py
if __name__ == "__main__":
    # Simulated RAG system
    class MockRAG:
        pass

    rag_system = MockRAG()
    app = StreamlitRAGApp(rag_system)
    app.run()
