# GO1713-27StreamlitUi
pip install streamlit
import streamlit as st
st.title("Chatbot RAG")
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
if prompt := st.chat_input("Sua pergunta"):
    st.session_state.messages.append({"role": "user",
                                      "content": prompt})
    response = rag_query(prompt)
    st.session_state.messages.append({"role": "assistant",
                                      "content": response})
    st.rerun()
