# GO1706-19ProjetoChatbotRagComOllama
import ollama
import chromadb
from chromadb.utils import embedding_functions
import os


if __name__ == "__main__":
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="documentos",
        metadata={"description": "Documentação da empresa"}
    )

    def load_documents(directory):
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename)) as f:
                    content = f.read()
                    chunks = [content[i:i+500]
                              for i in range(0, len(content), 450)]
                    documents.extend([
                        {"text": chunk, "source": filename}
                        for chunk in chunks
                    ])
        return documents

    def add_to_vectorstore(documents):
        for i, doc in enumerate(documents):
            emb = ollama.embeddings(
                model='nomic-embed-text',
                prompt=doc['text']
            )['embedding']

            collection.add(
                ids=[f"doc_{i}"],
                embeddings=[emb],
                documents=[doc['text']],
                metadatas=[{"source": doc['source']}]
            )

    def search(query, k=3):
        query_emb = ollama.embeddings(
            model='nomic-embed-text',
            prompt=query
        )['embedding']

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )
        return results['documents'][0]

    def rag_query(question):
        context_docs = search(question, k=3)
        context = "\n\n".join(context_docs)

        prompt = f"""Contexto:
    {context}

    Pergunta: {question}

    Responda baseado apenas no contexto acima."""

        response = ollama.generate(
            model='llama3.2',
            prompt=prompt
        )
        return response['response']

    def chatbot():
        print("Chatbot RAG (digite 'sair' para encerrar)")
        while True:
            question = input("\nVocê: ")
            if question.lower() == 'sair':
                break

            answer = rag_query(question)
            print(f"Bot: {answer}")

    # Exemplo de uso
    # docs = load_documents("./docs")
    # add_to_vectorstore(docs)
    # chatbot()
