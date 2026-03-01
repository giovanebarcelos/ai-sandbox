# GO1712-24AgentsComRag
# Simple agent with RAG tool
from langchain.agents import initialize_agent
from langchain.tools import Tool


if __name__ == "__main__":
    rag_tool = Tool(
        name="RAG",
        func=lambda q: rag_system.query(q),
        description="Search internal documents"
    )

    agent = initialize_agent([rag_tool], llm, verbose=True)
    agent.run("What is our return policy?")
