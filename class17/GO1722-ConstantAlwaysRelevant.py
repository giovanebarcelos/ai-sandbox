# GO1722-ConstantAlwaysRelevant
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Buffer memory


if __name__ == "__main__":
    memory = ConversationBufferMemory()

    # Summary memory (with LLM)
    memory = ConversationSummaryMemory(llm=llm)
