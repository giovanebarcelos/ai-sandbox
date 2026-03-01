# GO1724-Openai
# Código OpenAI funciona com Ollama
from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1")
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hi!"}]
    )
