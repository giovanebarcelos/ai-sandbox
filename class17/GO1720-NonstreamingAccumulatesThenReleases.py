# GO1720-NonstreamingAccumulatesThenReleases
import ollama

def stream_ollama(prompt: str):
    stream = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


if __name__ == '__main__':
    print("=== Streaming com Ollama ===")
    print()
    print("  Este código requer:")
    print("    pip install ollama")
    print("    ollama pull llama3.2  # Baixar o modelo (~2GB)")
    print()
    print("  Uso:")
    print("    stream_ollama('Explique o que é uma rede neural em 2 frases.')")
    print()

    try:
        import ollama
        prompt = "Explique o que é inteligência artificial em 1 frase."
        print(f"Prompt: {prompt}")
        print("Resposta: ", end='')
        stream_ollama(prompt)
        print()
    except ImportError:
        print("  ollama não instalado. Execute: pip install ollama")
    except Exception as e:
        print(f"  Erro ao conectar ao Ollama: {e}")
        print("  Certifique-se que o servidor Ollama está rodando: ollama serve")
