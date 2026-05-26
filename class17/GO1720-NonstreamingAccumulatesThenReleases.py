"""
GO1720 - Streaming vs Non-Streaming com Ollama
===============================================
Demonstra a diferença entre respostas com e sem streaming em LLMs.

Non-streaming: acumula TODA a resposta no servidor antes de devolver.
               O usuário fica esperando (pode ser 10-30 segundos).

Streaming: cada token gerado é enviado imediatamente ao cliente.
           O usuário vê a resposta sendo formada em tempo real.
           Sensação de latência muito menor (mesmo tempo total).

Instalação:
  pip install ollama
  ollama pull llama3.2  # Download do modelo (~2 GB)
  ollama serve          # Iniciar servidor Ollama

Uso: python GO1720-NonstreamingAccumulatesThenReleases.py
"""

import time


def stream_ollama(prompt: str, model: str = 'llama3.2') -> str:
    """
    Streaming: exibe cada token conforme é gerado pelo modelo.
    O usuário vê o texto aparecer progressivamente.
    """
    import ollama
    full_response = ""
    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True  # Ativa modo streaming
    )
    for chunk in stream:
        token = chunk['message']['content']
        # Exibe imediatamente sem quebra de linha
        print(token, end='', flush=True)
        full_response += token
    print()  # Nova linha ao final
    return full_response


def sem_stream_ollama(prompt: str, model: str = 'llama3.2') -> str:
    """
    Non-streaming: acumula toda a resposta antes de exibir.
    Comportamento padrão de uma API síncrona.
    """
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=False  # Aguarda resposta completa
    )
    return response['message']['content']


def demo_sem_ollama() -> None:
    """
    Demonstração da diferença streaming vs non-streaming
    sem precisar do Ollama instalado.
    """
    print("\nSIMULACAO: STREAMING vs NON-STREAMING:")
    print("─" * 60)

    # Simular resposta token por token
    resposta = "Redes neurais são sistemas computacionais inspirados no cérebro humano."
    tokens = resposta.split()

    print("\n  [NON-STREAMING] Aguarda toda a resposta...")
    print("  (o usuário espera sem ver nada por vários segundos)")
    time.sleep(0.5)  # Simula tempo de espera
    print(f"  Resposta completa: {resposta}")

    print()
    print("  [STREAMING] Exibe cada token conforme é gerado:")
    print("  Resposta: ", end='', flush=True)
    for token in tokens:
        print(token + " ", end='', flush=True)
        time.sleep(0.08)  # Simula geração token a token
    print()

    print()
    print("  Diferenca percebida pelo usuário:")
    print("    Non-streaming: espera 10s → vê tudo de uma vez")
    print("    Streaming:     0.1s → vê 'Redes' → 'neurais' → ...")
    print("    (mesmo tempo total, mas experiência muito melhor!)")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1720 - STREAMING vs NON-STREAMING (OLLAMA)")
    print("=" * 60)

    print("\nCONCEITO:")
    print()
    print("  Non-streaming: acumula TODA a resposta, libera de uma vez")
    print("  Streaming:     cada token liberado imediatamente ao cliente")
    print()
    print("  Para ativar streaming com Ollama:")
    print("    ollama.chat(model='llama3.2', messages=[...], stream=True)")
    print()
    print("  Para ativar streaming com OpenAI:")
    print("    client.chat.completions.create(..., stream=True)")

    # Tentar com Ollama real
    try:
        import ollama
        prompt = "Explique o que é inteligência artificial em 1 frase curta."
        print()
        print("─" * 60)
        print("TESTE COM OLLAMA REAL:")
        print("─" * 60)

        # Non-streaming
        print("\n  [NON-STREAMING]")
        print(f"  Prompt: {prompt}")
        t0 = time.time()
        resposta_ns = sem_stream_ollama(prompt)
        t_ns = time.time() - t0
        print(f"  Resposta: {resposta_ns[:100]}")
        print(f"  Tempo: {t_ns:.2f}s (esperou tudo antes de exibir)")

        # Streaming
        print()
        print("  [STREAMING]")
        print(f"  Prompt: {prompt}")
        t0 = time.time()
        print("  Resposta: ", end='', flush=True)
        resposta_s = stream_ollama(prompt)
        t_s = time.time() - t0
        print(f"  Tempo total: {t_s:.2f}s (exibido progressivamente)")

    except ImportError:
        print("\nOllama nao instalado.")
        print("Execute: pip install ollama")
        demo_sem_ollama()
    except Exception as e:
        print(f"\nErro ao conectar ao Ollama: {e}")
        print("Certifique-se que o servidor Ollama esta rodando: ollama serve")
        demo_sem_ollama()
