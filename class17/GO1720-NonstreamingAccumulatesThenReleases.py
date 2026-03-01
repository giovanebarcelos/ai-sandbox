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
