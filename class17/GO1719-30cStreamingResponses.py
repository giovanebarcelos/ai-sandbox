# GO1719-30cStreamingResponses
import time
from typing import Iterator, Dict
import asyncio

class StreamingRAG:
    """
    RAG com streaming de respostas token-by-token

    Benefícios:
    - UX melhor (usuário vê resposta aparecendo)
    - Lower perceived latency
    - Pode cancelar geração cedo
    - Melhor para respostas longas
    """

    def __init__(self, model_name: str = 'llama3.2'):
        self.model = model_name
        print(f"✅ Streaming RAG initialized with {model_name}")

    def _retrieve_context(self, query: str) -> str:
        """Recupera documentos relevantes"""
        time.sleep(0.3)  # Simulate retrieval
        return """
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn from data without being explicitly programmed. 
        It uses algorithms to identify patterns and make predictions.
        """

    def stream_query(self, query: str) -> Iterator[Dict]:
        """
        Stream resposta token-by-token

        Yields:
            Dict com 'token', 'metadata', 'done'
        """
        # Step 1: Retrieve context
        yield {
            'type': 'status',
            'message': 'Retrieving documents...',
            'done': False
        }

        context = self._retrieve_context(query)

        yield {
            'type': 'context',
            'message': f'Found {len(context.split())} words of context',
            'done': False
        }

        # Step 2: Generate prompt
        prompt = f"""Context: {context}

Question: {query}

Answer: """

        yield {
            'type': 'status',
            'message': 'Generating answer...',
            'done': False
        }

        # Step 3: Stream tokens
        # Simulate LLM generating tokens
        response_text = f"Based on the context, {query} can be answered as follows: Machine learning enables systems to learn from data and improve over time without explicit programming."

        words = response_text.split()
        for i, word in enumerate(words):
            time.sleep(0.05)  # Simulate generation latency

            yield {
                'type': 'token',
                'token': word + ' ',
                'done': False,
                'metadata': {
                    'token_index': i,
                    'total_tokens': len(words)
                }
            }

        # Final yield
        yield {
            'type': 'done',
            'message': 'Generation complete',
            'done': True,
            'metadata': {
                'total_tokens': len(words),
                'total_time': len(words) * 0.05
            }
        }

    async def stream_query_async(self, query: str) -> Iterator[Dict]:
        """Async version for better concurrency"""
        context = self._retrieve_context(query)

        yield {
            'type': 'context',
            'message': 'Context retrieved',
            'done': False
        }

        response_text = f"Async response for: {query}"
        words = response_text.split()

        for i, word in enumerate(words):
            await asyncio.sleep(0.05)
            yield {
                'type': 'token',
                'token': word + ' ',
                'done': False
            }

        yield {'type': 'done', 'done': True}

# === DEMO: Sync Streaming ===

print("🌊 Demo: Streaming RAG\n")
print("="*70)

rag = StreamingRAG()

query = "What is machine learning?"
print(f"Query: {query}\n")
print("Response: ", end='', flush=True)

full_response = ""
start_time = time.time()
first_token_time = None

for chunk in rag.stream_query(query):
    if chunk['type'] == 'status':
        print(f"\n[{chunk['message']}]")

    elif chunk['type'] == 'context':
        print(f"[{chunk['message']}]")
        print("\nAnswer: ", end='', flush=True)

    elif chunk['type'] == 'token':
        if first_token_time is None:
            first_token_time = time.time()

        print(chunk['token'], end='', flush=True)
        full_response += chunk['token']

    elif chunk['type'] == 'done':
        total_time = time.time() - start_time
        ttft = first_token_time - start_time if first_token_time else 0  # Time To First Token

        print(f"\n\n✅ Done!")
        print(f"   Time to first token: {ttft:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Tokens: {chunk['metadata']['total_tokens']}")
        print(f"   Tokens/second: {chunk['metadata']['total_tokens']/total_time:.1f}")

# === Demo: Streamlit Integration ===

streamlit_code = '''
import streamlit as st

# Streamlit com streaming
st.title("RAG Chatbot (Streaming)")

if prompt := st.chat_input("Sua pergunta"):
    # User message
    with st.chat_message("user"):
        st.write(prompt)

    # Assistant message with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in rag.stream_query(prompt):
            if chunk['type'] == 'token':
                full_response += chunk['token']
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
'''

print("\n\n📄 Streamlit Integration Code:")
print(streamlit_code)

# Visualize streaming performance
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Token generation timeline
ax = axes[0, 0]
n_tokens = 30
times = np.cumsum([0.05] * n_tokens)  # Cumulative time
ax.plot(range(n_tokens), times, marker='o', linewidth=2, markersize=4)
ax.set_xlabel('Token Index')
ax.set_ylabel('Cumulative Time (s)')
ax.set_title('Token Generation Timeline')
ax.grid(alpha=0.3)

# Highlight TTFT
ax.axhline(y=times[0], color='red', linestyle='--', alpha=0.5)
ax.text(n_tokens/2, times[0] + 0.1, f'TTFT: {times[0]:.2f}s', color='red')

# 2. Streaming vs Non-streaming UX
ax = axes[0, 1]
scenarios = ['Non-streaming\n(wait for all)', 'Streaming\n(token-by-token)']
perceived_latency = [3.0, 0.5]  # seconds
colors = ['lightcoral', 'lightgreen']
bars = ax.barh(scenarios, perceived_latency, color=colors, alpha=0.7)
ax.set_xlabel('Perceived Latency (s)')
ax.set_title('User Experience: Latency Perception')
ax.grid(axis='x', alpha=0.3)

for bar, lat in zip(bars, perceived_latency):
    width = bar.get_width()
    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
            f'{lat:.1f}s', ha='left', va='center')

# 3. Throughput comparison
ax = axes[1, 0]
batch_sizes = [1, 4, 8, 16]
streaming_throughput = [20, 75, 140, 250]  # tokens/sec
non_streaming_throughput = [15, 50, 90, 150]

ax.plot(batch_sizes, streaming_throughput, marker='o', label='Streaming', linewidth=2)
ax.plot(batch_sizes, non_streaming_throughput, marker='s', label='Non-streaming', linewidth=2)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (tokens/sec)')
ax.set_title('Throughput: Streaming vs Non-streaming')
ax.legend()
ax.grid(alpha=0.3)

# 4. Memory usage over time
ax = axes[1, 1]
time_points = np.linspace(0, 3, 50)
# Streaming: constant memory
streaming_mem = np.ones_like(time_points) * 500  # MB
# Non-streaming: accumulates then releases
non_streaming_mem = np.where(time_points < 2.5, time_points * 400, 100)

ax.plot(time_points, streaming_mem, label='Streaming', linewidth=2, color='green')
ax.plot(time_points, non_streaming_mem, label='Non-streaming', linewidth=2, color='red')
ax.fill_between(time_points, 0, streaming_mem, alpha=0.2, color='green')
ax.fill_between(time_points, 0, non_streaming_mem, alpha=0.2, color='red')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Memory Usage (MB)')
ax.set_title('Memory Usage Pattern')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('streaming_rag_performance.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: streaming_rag_performance.png")

print("\n✅ Streaming RAG implementado!")
print("\n💡 BENEFITS:")
print("   - Better UX (immediate feedback)")
print("   - Lower perceived latency")
print("   - Can cancel generation early")
print("   - Constant memory usage")
print("   - Better for long responses")
