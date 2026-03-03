# GO1456A-19aSmallLanguageModelsSlms
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

print("📱 SMALL LANGUAGE MODELS - Phi-3 Mini")
print("=" * 70)

# ──────────────────────────────────────────
# OPÇÃO 1: HUGGING FACE (FP16)
# ──────────────────────────────────────────

print("\n🔄 Carregando Phi-3 Mini (3.8B params)...")

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # FP16 para memória
    device_map="auto",          # Distribui entre GPU/CPU
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(f"✅ Modelo carregado")
print(f"   Parâmetros: 3.8B")
print(f"   Memória GPU: ~8 GB (FP16)")
print(f"   Contexto: 4K tokens\n")

# ──────────────────────────────────────────
# INFERÊNCIA
# ──────────────────────────────────────────

prompt = """
<|system|>
You are a helpful AI assistant.<|end|>
<|user|>
Explique o que são Small Language Models (SLMs) em 2-3 frases.<|end|>
<|assistant|>
"""

print("💬 Prompt:")
print(prompt)
print("\n🤖 Resposta:")

start = time.time()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("<|assistant|>")[-1].strip()

latency = (time.time() - start) * 1000

print(response)
print(f"\n⏱️ Latência: {latency:.0f}ms")
print(f"📊 Tokens gerados: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

# ──────────────────────────────────────────
# OPÇÃO 2: OLLAMA (MAIS FÁCIL)
# ──────────────────────────────────────────

print("\n" + "="*70)
print("🦙 ALTERNATIVA: OLLAMA (mais fácil)")
print("="*70)

ollama_example = '''
# Terminal 1: Iniciar servidor
ollama serve

# Terminal 2: Baixar modelo
ollama pull phi3

# Terminal 3: Testar
ollama run phi3 "Explique SLMs em uma frase"

# Python API
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    "model": "phi3",
    "prompt": "Explique SLMs em uma frase",
    "stream": False
})

print(response.json()['response'])
'''

print(ollama_example)

# ──────────────────────────────────────────
# COMPARAÇÃO DE HARDWARE
# ──────────────────────────────────────────

print("\n📊 REQUISITOS DE HARDWARE\n")

hardware_table = [
    ["Modelo", "Params", "FP16", "INT4", "Hardware Mínimo"],
    ["-" * 20, "-" * 8, "-" * 8, "-" * 8, "-" * 25],
    ["Llama 3.2 1B", "1B", "2 GB", "0.5 GB", "Smartphone (4GB RAM)"],
    ["Gemma 2B", "2B", "4 GB", "1 GB", "Tablet (6GB RAM)"],
    ["Phi-3 Mini", "3.8B", "8 GB", "2 GB", "Laptop M1 (8GB RAM)"],
    ["Llama 3.2 3B", "3B", "6 GB", "1.5 GB", "Laptop (8GB RAM)"],
    ["Mistral 7B", "7B", "14 GB", "3.5 GB", "Desktop RTX 3060 (12GB)"],
    ["Llama 2 13B", "13B", "26 GB", "6.5 GB", "Desktop RTX 4090 (24GB)"],
]

for row in hardware_table:
    print(f"{row[0]:<20} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]}")

print("\n💡 Dica: Use quantização INT4 para rodar em hardware mais limitado!")
