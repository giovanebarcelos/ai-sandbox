"""
GO1724 - API OpenAI Compatível com Ollama (local)
==================================================
Demonstra que o Ollama expõe uma API compatível com OpenAI.
Isso permite usar a biblioteca openai apontando para um servidor local.

Instalação:
    pip install openai
    ollama pull llama3.2  # ~2 GB

Execução:
    Iniciar servidor: ollama serve
    Executar script : python GO1724-Openai.py

Conceito: o Ollama implementa a mesma API REST da OpenAI.
Basta trocar a base_url para http://localhost:11434/v1
e usar api_key="ollama" (qualquer string, não validado localmente).
Isso facilita migrar código de OpenAI para modelos locais.
"""

import sys


def demo_compatibilidade() -> None:
    """
    Demonstra a compatibilidade de API OpenAI ↔ Ollama com exemplo prático.
    Tenta conectar ao Ollama local; exibe instrução se não disponível.
    """
    print("=" * 60)
    print("GO1724 - API OPENAI COMPATIVEL COM OLLAMA")
    print("=" * 60)
    print()
    print("Conceito: Ollama implementa o mesmo protocolo REST da OpenAI.")
    print("Trocar 'api.openai.com' por 'localhost:11434' é suficiente.")
    print()

    # ─── Mostrar equivalência dos dois códigos ────────────────
    print("─" * 60)
    print("CODIGO OPENAI (nuvem):")
    print("─" * 60)
    codigo_openai = """
from openai import OpenAI

client = OpenAI(api_key="sk-...")   # Chave real da OpenAI

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Olá!"}]
)
print(response.choices[0].message.content)
"""
    print(codigo_openai)

    print("─" * 60)
    print("CODIGO OLLAMA (local) — mesma biblioteca, URL diferente:")
    print("─" * 60)
    codigo_ollama = """
from openai import OpenAI

# Apenas muda a base_url e o model
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"            # Qualquer string (Ollama não valida)
)

response = client.chat.completions.create(
    model="llama3.2",           # Modelo local instalado com 'ollama pull'
    messages=[{"role": "user", "content": "Olá!"}]
)
print(response.choices[0].message.content)
"""
    print(codigo_ollama)

    # ─── Tentar conexão real ──────────────────────────────────
    print("─" * 60)
    print("TENTANDO CONEXAO REAL COM OLLAMA LOCAL...")
    print("─" * 60)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[{"role": "user", "content": "Responda em uma frase: o que é IA?"}],
            max_tokens=60,
        )
        print("Conexao com Ollama bem-sucedida!")
        print(f"Resposta: {response.choices[0].message.content}")

    except ImportError:
        print("openai nao instalado. Execute: pip install openai")

    except Exception as e:
        print(f"Ollama nao disponivel: {e}")
        print()
        print("Para usar o Ollama localmente:")
        print("  1. Instalar: https://ollama.com/download")
        print("  2. Iniciar servidor: ollama serve")
        print("  3. Baixar modelo:    ollama pull llama3.2")
        print("  4. Executar script:  python GO1724-Openai.py")

    print()
    print("─" * 60)
    print("VANTAGENS DO OLLAMA (modelos locais):")
    print("─" * 60)
    vantagens = [
        ("Privacidade", "Dados nunca saem da sua maquina"),
        ("Custo zero", "Sem pagar por tokens apos baixar o modelo"),
        ("Offline", "Funciona sem internet"),
        ("Latencia", "Sem RTT de rede (depende do hardware local)"),
        ("Compatibilidade", "Mesmo codigo funciona com OpenAI ou Ollama"),
    ]
    for nome, desc in vantagens:
        print(f"  + {nome:15s}: {desc}")

    print()
    print("  Desvantagens:")
    print("  - Modelos locais sao menores que GPT-4o")
    print("  - Requer GPU ou CPU rapida para boa performance")


if __name__ == "__main__":
    demo_compatibilidade()
