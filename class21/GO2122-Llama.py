# GO2122 - Llama 3.2 1B quantizado -- Processamento Local em Tempo Real
# ═══════════════════════════════════════════════════════════════════
# Casos: processamento local, decisões em tempo real, sem cloud.
# Demonstramos como a QUANTIZAÇÃO (q4/q8/fp16) afeta a velocidade de
# geração (tokens por segundo) -- quanto menor a precisão, mais rápido.
# ═══════════════════════════════════════════════════════════════════
import time, json

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

print("=== Llama 3.2 1B Quantizado ===")
vantagens = [
    "Processamento 100% local",
    "Decisoes em tempo real",
    "Sem dependencia de cloud",
]
for v in vantagens:
    print(f"  - {v}")


# ─── PONTO-CHAVE: latência menor => mais tokens por segundo ───
def simular_llm_local(entrada, modelo="llama3.2-1b-q4", latencia_ms=45):
    time.sleep(latencia_ms / 1000)
    return {
        "modelo": modelo,
        "entrada": entrada[:50],
        "saida": f"[Resposta do {modelo}]",
        "tokens_por_segundo": round(1000 / latencia_ms * 10, 1),
    }


if __name__ == "__main__":
    for p in ["Qual o capital do Brasil?", "Explique ML em uma linha."]:
        resultado = simular_llm_local(p)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))

    # ─── GRÁFICO: throughput por nível de quantização ───
    # q4 (4 bits) é o mais leve/rápido; fp16 é o mais pesado/preciso
    quantizacoes = {
        "q4 (4-bit)":  45,   # latência por token (ms) -> mais rápido
        "q8 (8-bit)":  70,
        "fp16 (16-bit)": 120,  # mais lento, porém maior qualidade
    }
    nomes = list(quantizacoes.keys())
    tokens_s = [round(1000 / ms * 10, 1) for ms in quantizacoes.values()]

    plt.figure(figsize=(9, 5))
    barras = plt.bar(nomes, tokens_s, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    for b, t in zip(barras, tokens_s):
        plt.text(b.get_x() + b.get_width() / 2, t + 1, f"{t:.0f}",
                 ha='center', fontweight='bold')
    plt.ylabel('Tokens por segundo (↑ melhor)')
    plt.title('Llama 3.2 1B — Velocidade por Quantização',
              fontsize=13, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✅ Gráfico de throughput por quantização gerado.")
