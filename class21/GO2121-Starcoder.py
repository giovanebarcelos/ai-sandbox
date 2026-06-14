# GO2121 - StarCoder / Phi-3 Small -- Assistente de Código Local
# ═══════════════════════════════════════════════════════════════════
# Casos: autocompletar código, explicar snippets, gerar testes.
# Medimos a latência de autocompletar de um modelo local e mostramos
# por que ela precisa ficar abaixo de ~50ms para ser "instantânea".
# ═══════════════════════════════════════════════════════════════════
import time

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

print("=== Assistente de Codigo Local (StarCoder 1B / Phi-3 Small) ===")
funcionalidades = [
    "Autocompletar codigo",
    "Explicar snippets",
    "Gerar testes unitarios",
    "Latencia abaixo de 50ms",
]
for f in funcionalidades:
    print(f"  - {f}")


# ─── PONTO-CHAVE: busca a melhor sugestão para o código parcial ───
def simular_autocompletar(codigo_parcial):
    sugestoes = {
        "def calcular_media(": "    return sum(valores) / len(valores)",
        "for i in range(": "    print(i)",
        "import pandas": "import pandas as pd",
    }
    for chave, comp in sugestoes.items():
        if codigo_parcial.startswith(chave):
            return comp
    return "    # continuacao gerada por LLM..."


if __name__ == "__main__":
    exemplos = ["def calcular_media(", "for i in range(", "import pandas"]

    latencias = []
    for ex in exemplos:
        inicio = time.time()
        comp = simular_autocompletar(ex)
        lat = (time.time() - inicio) * 1000
        latencias.append(lat)
        print(f"Input:  {ex}")
        print(f"Output: {comp}  ({lat:.1f}ms)")

    # ─── GRÁFICO: latência de cada autocompletar vs limite de 50ms ───
    # A linha vermelha marca o limite percebido como "instantâneo"
    plt.figure(figsize=(10, 5))
    barras = plt.bar([f"Sugestão {i+1}" for i in range(len(exemplos))],
                     latencias, color='#1f77b4')
    plt.axhline(50, color='red', linestyle='--', label='Limite "instantâneo" (50ms)')
    for b, lat in zip(barras, latencias):
        plt.text(b.get_x() + b.get_width() / 2, lat, f"{lat:.1f}ms",
                 ha='center', va='bottom', fontweight='bold')
    plt.ylabel('Latência (ms)')
    plt.title('Assistente de Código Local — Latência de Autocompletar',
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✅ Gráfico de latência do autocompletar gerado.")
