# GO2121 - StarCoder / Phi-3 Small -- Assistente de Codigo Local
# Casos: autocompletar codigo, explicar snippets, gerar testes
import time

print("=== Assistente de Codigo Local (StarCoder 1B / Phi-3 Small) ===")
funcionalidades = [
    "Autocompletar codigo",
    "Explicar snippets",
    "Gerar testes unitarios",
    "Latencia abaixo de 50ms",
]
for f in funcionalidades:
    print(f"  - {f}")

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
    for ex in ["def calcular_media(", "for i in range(", "import pandas"]:
        inicio = time.time()
        comp = simular_autocompletar(ex)
        lat = (time.time() - inicio) * 1000
        print(f"Input:  {ex}")
        print(f"Output: {comp}  ({lat:.1f}ms)")
