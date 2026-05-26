# GO2122 - Llama 3.2 1B quantizado -- Processamento Local em Tempo Real
# Casos: processamento local, decisoes em tempo real, sem cloud
import time, json

print("=== Llama 3.2 1B Quantizado ===")
vantagens = [
    "Processamento 100% local",
    "Decisoes em tempo real",
    "Sem dependencia de cloud",
]
for v in vantagens:
    print(f"  - {v}")

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
