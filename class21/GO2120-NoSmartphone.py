# GO2120 - LLMs no Smartphone (Phi-3, Gemma-2B)
# Casos de uso: responder emails, resumir, traduzir -- offline, privado, gratuito
import time

print("=== LLM Local no Smartphone ===")
casos_de_uso = [
    "Responder emails",
    "Resumir documentos",
    "Traduzir textos",
    "Tudo offline, privado, gratuito",
]
for i, caso in enumerate(casos_de_uso, 1):
    print(f"  {i}. {caso}")

def simular_inferencia_local(prompt, tempo_ms=120):
    time.sleep(tempo_ms / 1000)
    return f"[Resposta local para: {prompt[:30]}...]"

if __name__ == "__main__":
    for p in ["Resuma este email em 2 linhas.", "Traduza para ingles: Bom dia!"]:
        inicio = time.time()
        resp = simular_inferencia_local(p, tempo_ms=80)
        lat = (time.time() - inicio) * 1000
        print(f"  {resp}  ({lat:.0f}ms)")
