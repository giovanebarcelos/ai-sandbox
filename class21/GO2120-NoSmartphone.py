# GO2120 - LLMs no Smartphone (Phi-3, Gemma-2B)
# ═══════════════════════════════════════════════════════════════════
# Casos de uso: responder emails, resumir, traduzir -- offline, privado,
# gratuito. Aqui simulamos a inferência local e comparamos a latência
# de um modelo no aparelho vs. uma chamada à nuvem.
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

print("=== LLM Local no Smartphone ===")
casos_de_uso = [
    "Responder emails",
    "Resumir documentos",
    "Traduzir textos",
    "Tudo offline, privado, gratuito",
]
for i, caso in enumerate(casos_de_uso, 1):
    print(f"  {i}. {caso}")


# ─── PONTO-CHAVE: inferência roda no próprio aparelho (sem rede) ───
def simular_inferencia_local(prompt, tempo_ms=120):
    time.sleep(tempo_ms / 1000)  # simula o tempo de geração no dispositivo
    return f"[Resposta local para: {prompt[:30]}...]"


if __name__ == "__main__":
    prompts = ["Resuma este email em 2 linhas.", "Traduza para ingles: Bom dia!"]

    # Mede a latência real de cada inferência local simulada
    latencias_local = []
    for p in prompts:
        inicio = time.time()
        resp = simular_inferencia_local(p, tempo_ms=80)
        lat = (time.time() - inicio) * 1000
        latencias_local.append(lat)
        print(f"  {resp}  ({lat:.0f}ms)")

    # ─── GRÁFICO: latência Local (no aparelho) vs Cloud (com rede) ───
    # A nuvem soma latência de rede + fila + inferência -> bem mais lenta
    latencias_cloud = [lat + 850 for lat in latencias_local]  # ~850ms de rede
    x = range(len(prompts))
    largura = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - largura / 2 for i in x], latencias_local, largura,
            label='Local (no smartphone)', color='#2ca02c')
    plt.bar([i + largura / 2 for i in x], latencias_cloud, largura,
            label='Cloud (API + rede)', color='#d62728')
    plt.xticks(list(x), [f"Tarefa {i+1}" for i in x])
    plt.ylabel('Latência (ms)')
    plt.title('LLM no Smartphone: Local vs Cloud', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✅ Gráfico de latência (local vs cloud) gerado.")
