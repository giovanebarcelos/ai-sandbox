# GO2123 - Mistral 7B + RAG -- Documentos Sensíveis On-Premise
# ═══════════════════════════════════════════════════════════════════
# Casos: docs médicos/jurídicos, on-premise, GDPR/LGPD compliant.
# RAG = indexar documentos + recuperar os mais relevantes para a query.
# O gráfico mostra o "score" de relevância de cada documento na consulta.
# ═══════════════════════════════════════════════════════════════════
import hashlib, time

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

print("=== Mistral 7B + RAG On-Premise ===")
casos = [
    "Documentos sensiveis (medico, juridico)",
    "On-premise",
    "GDPR/LGPD compliant",
]
for c in casos:
    print(f"  - {c}")

documentos = [
    "Laudo medico: paciente com hipertensao grau 2.",
    "Contrato: servico de consultoria por 12 meses.",
    "Politica de privacidade: dados nao compartilhados.",
]


# ─── PONTO-CHAVE: indexação local (hash) -> dados nunca saem do servidor ───
def indexar_documentos(docs):
    return {hashlib.md5(d.encode()).hexdigest()[:8]: d for d in docs}


# ─── PONTO-CHAVE: score = nº de palavras da query encontradas no doc ───
def score_relevancia(query, doc):
    palavras = query.lower().split()
    return sum(1 for w in palavras if w in doc.lower())


def simular_rag_query(query, index, top_k=2):
    ranqueados = sorted(
        ((k, v, score_relevancia(query, v)) for k, v in index.items()),
        key=lambda t: t[2], reverse=True,
    )
    return [(k, v, s) for k, v, s in ranqueados if s > 0][:top_k]


if __name__ == "__main__":
    idx = indexar_documentos(documentos)
    print(f"{len(idx)} documentos indexados localmente.")

    consulta = "contrato privacidade dados"
    inicio = time.time()
    # Score de TODOS os documentos para visualizar o ranking
    scores = [(v[:35], score_relevancia(consulta, v)) for v in documentos]
    res = simular_rag_query(consulta, idx)
    lat = (time.time() - inicio) * 1000

    print(f"\nConsulta: {consulta!r}  ({lat:.1f}ms)")
    for k, v, s in res:
        print(f"  [{k}] (score={s}) {v[:60]}")

    # ─── GRÁFICO: relevância de cada documento para a consulta ───
    nomes = [n for n, _ in scores]
    valores = [s for _, s in scores]
    cores = ['#2ca02c' if s > 0 else '#cccccc' for s in valores]

    plt.figure(figsize=(10, 5))
    plt.barh(nomes, valores, color=cores)
    plt.xlabel('Score de relevância (palavras da query encontradas)')
    plt.title(f'RAG On-Premise — Consulta: "{consulta}"',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ Gráfico de relevância (RAG) gerado.")
