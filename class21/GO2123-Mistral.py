# GO2123 - Mistral 7B + RAG -- Documentos Sensiveis On-Premise
# Casos: docs medicos/juridicos, on-premise, GDPR/LGPD compliant
import hashlib, time

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

def indexar_documentos(docs):
    return {hashlib.md5(d.encode()).hexdigest()[:8]: d for d in docs}

def simular_rag_query(query, index, top_k=2):
    return [(k, v) for k, v in index.items()
            if any(w in v.lower() for w in query.lower().split())][:top_k]

if __name__ == "__main__":
    idx = indexar_documentos(documentos)
    print(f"{len(idx)} documentos indexados localmente.")
    for consulta in ["medico", "contrato privacidade"]:
        inicio = time.time()
        res = simular_rag_query(consulta, idx)
        lat = (time.time() - inicio) * 1000
        print(f"Consulta: {consulta!r}  ({lat:.1f}ms)")
        for k, v in res:
            print(f"  [{k}] {v[:60]}")
