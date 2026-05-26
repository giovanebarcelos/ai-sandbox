"""
GO1712 - Agentes com RAG (Retrieval-Augmented Generation)
==========================================================
Demonstra como um agente pode usar RAG como ferramenta de busca.
Requer apenas bibliotecas padrão (simulação sem LLM externo).

Conceito: um Agente é um LLM que decide QUANDO chamar ferramentas.
O RAG vira uma "Tool" que o agente aciona para recuperar contexto
antes de formular sua resposta. O fluxo:
  1. Usuário faz pergunta
  2. Agente decide: preciso buscar informação?
  3. Agente chama RAG Tool com query
  4. RAG retorna contexto relevante
  5. Agente gera resposta com base no contexto
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable


# ──────────────────────────────────────────────────────────────
# 1. RAG SIMPLES (base de conhecimento interna)
# ──────────────────────────────────────────────────────────────

@dataclass
class Doc:
    texto: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SistemaRAG:
    """
    Sistema RAG minimalista para demonstração.
    Usa similaridade por palavras-chave (TF-simples).
    """

    def __init__(self):
        self.docs: List[Doc] = []

    def ingerir(self, textos: List[str], metadatas: List[Dict] = None) -> None:
        if metadatas is None:
            metadatas = [{}] * len(textos)
        for t, m in zip(textos, metadatas):
            self.docs.append(Doc(texto=t, metadata=m))

    def buscar(self, query: str, k: int = 3) -> str:
        """Retorna os k trechos mais relevantes como string de contexto."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.docs:
            doc_words = set(doc.texto.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [doc.texto for _, doc in scored[:k] if _[0] > 0]
        if not top:
            return "Nenhuma informação encontrada na base."
        return "\n".join(f"- {t}" for t in top)

    def query(self, pergunta: str) -> str:
        contexto = self.buscar(pergunta, k=3)
        # Simulação de geração: combina contexto com pergunta
        return f"Com base nos documentos internos:\n{contexto}\n\n[Resposta gerada a partir do contexto acima]"


# ──────────────────────────────────────────────────────────────
# 2. FERRAMENTA (Tool) encapsulando o RAG
# ──────────────────────────────────────────────────────────────

@dataclass
class Ferramenta:
    nome: str
    descricao: str
    funcao: Callable[[str], str]

    def executar(self, argumento: str) -> str:
        return self.funcao(argumento)


# ──────────────────────────────────────────────────────────────
# 3. AGENTE SIMPLES
# ──────────────────────────────────────────────────────────────

class AgenteSimples:
    """
    Agente que usa heurística para decidir quando usar ferramentas.
    Em produção, o LLM faz essa decisão via function-calling.
    """

    def __init__(self, ferramentas: List[Ferramenta]):
        self.ferramentas = {f.nome: f for f in ferramentas}

    def _decidir_ferramenta(self, pergunta: str) -> str | None:
        """
        Heurística: se a pergunta parece pedir informação factual,
        usar a ferramenta RAG.
        """
        gatilhos_busca = [
            "qual", "o que é", "como", "quando", "política",
            "regras", "benefício", "retorno", "devolução", "prazo"
        ]
        pergunta_lower = pergunta.lower()
        if any(g in pergunta_lower for g in gatilhos_busca):
            return "RAG"
        return None

    def executar(self, pergunta: str, verbose: bool = True) -> str:
        if verbose:
            print(f"\n[Agente] Processando: '{pergunta}'")

        ferramenta_nome = self._decidir_ferramenta(pergunta)

        if ferramenta_nome and ferramenta_nome in self.ferramentas:
            ferramenta = self.ferramentas[ferramenta_nome]
            if verbose:
                print(f"[Agente] -> Usando ferramenta: {ferramenta.nome}")
                print(f"           Descricao: {ferramenta.descricao}")
            resultado = ferramenta.executar(pergunta)
            if verbose:
                print(f"[Agente] -> Contexto recuperado:")
                for linha in resultado.split("\n")[:4]:
                    print(f"           {linha}")
            return resultado
        else:
            resposta = f"Respondendo diretamente: '{pergunta}' -> [resposta do LLM sem RAG]"
            if verbose:
                print(f"[Agente] -> Respondendo sem ferramenta.")
            return resposta


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1712 - AGENTES COM RAG")
    print("=" * 60)

    # Montar base de conhecimento da empresa
    rag = SistemaRAG()
    rag.ingerir(
        textos=[
            "Nossa política de retorno permite devolução em até 30 dias.",
            "Produtos com defeito têm garantia de 12 meses.",
            "O prazo de entrega padrão é de 5 a 10 dias úteis.",
            "Frete grátis para compras acima de R$ 150.",
            "Pagamento parcelado em até 12x sem juros no cartão.",
            "Atendimento disponível de segunda a sexta, das 8h às 18h.",
        ],
        metadatas=[{"source": "faq.pdf"}] * 6,
    )

    # Criar ferramenta RAG
    rag_tool = Ferramenta(
        nome="RAG",
        funcao=lambda q: rag.query(q),
        descricao="Busca informações nos documentos internos da empresa",
    )

    # Criar agente com a ferramenta
    agente = AgenteSimples(ferramentas=[rag_tool])
    agente.executar("Qual a política de retorno?", verbose=True)

    print()
    agente.executar("Como funciona o frete grátis?", verbose=True)

    print()
    agente.executar("Olá, tudo bem?", verbose=True)

    print()
    print("  Em producao:")
    print("  - LLM decide quais ferramentas chamar via function-calling")
    print("  - Agente pode encadear multiplas ferramentas (RAG + Calculadora + API)")
    print("  - Frameworks: LangChain, LlamaIndex, AutoGen, CrewAI")
