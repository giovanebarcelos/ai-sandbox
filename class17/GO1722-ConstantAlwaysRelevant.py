"""
GO1722 - Memória Conversacional em RAG (Buffer e Summary)
=========================================================
Demonstra dois tipos de memória para chatbots: Buffer e Summary.
Requer apenas bibliotecas padrão (simulação sem LangChain externo).

Conceito: LLMs são stateless — cada chamada é independente.
Para "lembrar" conversas anteriores, precisamos de memória explícita.
Dois padrões principais:
- ConversationBufferMemory: guarda TODAS as mensagens (simples, mas cresce)
- ConversationSummaryMemory: resume mensagens antigas (escala melhor)

Custo vs. Memória:
    Buffer: O(n) tokens por turn — cresce linearmente
    Summary: O(1) tokens por turn — custo constante após compressão
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Mensagem:
    role: str   # "user" ou "assistant"
    content: str


# ──────────────────────────────────────────────────────────────
# 1. BUFFER MEMORY — guarda tudo
# ──────────────────────────────────────────────────────────────

class ConversationBufferMemory:
    """
    Mantém o histórico completo da conversa.
    Equivalente ao ConversationBufferMemory do LangChain.

    Vantagem: perfeito para conversas curtas.
    Desvantagem: context window do LLM se esgota em conversas longas.
    """

    def __init__(self):
        self.historico: List[Mensagem] = []

    def salvar(self, role: str, content: str) -> None:
        self.historico.append(Mensagem(role=role, content=content))

    def carregar(self) -> str:
        """Retorna o histórico como string para incluir no prompt."""
        if not self.historico:
            return ""
        linhas = [f"{m.role.capitalize()}: {m.content}" for m in self.historico]
        return "Histórico da conversa:\n" + "\n".join(linhas)

    def contar_tokens_estimado(self) -> int:
        """Estimativa: ~4 chars por token."""
        total = sum(len(m.content) for m in self.historico)
        return total // 4

    def __len__(self):
        return len(self.historico)


# ──────────────────────────────────────────────────────────────
# 2. SUMMARY MEMORY — resume para controlar tamanho
# ──────────────────────────────────────────────────────────────

class ConversationSummaryMemory:
    """
    Resume conversas antigas para manter o contexto compacto.
    Equivalente ao ConversationSummaryMemory do LangChain.

    Funciona assim:
    - Até N turnos: guarda tudo (igual ao Buffer)
    - Após N turnos: resume as mensagens antigas com LLM (aqui simulado)
    - Mantém resumo + últimas K mensagens completas

    Vantagem: contexto tem tamanho controlado.
    Desvantagem: perde detalhes das mensagens antigas.
    """

    def __init__(self, max_turnos: int = 4):
        self.max_turnos = max_turnos
        self.resumo: str = ""
        self.recentes: List[Mensagem] = []

    def _resumir_com_llm(self, mensagens: List[Mensagem]) -> str:
        """
        Em produção: chama o LLM para resumir.
        Aqui: simulação baseada nas primeiras palavras.
        """
        trechos = [f"{m.role}: {m.content[:40]}..." for m in mensagens]
        return f"[Resumo automático de {len(mensagens)} mensagens: {'; '.join(trechos[:2])}]"

    def salvar(self, role: str, content: str) -> None:
        self.recentes.append(Mensagem(role=role, content=content))

        # Quando ultrapassa o limite, comprimir as mais antigas
        if len(self.recentes) > self.max_turnos * 2:
            antigas = self.recentes[:self.max_turnos]
            novo_resumo = self._resumir_com_llm(antigas)
            if self.resumo:
                self.resumo += " | " + novo_resumo
            else:
                self.resumo = novo_resumo
            self.recentes = self.recentes[self.max_turnos:]

    def carregar(self) -> str:
        """Retorna resumo + mensagens recentes como contexto."""
        partes = []
        if self.resumo:
            partes.append(f"Resumo da conversa anterior:\n{self.resumo}")
        if self.recentes:
            linhas = [f"{m.role.capitalize()}: {m.content}" for m in self.recentes]
            partes.append("Mensagens recentes:\n" + "\n".join(linhas))
        return "\n\n".join(partes)

    def contar_tokens_estimado(self) -> int:
        total = len(self.resumo) + sum(len(m.content) for m in self.recentes)
        return total // 4


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1722 - MEMÓRIA CONVERSACIONAL")
    print("=" * 60)

    conversa = [
        ("user", "Olá! Me fale sobre inteligência artificial."),
        ("assistant", "IA é a simulação de inteligência humana por máquinas."),
        ("user", "Qual a diferença entre ML e deep learning?"),
        ("assistant", "ML inclui algoritmos clássicos; deep learning usa redes neurais profundas."),
        ("user", "Quando o deep learning surgiu?"),
        ("assistant", "O deep learning ganhou destaque em 2012 com AlexNet na ImageNet."),
        ("user", "Quais frameworks usar para deep learning?"),
        ("assistant", "TensorFlow, PyTorch e Keras são os mais populares."),
        ("user", "Pytorch é melhor que TensorFlow?"),
        ("assistant", "PyTorch é mais popular em pesquisa; TensorFlow em produção."),
    ]

    # ─── Buffer Memory ───────────────────────────────────────
    print("\n1. CONVERSATION BUFFER MEMORY")
    print("─" * 40)
    buffer = ConversationBufferMemory()

    for role, content in conversa[:4]:
        buffer.salvar(role, content)
        if role == "assistant":
            tokens = buffer.contar_tokens_estimado()
            print(f"  [{len(buffer):2d} msgs] ~{tokens} tokens estimados")

    print(f"\n  Contexto gerado para o LLM:")
    print(buffer.carregar())

    # ─── Summary Memory ──────────────────────────────────────
    print("\n\n2. CONVERSATION SUMMARY MEMORY (max 4 turnos)")
    print("─" * 40)
    summary_mem = ConversationSummaryMemory(max_turnos=4)

    print("  Adicionando mensagens (compressão ocorre ao exceder limite):")
    for i, (role, content) in enumerate(conversa):
        summary_mem.salvar(role, content)
        if role == "assistant":
            tokens = summary_mem.contar_tokens_estimado()
            recentes = len(summary_mem.recentes)
            tem_resumo = bool(summary_mem.resumo)
            print(f"  [turno {i//2 + 1}] ~{tokens} tokens | "
                  f"recentes={recentes} | resumo={'sim' if tem_resumo else 'nao'}")

    print(f"\n  Contexto comprimido para o LLM:")
    print(summary_mem.carregar())

    # ─── Comparação ─────────────────────────────────────────
    print("\n3. COMPARAÇÃO")
    print("─" * 40)
    buffer_full = ConversationBufferMemory()
    summary_full = ConversationSummaryMemory(max_turnos=4)
    for role, content in conversa:
        buffer_full.salvar(role, content)
        summary_full.salvar(role, content)

    print(f"  Buffer  : ~{buffer_full.contar_tokens_estimado():4d} tokens (cresce linearmente)")
    print(f"  Summary : ~{summary_full.contar_tokens_estimado():4d} tokens (cresce lentamente)")
    print()
    print("  Escolha:")
    print("  - Buffer   -> conversas curtas (< 20 turnos)")
    print("  - Summary  -> conversas longas, atendimento ao cliente")
    print("  - Hybrid   -> summary + buffer das últimas N mensagens (melhor dos dois)")
