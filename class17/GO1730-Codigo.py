"""
GO1730 - Few-Shot Prompting para Extração de Informações
=========================================================
Demonstra Few-Shot prompting para tarefas de extração estruturada.
Requer apenas bibliotecas padrão (simulação sem LLM externo).

Conceito: Few-Shot prompting fornece exemplos de entrada/saída no prompt.
O LLM "aprende" o padrão via in-context learning e aplica ao novo input.
É especialmente eficaz para extrair entidades nomeadas (NER) de forma
estruturada sem fine-tuning.

Aplicações práticas:
- Extrair dados de currículos (nome, cargo, empresa)
- Processar contratos (partes, valores, datas)
- Estruturar registros médicos
"""

import re
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────
# 1. TEMPLATE DE PROMPT FEW-SHOT
# ──────────────────────────────────────────────────────────────

PROMPT_FEW_SHOT = """Extraia Nome, Cargo e Empresa do texto:

Texto: "João Silva é engenheiro na Microsoft"
Nome: João Silva
Cargo: Engenheiro
Empresa: Microsoft

Texto: "Maria Santos trabalha como designer na Apple"
Nome: Maria Santos
Cargo: Designer
Empresa: Apple

Texto: "Carlos Pereira atua como gerente de projetos no Google"
Nome: Carlos Pereira
Cargo: Gerente de Projetos
Empresa: Google

Agora extraia:
Texto: "{texto}"
"""


def construir_prompt(texto: str) -> str:
    """Monta o prompt few-shot com o novo texto para extração."""
    return PROMPT_FEW_SHOT.format(texto=texto)


# ──────────────────────────────────────────────────────────────
# 2. PARSER DE SAÍDA (simula o parsing da resposta do LLM)
# ──────────────────────────────────────────────────────────────

def parsear_resposta(resposta: str) -> Dict[str, str]:
    """
    Extrai os campos estruturados da resposta do LLM.
    LLM bem-comportado segue o padrão 'Campo: Valor'.
    """
    campos = {}
    for linha in resposta.strip().split("\n"):
        if ":" in linha:
            chave, _, valor = linha.partition(":")
            chave = chave.strip()
            valor = valor.strip()
            if chave in ("Nome", "Cargo", "Empresa"):
                campos[chave] = valor
    return campos


# ──────────────────────────────────────────────────────────────
# 3. EXTRATOR REGEX (baseline sem LLM para comparação)
# ──────────────────────────────────────────────────────────────

EMPRESAS_CONHECIDAS = {
    "microsoft", "apple", "google", "amazon", "meta",
    "netflix", "ibm", "oracle", "salesforce", "tesla",
}

CARGOS_CONHECIDOS = {
    "engenheiro", "designer", "gerente", "analista", "cientista",
    "desenvolvedor", "arquiteto", "coordenador", "diretor",
}


def extrator_regex(texto: str) -> Dict[str, Optional[str]]:
    """
    Extrator baseado em heurísticas (baseline sem LLM).
    Demonstra limitações de abordagens baseadas em regras.
    """
    resultado = {"Nome": None, "Cargo": None, "Empresa": None}
    texto_lower = texto.lower()

    # Empresa: busca por nome conhecido
    for empresa in EMPRESAS_CONHECIDAS:
        if empresa in texto_lower:
            resultado["Empresa"] = empresa.capitalize()
            break

    # Cargo: busca por título conhecido
    for cargo in CARGOS_CONHECIDOS:
        if cargo in texto_lower:
            resultado["Cargo"] = cargo.capitalize()
            break

    # Nome: heurística fraca (primeiras palavras maiúsculas)
    palavras = texto.split()
    nome_candidato = []
    for palavra in palavras[:4]:
        if palavra and palavra[0].isupper() and palavra.lower() not in ("é", "na", "no"):
            nome_candidato.append(palavra)
        elif nome_candidato:
            break
    resultado["Nome"] = " ".join(nome_candidato) if nome_candidato else None

    return resultado


# ──────────────────────────────────────────────────────────────
# 4. SIMULAÇÃO DA RESPOSTA DO LLM
# ──────────────────────────────────────────────────────────────

def simular_llm(texto: str) -> str:
    """
    Simula a resposta do LLM seguindo o padrão few-shot.
    Em produção: chamar Ollama ou OpenAI.
    Retorna string no formato 'Campo: Valor\\n...'
    """
    # Base de dados simulada para demonstração
    casos = {
        "ana costa": ("Ana Costa", "Cientista de Dados", "Amazon"),
        "pedro lima": ("Pedro Lima", "Arquiteto de Soluções", "IBM"),
        "fernanda": ("Fernanda Rocha", "Diretora de Tecnologia", "Salesforce"),
    }

    texto_lower = texto.lower()
    for chave, (nome, cargo, empresa) in casos.items():
        if chave in texto_lower:
            return f"Nome: {nome}\nCargo: {cargo}\nEmpresa: {empresa}"

    # Fallback: tentar extrair via regex
    dados = extrator_regex(texto)
    linhas = [f"{k}: {v}" for k, v in dados.items() if v]
    return "\n".join(linhas) if linhas else "Não foi possível extrair as informações."


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1730 - FEW-SHOT: EXTRAÇÃO DE INFORMAÇÕES")
    print("=" * 60)

    # Textos para extração
    casos_teste = [
        "Ana Costa é cientista de dados na Amazon",
        "Pedro Lima trabalha como arquiteto de soluções na IBM",
        "Fernanda Rocha é diretora de tecnologia na Salesforce",
        "Roberto Alves atua como desenvolvedor senior no Google",
    ]

    print("\nPROMPT FEW-SHOT (template):")
    print("─" * 60)
    print(construir_prompt("[texto aqui]"))

    print("─" * 60)
    print("RESULTADO DA EXTRAÇÃO:")
    print("─" * 60)

    for texto in casos_teste:
        print(f"\nTexto: \"{texto}\"")

        # Simular resposta do LLM
        resposta_llm = simular_llm(texto)
        dados_llm = parsear_resposta(resposta_llm)

        print("  LLM (Few-Shot):")
        for campo, valor in dados_llm.items():
            print(f"    {campo}: {valor}")

        # Baseline regex
        dados_regex = extrator_regex(texto)
        print("  Regex (baseline):")
        for campo, valor in dados_regex.items():
            print(f"    {campo}: {valor if valor else '---'}")

    print()
    print("─" * 60)
    print("POR QUE FEW-SHOT SUPERA REGEX?")
    print("─" * 60)
    print("  Regex: precisa listar TODAS as variações possíveis")
    print("  Few-Shot LLM: generaliza para qualquer variação linguística")
    print()
    print("  Exemplo: 'Ana lidera a equipe de dados na Amazon'")
    print("  Regex: não reconhece 'lidera a equipe de dados' como cargo")
    print("  LLM  : extrai Cargo='Lider de Dados' por compreensão semântica")
    print()
    print("  Para uso real: pip install ollama | python GO1733-SimulateTool.py")
