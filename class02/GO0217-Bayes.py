# GO0217-Bayes
from typing import Dict, List, Tuple, Set
from itertools import product
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. REDE BAYESIANA - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("REDES BAYESIANAS - DIAGNÓSTICO MÉDICO")
print("="*70)

class VariavelAleatoria:
    """
    Variável aleatória discreta
    """

    def __init__(self, nome: str, dominio: List[str]):
        """
        Args:
            nome: Nome da variável (ex: "Gripe")
            dominio: Valores possíveis (ex: ["sim", "nao"])
        """
        self.nome = nome
        self.dominio = dominio

    def __repr__(self):
        return f"Var({self.nome}, {self.dominio})"

class CPT:
    """
    Conditional Probability Table (Tabela de Probabilidade Condicional)

    P(X | Pais)
    """

    def __init__(self, variavel: VariavelAleatoria, 
                 pais: List[VariavelAleatoria] = None):
        """
        Args:
            variavel: Variável alvo
            pais: Variáveis pais (causas)
        """
        self.variavel = variavel
        self.pais = pais or []

        # Tabela: {(valores_pais): {valor_var: probabilidade}}
        self.tabela: Dict[Tuple, Dict[str, float]] = {}

    def definir_probabilidade(self, valores_pais: Tuple, 
                             distribuicao: Dict[str, float]):
        """
        Definir probabilidade condicional

        Args:
            valores_pais: Tupla de valores dos pais (ordem importa)
            distribuicao: {valor: probabilidade}

        Exemplo:
            cpt.definir_probabilidade(("sim",), {"sim": 0.8, "nao": 0.2})
        """
        # Validar que soma = 1.0
        soma = sum(distribuicao.values())
        if not np.isclose(soma, 1.0):
            raise ValueError(f"Probabilidades devem somar 1.0 (soma={soma})")

        self.tabela[valores_pais] = distribuicao

    def obter_probabilidade(self, valor_variavel: str, 
                           valores_pais: Tuple = ()) -> float:
        """Obter P(variavel=valor | pais=valores_pais)"""
        if valores_pais not in self.tabela:
            raise KeyError(f"Configuração de pais não encontrada: {valores_pais}")

        return self.tabela[valores_pais].get(valor_variavel, 0.0)

    def __repr__(self):
        pais_str = ", ".join([p.nome for p in self.pais])
        return f"CPT(P({self.variavel.nome} | {pais_str}))"

class RedeBayesiana:
    """
    Rede Bayesiana: Grafo acíclico direcionado + CPTs
    """

    def __init__(self):
        self.variaveis: Dict[str, VariavelAleatoria] = {}
        self.cpts: Dict[str, CPT] = {}
        self.dependencias: Dict[str, List[str]] = {}  # var -> pais

    def adicionar_variavel(self, var: VariavelAleatoria):
        """Adicionar variável à rede"""
        self.variaveis[var.nome] = var
        self.dependencias[var.nome] = []

    def adicionar_cpt(self, cpt: CPT):
        """Adicionar CPT (define dependências)"""
        self.cpts[cpt.variavel.nome] = cpt

        # Registrar dependências
        pais_nomes = [p.nome for p in cpt.pais]
        self.dependencias[cpt.variavel.nome] = pais_nomes

    def inferencia_por_enumeracao(self, consulta: str, 
                                  evidencias: Dict[str, str]) -> Dict[str, float]:
        """
        Inferência Bayesiana por enumeração completa

        Calcula P(consulta | evidencias)

        Args:
            consulta: Nome da variável de interesse
            evidencias: {variavel: valor_observado}

        Returns:
            Distribuição de probabilidade {valor: prob}
        """
        print(f"\n🔍 INFERÊNCIA: P({consulta} | {evidencias})")
        print("-"*70)

        var_consulta = self.variaveis[consulta]

        # Variáveis ocultas (não são consulta nem evidência)
        variaveis_ocultas = [v for v in self.variaveis.keys() 
                            if v != consulta and v not in evidencias]

        print(f"   Variáveis ocultas: {variaveis_ocultas}")

        # Para cada valor possível da consulta
        distribuicao = {}

        for valor_consulta in var_consulta.dominio:
            # Fixar valor da consulta
            atribuicao = {**evidencias, consulta: valor_consulta}

            # Somar sobre todas as atribuições das ocultas
            prob_total = 0.0

            if variaveis_ocultas:
                # Gerar todas as combinações de valores das ocultas
                dominios_ocultas = [self.variaveis[v].dominio for v in variaveis_ocultas]

                for valores_ocultas in product(*dominios_ocultas):
                    # Atribuição completa
                    atribuicao_completa = {**atribuicao}
                    for var_oculta, valor_oculto in zip(variaveis_ocultas, valores_ocultas):
                        atribuicao_completa[var_oculta] = valor_oculto

                    # Calcular P(atribuicao_completa)
                    prob = self._calcular_probabilidade_conjunta(atribuicao_completa)
                    prob_total += prob
            else:
                # Sem variáveis ocultas
                prob_total = self._calcular_probabilidade_conjunta(atribuicao)

            distribuicao[valor_consulta] = prob_total

        # Normalizar
        soma = sum(distribuicao.values())
        if soma > 0:
            distribuicao = {k: v/soma for k, v in distribuicao.items()}

        print(f"\n   Resultado:")
        for valor, prob in distribuicao.items():
            print(f"      P({consulta}={valor} | evidências) = {prob:.4f}")

        return distribuicao

    def _calcular_probabilidade_conjunta(self, atribuicao: Dict[str, str]) -> float:
        """
        Calcular P(atribuicao) = ∏ P(Xi | Pais(Xi))

        Args:
            atribuicao: {variavel: valor} para TODAS as variáveis
        """
        prob = 1.0

        for var_nome in self.variaveis.keys():
            cpt = self.cpts[var_nome]
            valor = atribuicao[var_nome]

            # Valores dos pais
            valores_pais = tuple(atribuicao[p] for p in self.dependencias[var_nome])

            # P(var=valor | pais)
            prob_condicional = cpt.obter_probabilidade(valor, valores_pais)
            prob *= prob_condicional

        return prob

    def resumo(self):
        """Resumo da rede"""
        return {
            "variaveis": len(self.variaveis),
            "cpts": len(self.cpts),
            "arestas": sum(len(pais) for pais in self.dependencias.values())
        }

# ═══════════════════════════════════════════════════════════════════
# 2. CONSTRUIR REDE BAYESIANA - DIAGNÓSTICO MÉDICO
# ═══════════════════════════════════════════════════════════════════

print("\n🏥 CONSTRUINDO REDE BAYESIANA...")
print("="*70)

rede_bayesiana = RedeBayesiana()

# Variáveis
gripe = VariavelAleatoria("Gripe", ["sim", "nao"])
covid = VariavelAleatoria("COVID", ["sim", "nao"])
febre = VariavelAleatoria("Febre", ["sim", "nao"])
tosse = VariavelAleatoria("Tosse", ["sim", "nao"])

rede_bayesiana.adicionar_variavel(gripe)
rede_bayesiana.adicionar_variavel(covid)
rede_bayesiana.adicionar_variavel(febre)
rede_bayesiana.adicionar_variavel(tosse)

print("\n✅ 4 variáveis adicionadas:")
print("   • Gripe (doença)")
print("   • COVID (doença)")
print("   • Febre (sintoma)")
print("   • Tosse (sintoma)")

# CPTs

# P(Gripe) - Prior
cpt_gripe = CPT(gripe, pais=[])
cpt_gripe.definir_probabilidade((), {"sim": 0.1, "nao": 0.9})
rede_bayesiana.adicionar_cpt(cpt_gripe)

print("\n✅ CPT 1: P(Gripe)")
print("   P(Gripe=sim) = 0.1")
print("   P(Gripe=nao) = 0.9")

# P(COVID) - Prior
cpt_covid = CPT(covid, pais=[])
cpt_covid.definir_probabilidade((), {"sim": 0.05, "nao": 0.95})
rede_bayesiana.adicionar_cpt(cpt_covid)

print("\n✅ CPT 2: P(COVID)")
print("   P(COVID=sim) = 0.05")
print("   P(COVID=nao) = 0.95")

# P(Febre | Gripe, COVID) - Dependência de ambas doenças
cpt_febre = CPT(febre, pais=[gripe, covid])

# Gripe=sim, COVID=sim
cpt_febre.definir_probabilidade(("sim", "sim"), {"sim": 0.95, "nao": 0.05})

# Gripe=sim, COVID=nao
cpt_febre.definir_probabilidade(("sim", "nao"), {"sim": 0.8, "nao": 0.2})

# Gripe=nao, COVID=sim
cpt_febre.definir_probabilidade(("nao", "sim"), {"sim": 0.85, "nao": 0.15})

# Gripe=nao, COVID=nao
cpt_febre.definir_probabilidade(("nao", "nao"), {"sim": 0.05, "nao": 0.95})

rede_bayesiana.adicionar_cpt(cpt_febre)

print("\n✅ CPT 3: P(Febre | Gripe, COVID)")
print("   P(Febre=sim | Gripe=sim, COVID=sim) = 0.95")
print("   P(Febre=sim | Gripe=sim, COVID=nao) = 0.8")
print("   P(Febre=sim | Gripe=nao, COVID=sim) = 0.85")
print("   P(Febre=sim | Gripe=nao, COVID=nao) = 0.05")

# P(Tosse | Gripe, COVID)
cpt_tosse = CPT(tosse, pais=[gripe, covid])

# Gripe=sim, COVID=sim
cpt_tosse.definir_probabilidade(("sim", "sim"), {"sim": 0.9, "nao": 0.1})

# Gripe=sim, COVID=nao
cpt_tosse.definir_probabilidade(("sim", "nao"), {"sim": 0.7, "nao": 0.3})

# Gripe=nao, COVID=sim
cpt_tosse.definir_probabilidade(("nao", "sim"), {"sim": 0.75, "nao": 0.25})

# Gripe=nao, COVID=nao
cpt_tosse.definir_probabilidade(("nao", "nao"), {"sim": 0.1, "nao": 0.9})

rede_bayesiana.adicionar_cpt(cpt_tosse)

print("\n✅ CPT 4: P(Tosse | Gripe, COVID)")
print("   P(Tosse=sim | Gripe=sim, COVID=sim) = 0.9")
print("   P(Tosse=sim | Gripe=sim, COVID=nao) = 0.7")
print("   P(Tosse=sim | Gripe=nao, COVID=sim) = 0.75")
print("   P(Tosse=sim | Gripe=nao, COVID=nao) = 0.1")

resumo = rede_bayesiana.resumo()
print(f"\n📊 REDE CONSTRUÍDA:")
print(f"   Variáveis: {resumo['variaveis']}")
print(f"   CPTs: {resumo['cpts']}")
print(f"   Dependências: {resumo['arestas']}")

# ═══════════════════════════════════════════════════════════════════
# 3. INFERÊNCIAS (DIAGNÓSTICO)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("DIAGNÓSTICO MÉDICO COM INFERÊNCIA BAYESIANA")
print("="*70)

# Cenário 1: Paciente com febre e tosse
print("\n🩺 CENÁRIO 1: Paciente com febre e tosse")
evidencias1 = {"Febre": "sim", "Tosse": "sim"}

dist_gripe1 = rede_bayesiana.inferencia_por_enumeracao("Gripe", evidencias1)
dist_covid1 = rede_bayesiana.inferencia_por_enumeracao("COVID", evidencias1)

print(f"\n💊 DIAGNÓSTICO:")
print(f"   Probabilidade de Gripe: {dist_gripe1['sim']*100:.1f}%")
print(f"   Probabilidade de COVID: {dist_covid1['sim']*100:.1f}%")

# Cenário 2: Paciente apenas com febre
print("\n" + "="*70)
print("\n🩺 CENÁRIO 2: Paciente apenas com febre (sem tosse)")
evidencias2 = {"Febre": "sim", "Tosse": "nao"}

dist_gripe2 = rede_bayesiana.inferencia_por_enumeracao("Gripe", evidencias2)
dist_covid2 = rede_bayesiana.inferencia_por_enumeracao("COVID", evidencias2)

print(f"\n💊 DIAGNÓSTICO:")
print(f"   Probabilidade de Gripe: {dist_gripe2['sim']*100:.1f}%")
print(f"   Probabilidade de COVID: {dist_covid2['sim']*100:.1f}%")

# Cenário 3: Nenhum sintoma
print("\n" + "="*70)
print("\n🩺 CENÁRIO 3: Paciente sem sintomas")
evidencias3 = {"Febre": "nao", "Tosse": "nao"}

dist_gripe3 = rede_bayesiana.inferencia_por_enumeracao("Gripe", evidencias3)
dist_covid3 = rede_bayesiana.inferencia_por_enumeracao("COVID", evidencias3)

print(f"\n💊 DIAGNÓSTICO:")
print(f"   Probabilidade de Gripe: {dist_gripe3['sim']*100:.1f}%")
print(f"   Probabilidade de COVID: {dist_covid3['sim']*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# 4. COMPARAÇÃO: COM vs SEM SINTOMAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPARAÇÃO: IMPACTO DOS SINTOMAS")
print("="*70)

cenarios = [
    ("Febre + Tosse", evidencias1, dist_gripe1['sim'], dist_covid1['sim']),
    ("Apenas Febre", evidencias2, dist_gripe2['sim'], dist_covid2['sim']),
    ("Sem Sintomas", evidencias3, dist_gripe3['sim'], dist_covid3['sim']),
]

print(f"\n{'Cenário':<20} {'P(Gripe)':<15} {'P(COVID)':<15}")
print("-"*50)
for nome, _, prob_gripe, prob_covid in cenarios:
    print(f"{nome:<20} {prob_gripe*100:>6.1f}%        {prob_covid*100:>6.1f}%")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÃO...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Estrutura da Rede Bayesiana
ax1 = axes[0]

G_estrutura = nx.DiGraph()
G_estrutura.add_edges_from([
    ("Gripe", "Febre"),
    ("Gripe", "Tosse"),
    ("COVID", "Febre"),
    ("COVID", "Tosse"),
])

pos_estrutura = {
    "Gripe": (0, 1),
    "COVID": (2, 1),
    "Febre": (0.5, 0),
    "Tosse": (1.5, 0)
}

cores_nos = ['lightcoral', 'lightcoral', 'lightblue', 'lightblue']

nx.draw_networkx_nodes(G_estrutura, pos_estrutura, node_color=cores_nos, 
                       node_size=3000, ax=ax1)
nx.draw_networkx_labels(G_estrutura, pos_estrutura, font_size=12, 
                       font_weight='bold', ax=ax1)
nx.draw_networkx_edges(G_estrutura, pos_estrutura, edge_color='gray', 
                       arrows=True, arrowsize=20, width=2, ax=ax1)

ax1.set_title("Estrutura da Rede Bayesiana", fontsize=14, fontweight='bold')
ax1.axis('off')

# Gráfico 2: Probabilidades por cenário
ax2 = axes[1]

cenarios_nomes = ["Febre+Tosse", "Apenas Febre", "Sem Sintomas"]
probs_gripe = [c[2]*100 for c in cenarios]
probs_covid = [c[3]*100 for c in cenarios]

x = np.arange(len(cenarios_nomes))
width = 0.35

bars1 = ax2.bar(x - width/2, probs_gripe, width, label='Gripe', 
               color='lightcoral', edgecolor='black')
bars2 = ax2.bar(x + width/2, probs_covid, width, label='COVID', 
               color='lightblue', edgecolor='black')

ax2.set_xlabel('Cenário', fontweight='bold')
ax2.set_ylabel('Probabilidade (%)', fontweight='bold')
ax2.set_title('Probabilidades de Diagnóstico por Cenário', 
             fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(cenarios_nomes)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Adicionar valores
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - REDES BAYESIANAS")
print("="*70)

print(f"\n🏗️ COMPONENTES:")
print(f"   • Variáveis aleatórias: {resumo['variaveis']}")
print(f"   • CPTs (Tabelas de Prob. Condicional): {resumo['cpts']}")
print(f"   • Dependências (arestas): {resumo['arestas']}")

print(f"\n🧠 INFERÊNCIA:")
print(f"   • Método: Enumeração completa")
print(f"   • Alternativas: Variable Elimination, Belief Propagation")
print(f"   • Complexidade: Exponencial (O(2^n)), mas explorável com estrutura")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Diagnóstico médico (sintomas → doenças)")
print(f"   • Detecção de fraude (padrões → anomalia)")
print(f"   • Sistemas de recomendação (preferências → produtos)")
print(f"   • Filtros de spam (palavras → spam/ham)")
print(f"   • Previsão de falhas (sensores → quebra)")

print(f"\n💡 VANTAGENS:")
print(f"   ✅ INCERTEZA: Lida naturalmente com probabilidades")
print(f"   ✅ CAUSAL: Modela relações causa-efeito")
print(f"   ✅ EXPLICÁVEL: Estrutura do grafo é interpretável")
print(f"   ✅ APRENDÍVEL: CPTs podem ser aprendidas de dados")

print(f"\n🔬 TEOREMA DE BAYES:")
print(f"   P(Doença | Sintomas) = P(Sintomas | Doença) * P(Doença) / P(Sintomas)")
print(f"   • P(Doença): Prior (prevalência)")
print(f"   • P(Sintomas | Doença): Likelihood (evidência)")
print(f"   • P(Doença | Sintomas): Posterior (diagnóstico)")

print("\n✅ REDE BAYESIANA COMPLETA!")
