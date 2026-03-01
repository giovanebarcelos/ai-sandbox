# GO0215-STRIPSPlanning
from typing import Set, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass
from collections import deque

# ═══════════════════════════════════════════════════════════════════
# 1. STRIPS - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("STRIPS PLANNING - ROBÔ DE LOGÍSTICA")
print("="*70)

@dataclass(frozen=True)
class Predicado:
    """
    Predicado: Representação de fato no mundo

    Exemplos:
    - at(robo, sala1)
    - holding(robo, caixa_a)
    - on(caixa_a, mesa)
    """
    nome: str
    argumentos: Tuple[str, ...]

    def __str__(self):
        args_str = ", ".join(self.argumentos)
        return f"{self.nome}({args_str})"

    def __repr__(self):
        return self.__str__()

class Estado:
    """
    Estado: Conjunto de predicados verdadeiros
    """

    def __init__(self, predicados: Set[Predicado] = None):
        self.predicados: FrozenSet[Predicado] = frozenset(predicados or set())

    def satisfaz(self, condicoes: Set[Predicado]) -> bool:
        """Verifica se estado satisfaz conjunto de condições"""
        return condicoes.issubset(self.predicados)

    def aplicar_efeitos(self, add_list: Set[Predicado], 
                       delete_list: Set[Predicado]) -> 'Estado':
        """Aplicar efeitos de uma ação"""
        novos_pred = set(self.predicados)
        novos_pred -= delete_list  # Remover
        novos_pred |= add_list     # Adicionar
        return Estado(novos_pred)

    def __hash__(self):
        return hash(self.predicados)

    def __eq__(self, other):
        return self.predicados == other.predicados

    def __str__(self):
        return "{" + ", ".join(str(p) for p in sorted(self.predicados, 
                                                      key=str)) + "}"

@dataclass
class Acao:
    """
    Ação STRIPS

    Componentes:
    - nome: Identificador da ação
    - parametros: Variáveis (ex: ?x, ?y)
    - preconditions: Predicados que devem ser verdadeiros
    - add_effects: Predicados adicionados ao estado
    - delete_effects: Predicados removidos do estado
    """
    nome: str
    parametros: Tuple[str, ...]
    preconditions: Set[Predicado]
    add_effects: Set[Predicado]
    delete_effects: Set[Predicado]

    def __str__(self):
        params_str = ", ".join(self.parametros)
        return f"{self.nome}({params_str})"

    def aplicavel(self, estado: Estado) -> bool:
        """Verifica se ação é aplicável no estado"""
        return estado.satisfaz(self.preconditions)

    def aplicar(self, estado: Estado) -> Estado:
        """Aplicar ação ao estado"""
        if not self.aplicavel(estado):
            raise ValueError(f"Ação {self} não aplicável em {estado}")

        return estado.aplicar_efeitos(self.add_effects, self.delete_effects)

# ═══════════════════════════════════════════════════════════════════
# 2. DOMÍNIO DO PROBLEMA: LOGÍSTICA
# ═══════════════════════════════════════════════════════════════════

print("\n🏗️ DEFININDO DOMÍNIO: Robô de Logística")
print("="*70)

# Objetos
robo = "robo"
salas = ["sala1", "sala2", "sala3"]
caixas = ["caixa_a", "caixa_b"]

print(f"\n📦 OBJETOS:")
print(f"   Robô: {robo}")
print(f"   Salas: {salas}")
print(f"   Caixas: {caixas}")

# Ações possíveis

# 1. MOVER: robo vai de uma sala para outra
def criar_acao_mover(sala_origem: str, sala_destino: str) -> Acao:
    """Ação: mover robô entre salas"""
    return Acao(
        nome="mover",
        parametros=(robo, sala_origem, sala_destino),
        preconditions={
            Predicado("at", (robo, sala_origem)),
        },
        add_effects={
            Predicado("at", (robo, sala_destino)),
        },
        delete_effects={
            Predicado("at", (robo, sala_origem)),
        }
    )

# 2. PEGAR: robo pega caixa (mãos livres)
def criar_acao_pegar(caixa: str, sala: str) -> Acao:
    """Ação: pegar caixa"""
    return Acao(
        nome="pegar",
        parametros=(robo, caixa, sala),
        preconditions={
            Predicado("at", (robo, sala)),
            Predicado("at", (caixa, sala)),
            Predicado("livre", (robo,)),
        },
        add_effects={
            Predicado("segurando", (robo, caixa)),
        },
        delete_effects={
            Predicado("at", (caixa, sala)),
            Predicado("livre", (robo,)),
        }
    )

# 3. SOLTAR: robo solta caixa
def criar_acao_soltar(caixa: str, sala: str) -> Acao:
    """Ação: soltar caixa"""
    return Acao(
        nome="soltar",
        parametros=(robo, caixa, sala),
        preconditions={
            Predicado("at", (robo, sala)),
            Predicado("segurando", (robo, caixa)),
        },
        add_effects={
            Predicado("at", (caixa, sala)),
            Predicado("livre", (robo,)),
        },
        delete_effects={
            Predicado("segurando", (robo, caixa)),
        }
    )

# Gerar todas as ações possíveis (grounding)
acoes: List[Acao] = []

# Mover entre todas as salas
for s1 in salas:
    for s2 in salas:
        if s1 != s2:
            acoes.append(criar_acao_mover(s1, s2))

# Pegar/soltar cada caixa em cada sala
for caixa in caixas:
    for sala in salas:
        acoes.append(criar_acao_pegar(caixa, sala))
        acoes.append(criar_acao_soltar(caixa, sala))

print(f"\n⚙️ AÇÕES POSSÍVEIS: {len(acoes)}")
print(f"   Mover: {len([a for a in acoes if a.nome == 'mover'])}")
print(f"   Pegar: {len([a for a in acoes if a.nome == 'pegar'])}")
print(f"   Soltar: {len([a for a in acoes if a.nome == 'soltar'])}")

# ═══════════════════════════════════════════════════════════════════
# 3. PLANEJADOR (BUSCA EM LARGURA)
# ═══════════════════════════════════════════════════════════════════

class PlanejadorSTRIPS:
    """
    Planejador STRIPS usando busca em largura progressiva
    """

    def __init__(self, estado_inicial: Estado, objetivo: Set[Predicado], 
                 acoes: List[Acao]):
        self.estado_inicial = estado_inicial
        self.objetivo = objetivo
        self.acoes = acoes
        self.estatisticas = {
            "estados_explorados": 0,
            "acoes_testadas": 0,
        }

    def planejar(self) -> Optional[List[Acao]]:
        """
        Busca em largura progressiva

        Returns:
            Lista de ações (plano) ou None se não encontrar
        """
        print("\n🔍 INICIANDO PLANEJAMENTO (Busca em Largura)...")

        # Fila: (estado_atual, caminho_de_acoes)
        fila = deque([(self.estado_inicial, [])])
        visitados: Set[Estado] = {self.estado_inicial}

        while fila:
            estado_atual, caminho = fila.popleft()
            self.estatisticas["estados_explorados"] += 1

            # Verificar se objetivo foi atingido
            if estado_atual.satisfaz(self.objetivo):
                print(f"\n✅ OBJETIVO ATINGIDO!")
                return caminho

            # Explorar ações aplicáveis
            for acao in self.acoes:
                self.estatisticas["acoes_testadas"] += 1

                if acao.aplicavel(estado_atual):
                    novo_estado = acao.aplicar(estado_atual)

                    if novo_estado not in visitados:
                        visitados.add(novo_estado)
                        novo_caminho = caminho + [acao]
                        fila.append((novo_estado, novo_caminho))

        print(f"\n❌ NENHUM PLANO ENCONTRADO!")
        return None

# ═══════════════════════════════════════════════════════════════════
# 4. DEFINIR PROBLEMA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PROBLEMA: Mover caixas entre salas")
print("="*70)

# Estado inicial
estado_inicial = Estado({
    Predicado("at", (robo, "sala1")),
    Predicado("at", ("caixa_a", "sala1")),
    Predicado("at", ("caixa_b", "sala2")),
    Predicado("livre", (robo,)),
})

print(f"\n📍 ESTADO INICIAL:")
for pred in sorted(estado_inicial.predicados, key=str):
    print(f"   • {pred}")

# Objetivo
objetivo = {
    Predicado("at", ("caixa_a", "sala3")),
    Predicado("at", ("caixa_b", "sala3")),
}

print(f"\n🎯 OBJETIVO:")
for pred in sorted(objetivo, key=str):
    print(f"   • {pred}")

# ═══════════════════════════════════════════════════════════════════
# 5. RESOLVER
# ═══════════════════════════════════════════════════════════════════

planejador = PlanejadorSTRIPS(estado_inicial, objetivo, acoes)

plano = planejador.planejar()

# ═══════════════════════════════════════════════════════════════════
# 6. EXIBIR PLANO
# ═══════════════════════════════════════════════════════════════════

if plano:
    print(f"\n📋 PLANO ENCONTRADO ({len(plano)} ações):")
    print("="*70)

    estado_atual = estado_inicial

    for i, acao in enumerate(plano, 1):
        print(f"\n   PASSO {i}: {acao}")
        print(f"      Precondições:")
        for prec in sorted(acao.preconditions, key=str):
            print(f"         ✓ {prec}")

        # Aplicar ação
        estado_atual = acao.aplicar(estado_atual)

        print(f"      Efeitos:")
        if acao.add_effects:
            print(f"         Adiciona:")
            for add in sorted(acao.add_effects, key=str):
                print(f"            + {add}")
        if acao.delete_effects:
            print(f"         Remove:")
            for delete in sorted(acao.delete_effects, key=str):
                print(f"            - {delete}")

    # Verificar se objetivo foi atingido
    print(f"\n🎯 ESTADO FINAL:")
    for pred in sorted(estado_atual.predicados, key=str):
        simbolo = "✅" if pred in objetivo else "  "
        print(f"   {simbolo} {pred}")

    objetivo_atingido = estado_atual.satisfaz(objetivo)
    print(f"\n{'✅ OBJETIVO ATINGIDO!' if objetivo_atingido else '❌ OBJETIVO NÃO ATINGIDO!'}")

print(f"\n📊 ESTATÍSTICAS:")
print(f"   Estados explorados: {planejador.estatisticas['estados_explorados']}")
print(f"   Ações testadas: {planejador.estatisticas['acoes_testadas']}")

# ═══════════════════════════════════════════════════════════════════
# 7. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - STRIPS PLANNING")
print("="*70)

print(f"\n🏗️ COMPONENTES STRIPS:")
print(f"   • Predicados: Fatos sobre o mundo")
print(f"   • Estado: Conjunto de predicados verdadeiros")
print(f"   • Ações: Precondições + Efeitos (add/delete)")
print(f"   • Plano: Sequência de ações")

print(f"\n🧠 ALGORITMO:")
print(f"   • Busca progressiva (forward search)")
print(f"   • BFS (largura) garante plano mais curto")
print(f"   • Alternativa: Busca regressiva (backward)")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Robótica (navegação, manipulação)")
print(f"   • Logística (transporte, armazém)")
print(f"   • Jogos (NPCs, puzzles)")
print(f"   • Automação industrial")
print(f"   • Planejamento de missões (NASA)")

print(f"\n💡 STRIPS vs OUTROS:")
print(f"   • STRIPS: Clássico, assumiões fortes (mundo determinístico)")
print(f"   • PDDL: Extensão moderna do STRIPS")
print(f"   • HTN: Hierarchical Task Networks (decomposição)")
print(f"   • RL: Reinforcement Learning (aprende, não planeja)")

print(f"\n🔬 LIMITAÇÕES DO STRIPS:")
print(f"   • Mundo determinístico (sem incerteza)")
print(f"   • Ações instantâneas (sem duração)")
print(f"   • Estado completamente observável")
print(f"   • Sem concorrência (ações sequenciais)")

print("\n✅ PLANEJADOR STRIPS COMPLETO!")
