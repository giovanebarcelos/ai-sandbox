# GO0214-ConstraintSatisfactionProblemsCSP
from typing import List, Dict, Set, Tuple, Optional, Callable
from itertools import combinations
import random

# ═══════════════════════════════════════════════════════════════════
# 1. CSP - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("CONSTRAINT SATISFACTION PROBLEM - AGENDAMENTO DE EXAMES")
print("="*70)

class CSP:
    """
    Constraint Satisfaction Problem Solver

    Componentes:
    - Variáveis: X = {X1, X2, ..., Xn}
    - Domínios: D = {D1, D2, ..., Dn}
    - Restrições: C = {C1, C2, ..., Cm}
    """

    def __init__(self):
        self.variaveis: List[str] = []
        self.dominios: Dict[str, List] = {}
        self.restricoes: List[Callable] = []
        self.vizinhos: Dict[str, Set[str]] = {}  # Grafo de restrições
        self.estatisticas = {
            "atribuicoes": 0,
            "backtracks": 0,
            "consistencia_checks": 0
        }

    def adicionar_variavel(self, var: str, dominio: List):
        """Adicionar variável com seu domínio"""
        self.variaveis.append(var)
        self.dominios[var] = dominio.copy()
        self.vizinhos[var] = set()

    def adicionar_restricao(self, restricao: Callable, variaveis_envolvidas: List[str]):
        """
        Adicionar restrição

        Args:
            restricao: Função que recebe atribuição e retorna True/False
            variaveis_envolvidas: Lista de variáveis na restrição
        """
        self.restricoes.append(restricao)

        # Atualizar grafo de vizinhos
        for v1, v2 in combinations(variaveis_envolvidas, 2):
            if v1 in self.vizinhos:
                self.vizinhos[v1].add(v2)
            if v2 in self.vizinhos:
                self.vizinhos[v2].add(v1)

    def consistente(self, atribuicao: Dict[str, any]) -> bool:
        """Verificar se atribuição satisfaz todas as restrições"""
        self.estatisticas["consistencia_checks"] += 1

        for restricao in self.restricoes:
            if not restricao(atribuicao):
                return False
        return True

    def selecionar_variavel_nao_atribuida(self, atribuicao: Dict) -> Optional[str]:
        """
        Heurística MRV (Minimum Remaining Values)

        Escolhe variável com menor domínio restante
        """
        nao_atribuidas = [v for v in self.variaveis if v not in atribuicao]

        if not nao_atribuidas:
            return None

        # MRV: Variável com menor número de valores possíveis
        return min(nao_atribuidas, key=lambda v: len(self.dominios[v]))

    def ordenar_valores_dominio(self, var: str, atribuicao: Dict) -> List:
        """
        Heurística LCV (Least Constraining Value)

        Ordena valores que menos restringem outras variáveis
        """
        # Simplificado: retorna domínio embaralhado
        valores = self.dominios[var].copy()
        random.shuffle(valores)
        return valores

    def backtracking_search(self, atribuicao: Dict = None) -> Optional[Dict]:
        """
        Busca com backtracking

        Algoritmo:
        1. Se atribuição completa → retornar
        2. Selecionar variável não atribuída
        3. Para cada valor no domínio:
           - Atribuir valor
           - Se consistente → recursão
           - Se sucesso → retornar
           - Senão → remover atribuição (backtrack)
        """
        if atribuicao is None:
            atribuicao = {}

        # Caso base: atribuição completa
        if len(atribuicao) == len(self.variaveis):
            return atribuicao

        # Selecionar variável (MRV)
        var = self.selecionar_variavel_nao_atribuida(atribuicao)

        if var is None:
            return atribuicao

        # Tentar valores (LCV)
        for valor in self.ordenar_valores_dominio(var, atribuicao):
            self.estatisticas["atribuicoes"] += 1

            # Atribuir
            atribuicao[var] = valor

            # Verificar consistência
            if self.consistente(atribuicao):
                # Recursão
                resultado = self.backtracking_search(atribuicao)

                if resultado is not None:
                    return resultado

            # Backtrack
            del atribuicao[var]
            self.estatisticas["backtracks"] += 1

        return None

    def resolver(self) -> Optional[Dict]:
        """Resolver CSP"""
        print("\n🔍 INICIANDO BUSCA COM BACKTRACKING...")
        print(f"   Variáveis: {len(self.variaveis)}")
        print(f"   Restrições: {len(self.restricoes)}")

        solucao = self.backtracking_search()

        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Atribuições tentadas: {self.estatisticas['atribuicoes']}")
        print(f"   Backtracks: {self.estatisticas['backtracks']}")
        print(f"   Verificações de consistência: {self.estatisticas['consistencia_checks']}")

        return solucao

# ═══════════════════════════════════════════════════════════════════
# 2. PROBLEMA: AGENDAMENTO DE EXAMES MÉDICOS
# ═══════════════════════════════════════════════════════════════════

print("\n🏥 PROBLEMA: Agendar 5 exames médicos")
print("="*70)

# Exames a agendar
exames = ["RaioX", "Sangue", "Ultrassom", "Ressonancia", "ECG"]

# Médicos disponíveis
medicos = ["Dr. Silva", "Dra. Santos", "Dr. Costa"]

# Horários disponíveis (slots de 30min)
horarios = ["08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00"]

# Salas disponíveis
salas = ["Sala_A", "Sala_B", "Sala_C"]

print(f"\n📋 DADOS:")
print(f"   Exames: {exames}")
print(f"   Médicos: {medicos}")
print(f"   Horários: {horarios}")
print(f"   Salas: {salas}")

# Criar CSP
csp = CSP()

# Adicionar variáveis (cada exame precisa: médico, horário, sala)
for exame in exames:
    # Variável: exame_medico
    csp.adicionar_variavel(f"{exame}_medico", medicos)

    # Variável: exame_horario
    csp.adicionar_variavel(f"{exame}_horario", horarios)

    # Variável: exame_sala
    csp.adicionar_variavel(f"{exame}_sala", salas)

print(f"\n✅ {len(csp.variaveis)} variáveis criadas")

# ═══════════════════════════════════════════════════════════════════
# 3. DEFINIR RESTRIÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n📋 DEFININDO RESTRIÇÕES...")

# Restrição 1: Médico não pode atender 2 exames no mesmo horário
def restricao_medico_horario(atrib: Dict) -> bool:
    """Médico só atende 1 exame por horário"""
    for e1, e2 in combinations(exames, 2):
        if f"{e1}_medico" in atrib and f"{e2}_medico" in atrib:
            if f"{e1}_horario" in atrib and f"{e2}_horario" in atrib:
                if (atrib[f"{e1}_medico"] == atrib[f"{e2}_medico"] and
                    atrib[f"{e1}_horario"] == atrib[f"{e2}_horario"]):
                    return False
    return True

csp.adicionar_restricao(
    restricao_medico_horario,
    [f"{e}_{t}" for e in exames for t in ["medico", "horario"]]
)

print("   ✅ R1: Médico não pode atender 2 exames no mesmo horário")

# Restrição 2: Sala não pode ter 2 exames no mesmo horário
def restricao_sala_horario(atrib: Dict) -> bool:
    """Sala só pode ter 1 exame por horário"""
    for e1, e2 in combinations(exames, 2):
        if f"{e1}_sala" in atrib and f"{e2}_sala" in atrib:
            if f"{e1}_horario" in atrib and f"{e2}_horario" in atrib:
                if (atrib[f"{e1}_sala"] == atrib[f"{e2}_sala"] and
                    atrib[f"{e1}_horario"] == atrib[f"{e2}_horario"]):
                    return False
    return True

csp.adicionar_restricao(
    restricao_sala_horario,
    [f"{e}_{t}" for e in exames for t in ["sala", "horario"]]
)

print("   ✅ R2: Sala não pode ter 2 exames no mesmo horário")

# Restrição 3: Ressonância só pode ser feita por Dr. Silva
def restricao_ressonancia_especialista(atrib: Dict) -> bool:
    """Ressonância requer especialista"""
    if "Ressonancia_medico" in atrib:
        return atrib["Ressonancia_medico"] == "Dr. Silva"
    return True

csp.adicionar_restricao(
    restricao_ressonancia_especialista,
    ["Ressonancia_medico"]
)

print("   ✅ R3: Ressonância só pode ser feita por Dr. Silva")

# Restrição 4: ECG deve ser antes de 10:00 (equipamento compartilhado)
def restricao_ecg_manha(atrib: Dict) -> bool:
    """ECG apenas pela manhã (antes 10:00)"""
    if "ECG_horario" in atrib:
        hora = atrib["ECG_horario"]
        return hora < "10:00"
    return True

csp.adicionar_restricao(
    restricao_ecg_manha,
    ["ECG_horario"]
)

print("   ✅ R4: ECG deve ser antes de 10:00")

# Restrição 5: Exame de Sangue deve ser em jejum (primeiro horário)
def restricao_sangue_jejum(atrib: Dict) -> bool:
    """Sangue deve ser primeiro (08:00)"""
    if "Sangue_horario" in atrib:
        return atrib["Sangue_horario"] == "08:00"
    return True

csp.adicionar_restricao(
    restricao_sangue_jejum,
    ["Sangue_horario"]
)

print("   ✅ R5: Exame de Sangue deve ser às 08:00 (jejum)")

# Restrição 6: Ultrassom precisa da Sala_A (equipamento específico)
def restricao_ultrassom_sala(atrib: Dict) -> bool:
    """Ultrassom requer Sala_A"""
    if "Ultrassom_sala" in atrib:
        return atrib["Ultrassom_sala"] == "Sala_A"
    return True

csp.adicionar_restricao(
    restricao_ultrassom_sala,
    ["Ultrassom_sala"]
)

print("   ✅ R6: Ultrassom precisa da Sala_A")

print(f"\n✅ {len(csp.restricoes)} restrições definidas")

# ═══════════════════════════════════════════════════════════════════
# 4. RESOLVER CSP
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RESOLVENDO CSP")
print("="*70)

solucao = csp.resolver()

# ═══════════════════════════════════════════════════════════════════
# 5. EXIBIR SOLUÇÃO
# ═══════════════════════════════════════════════════════════════════

if solucao:
    print("\n✅ SOLUÇÃO ENCONTRADA!")
    print("="*70)

    # Organizar por exame
    agendamentos = {}
    for exame in exames:
        agendamentos[exame] = {
            "medico": solucao[f"{exame}_medico"],
            "horario": solucao[f"{exame}_horario"],
            "sala": solucao[f"{exame}_sala"]
        }

    # Exibir
    print("\n📅 AGENDA DE EXAMES:")
    for exame, info in sorted(agendamentos.items(), 
                              key=lambda x: x[1]['horario']):
        print(f"\n   🔹 {exame}:")
        print(f"      Horário: {info['horario']}")
        print(f"      Médico: {info['medico']}")
        print(f"      Sala: {info['sala']}")

    # Verificar se atende todas as restrições
    print("\n🔍 VERIFICAÇÃO DE RESTRIÇÕES:")
    todas_ok = csp.consistente(solucao)

    if todas_ok:
        print("   ✅ Todas as restrições satisfeitas!")
    else:
        print("   ❌ Algumas restrições violadas!")

    # Visualização: Grade de agendamento
    print("\n📊 GRADE DE AGENDAMENTO:")
    print("="*70)

    # Cabeçalho
    print(f"{'Horário':<10}", end="")
    for sala in salas:
        print(f"{sala:<20}", end="")
    print()
    print("-"*70)

    # Linhas
    for horario in horarios:
        print(f"{horario:<10}", end="")

        for sala in salas:
            # Buscar exame neste horário/sala
            exame_agendado = None
            medico_agendado = None

            for exame, info in agendamentos.items():
                if info['horario'] == horario and info['sala'] == sala:
                    exame_agendado = exame
                    medico_agendado = info['medico']
                    break

            if exame_agendado:
                texto = f"{exame_agendado[:8]}/{medico_agendado.split()[1][:3]}"
                print(f"{texto:<20}", end="")
            else:
                print(f"{'---':<20}", end="")

        print()

else:
    print("\n❌ NENHUMA SOLUÇÃO ENCONTRADA!")
    print("   Possíveis causas:")
    print("   • Restrições muito rígidas")
    print("   • Recursos insuficientes")
    print("   • Conflito entre restrições")

# ═══════════════════════════════════════════════════════════════════
# 6. ANÁLISE DE COMPLEXIDADE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANÁLISE DE COMPLEXIDADE")
print("="*70)

num_vars = len(csp.variaveis)
tamanho_dominio_medio = sum(len(d) for d in csp.dominios.values()) / num_vars

complexidade_bruta = tamanho_dominio_medio ** num_vars

print(f"\n📊 MÉTRICAS:")
print(f"   Variáveis: {num_vars}")
print(f"   Domínio médio: {tamanho_dominio_medio:.1f} valores")
print(f"   Espaço de busca (força bruta): ~{complexidade_bruta:.2e} combinações")
print(f"   Atribuições testadas: {csp.estatisticas['atribuicoes']}")
print(f"   Redução: {(1 - csp.estatisticas['atribuicoes']/complexidade_bruta)*100:.2f}%")

print(f"\n🧠 TÉCNICAS DE OTIMIZAÇÃO:")
print(f"   • Backtracking: Retrocede ao violar restrição")
print(f"   • MRV (Minimum Remaining Values): Escolhe variável com menor domínio")
print(f"   • LCV (Least Constraining Value): Tenta valores menos restritivos primeiro")
print(f"   • Forward Checking: Poderia remover valores inconsistentes dos domínios")
print(f"   • AC-3: Arc Consistency (propagação de restrições)")

# ═══════════════════════════════════════════════════════════════════
# 7. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - CSP")
print("="*70)

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Agendamento (aulas, exames, reuniões)")
print(f"   • Alocação de recursos (salas, equipamentos)")
print(f"   • Planejamento (rotas, turnos)")
print(f"   • Design de circuitos (VLSI)")
print(f"   • Configuração de produtos")
print(f"   • Sudoku, N-Queens, coloração de grafos")

print(f"\n💡 CSP vs OUTROS MÉTODOS:")
print(f"   • CSP vs Busca: CSP explora estrutura do problema")
print(f"   • CSP vs Otimização: CSP busca satisfação, não otimização")
print(f"   • CSP vs SAT: SAT é caso especial (domínios booleanos)")

print(f"\n🔬 COMPONENTES-CHAVE:")
print(f"   • Variáveis: Entidades a atribuir")
print(f"   • Domínios: Valores possíveis")
print(f"   • Restrições: Regras limitando combinações")
print(f"   • Backtracking: Busca com retrocesso")
print(f"   • Heurísticas: MRV, LCV, Degree heuristic")

print("\n✅ CSP SOLVER COMPLETO!")
