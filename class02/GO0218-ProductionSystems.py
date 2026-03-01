# GO0218-ProductionSystems
from typing import List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

# ═══════════════════════════════════════════════════════════════════
# 1. SISTEMA DE PRODUÇÃO - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("="*70)
    print("SISTEMA DE PRODUÇÃO - ASSESSORIA FINANCEIRA")
    print("="*70)

    @dataclass
    class Fato:
        """
        Fato na Working Memory
        """
        nome: str
        atributos: Dict[str, Any] = field(default_factory=dict)

        def match(self, padrao: Dict[str, Any]) -> bool:
            """Verifica se fato corresponde a padrão"""
            for chave, valor_padrao in padrao.items():
                if chave not in self.atributos:
                    return False

                # Suporta wildcards
                if valor_padrao == "*":
                    continue

                # Suporta comparações
                if isinstance(valor_padrao, str) and valor_padrao.startswith(">="):
                    valor_comparacao = float(valor_padrao[2:])
                    if float(self.atributos[chave]) < valor_comparacao:
                        return False
                elif isinstance(valor_padrao, str) and valor_padrao.startswith("<="):
                    valor_comparacao = float(valor_padrao[2:])
                    if float(self.atributos[chave]) > valor_comparacao:
                        return False
                elif self.atributos[chave] != valor_padrao:
                    return False

            return True

        def __repr__(self):
            attrs_str = ", ".join([f"{k}={v}" for k, v in self.atributos.items()])
            return f"{self.nome}({attrs_str})"

    class WorkingMemory:
        """
        Working Memory: Conjunto de fatos atuais
        """

        def __init__(self):
            self.fatos: List[Fato] = []
            self.historico: List[str] = []

        def adicionar_fato(self, fato: Fato):
            """Adicionar fato à memória"""
            self.fatos.append(fato)
            self.historico.append(f"ADDED: {fato}")

        def remover_fato(self, fato: Fato):
            """Remover fato da memória"""
            if fato in self.fatos:
                self.fatos.remove(fato)
                self.historico.append(f"REMOVED: {fato}")

        def buscar_fatos(self, nome: str, padrao: Dict[str, Any] = None) -> List[Fato]:
            """Buscar fatos que correspondem a padrão"""
            resultado = []

            for fato in self.fatos:
                if fato.nome == nome:
                    if padrao is None or fato.match(padrao):
                        resultado.append(fato)

            return resultado

        def existe_fato(self, nome: str, padrao: Dict[str, Any] = None) -> bool:
            """Verifica se existe fato correspondente"""
            return len(self.buscar_fatos(nome, padrao)) > 0

        def __repr__(self):
            return f"WM({len(self.fatos)} fatos)"

    @dataclass
    class Regra:
        """
        Regra de Produção: IF condições THEN ações
        """
        nome: str
        condicoes: List[Callable[[WorkingMemory], bool]]
        acoes: List[Callable[[WorkingMemory], None]]
        prioridade: int = 0
        descricao: str = ""

        def aplicavel(self, wm: WorkingMemory) -> bool:
            """Verifica se regra é aplicável (todas condições satisfeitas)"""
            return all(condicao(wm) for condicao in self.condicoes)

        def executar(self, wm: WorkingMemory):
            """Executar ações da regra"""
            for acao in self.acoes:
                acao(wm)

        def __repr__(self):
            return f"Regra({self.nome}, prioridade={self.prioridade})"

    class SistemaProducao:
        """
        Sistema de Produção com ciclo Match-Resolve-Execute
        """

        def __init__(self, estrategia_resolucao: str = "prioridade"):
            """
            Args:
                estrategia_resolucao: 'prioridade', 'especificidade', 'recencia'
            """
            self.wm = WorkingMemory()
            self.regras: List[Regra] = []
            self.estrategia = estrategia_resolucao
            self.estatisticas = {
                "ciclos": 0,
                "regras_disparadas": 0,
                "regras_por_ciclo": []
            }

        def adicionar_regra(self, regra: Regra):
            """Adicionar regra ao sistema"""
            self.regras.append(regra)

        def match(self) -> List[Regra]:
            """
            MATCH: Encontrar regras aplicáveis (conflict set)

            Returns:
                Lista de regras aplicáveis
            """
            conflict_set = []

            for regra in self.regras:
                if regra.aplicavel(self.wm):
                    conflict_set.append(regra)

            return conflict_set

        def resolve(self, conflict_set: List[Regra]) -> Regra:
            """
            RESOLVE: Escolher regra a executar (conflict resolution)

            Estratégias:
            - prioridade: Maior prioridade
            - especificidade: Mais condições
            - recencia: Regra não usada recentemente
            """
            if not conflict_set:
                return None

            if self.estrategia == "prioridade":
                return max(conflict_set, key=lambda r: r.prioridade)

            elif self.estrategia == "especificidade":
                return max(conflict_set, key=lambda r: len(r.condicoes))

            elif self.estrategia == "recencia":
                # Simplificado: primeira regra
                return conflict_set[0]

            return conflict_set[0]

        def executar_ciclo(self, verbose: bool = False) -> bool:
            """
            Um ciclo Match-Resolve-Execute

            Returns:
                True se regra foi executada, False se nenhuma aplicável
            """
            self.estatisticas["ciclos"] += 1

            # MATCH
            conflict_set = self.match()

            if verbose:
                print(f"\n   CICLO {self.estatisticas['ciclos']}:")
                print(f"      Conflict Set: {len(conflict_set)} regras aplicáveis")

            if not conflict_set:
                if verbose:
                    print("      ❌ Nenhuma regra aplicável - HALT")
                return False

            # RESOLVE
            regra_escolhida = self.resolve(conflict_set)

            if verbose:
                print(f"      ✅ Regra escolhida: {regra_escolhida.nome}")
                if regra_escolhida.descricao:
                    print(f"         {regra_escolhida.descricao}")

            # EXECUTE
            regra_escolhida.executar(self.wm)

            self.estatisticas["regras_disparadas"] += 1
            self.estatisticas["regras_por_ciclo"].append(regra_escolhida.nome)

            return True

        def executar(self, max_ciclos: int = 100, verbose: bool = True):
            """Executar sistema até quiescência ou max_ciclos"""
            print(f"\n🔄 EXECUTANDO SISTEMA DE PRODUÇÃO...")
            print(f"   Estratégia: {self.estrategia}")
            print(f"   Max ciclos: {max_ciclos}")
            print("="*70)

            ciclo = 0
            while ciclo < max_ciclos:
                if not self.executar_ciclo(verbose=verbose):
                    break
                ciclo += 1

            print(f"\n✅ EXECUÇÃO FINALIZADA")
            print(f"   Ciclos: {self.estatisticas['ciclos']}")
            print(f"   Regras disparadas: {self.estatisticas['regras_disparadas']}")

        def relatorio(self):
            """Gerar relatório de execução"""
            print(f"\n📊 RELATÓRIO:")
            print(f"   Working Memory final: {len(self.wm.fatos)} fatos")
            print(f"\n   Fatos finais:")
            for fato in self.wm.fatos:
                print(f"      • {fato}")

            print(f"\n   Sequência de regras:")
            for i, regra_nome in enumerate(self.estatisticas["regras_por_ciclo"], 1):
                print(f"      {i}. {regra_nome}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. DOMÍNIO: ASSESSORIA FINANCEIRA
    # ═══════════════════════════════════════════════════════════════════

    print("\n💼 CONSTRUINDO SISTEMA DE ASSESSORIA FINANCEIRA...")
    print("="*70)

    sistema = SistemaProducao(estrategia_resolucao="prioridade")

    # Estado inicial (Working Memory)
    perfil_cliente = Fato("Cliente", {
        "idade": 35,
        "renda_mensal": 8000,
        "patrimonio": 50000,
        "tolerancia_risco": "moderada",  # conservadora, moderada, agressiva
        "objetivo": "aposentadoria"  # aposentadoria, compra_imovel, educacao
    })

    sistema.wm.adicionar_fato(perfil_cliente)

    print("\n✅ Estado inicial (Working Memory):")
    print(f"   {perfil_cliente}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. DEFINIR REGRAS DE PRODUÇÃO
    # ═══════════════════════════════════════════════════════════════════

    print("\n📋 DEFININDO REGRAS DE PRODUÇÃO...")

    # Regra 1: Criar reserva de emergência (alta prioridade)
    def cond_sem_reserva(wm: WorkingMemory) -> bool:
        return not wm.existe_fato("Recomendacao", {"tipo": "reserva_emergencia"})

    def acao_criar_reserva(wm: WorkingMemory):
        clientes = wm.buscar_fatos("Cliente")
        if clientes:
            cliente = clientes[0]
            renda = cliente.atributos["renda_mensal"]
            reserva_ideal = renda * 6

            recom = Fato("Recomendacao", {
                "tipo": "reserva_emergencia",
                "valor": reserva_ideal,
                "prioridade": "ALTA",
                "justificativa": f"Reserva de emergência de 6 meses ({reserva_ideal:.0f})"
            })
            wm.adicionar_fato(recom)

    regra1 = Regra(
        nome="Criar_Reserva_Emergencia",
        condicoes=[cond_sem_reserva],
        acoes=[acao_criar_reserva],
        prioridade=10,
        descricao="Criar reserva de emergência (6 meses de renda)"
    )

    sistema.adicionar_regra(regra1)
    print("   ✅ R1: Criar_Reserva_Emergencia (prioridade=10)")

    # Regra 2: Investimento conservador para baixa tolerância
    def cond_conservador(wm: WorkingMemory) -> bool:
        clientes = wm.buscar_fatos("Cliente", {"tolerancia_risco": "conservadora"})
        return len(clientes) > 0 and not wm.existe_fato("Alocacao")

    def acao_alocar_conservador(wm: WorkingMemory):
        alocacao = Fato("Alocacao", {
            "renda_fixa": 80,
            "acoes": 10,
            "imoveis": 10,
            "perfil": "conservador"
        })
        wm.adicionar_fato(alocacao)

    regra2 = Regra(
        nome="Alocacao_Conservadora",
        condicoes=[cond_conservador],
        acoes=[acao_alocar_conservador],
        prioridade=8,
        descricao="Alocação 80% renda fixa, 10% ações, 10% imóveis"
    )

    sistema.adicionar_regra(regra2)
    print("   ✅ R2: Alocacao_Conservadora (prioridade=8)")

    # Regra 3: Investimento moderado
    def cond_moderado(wm: WorkingMemory) -> bool:
        clientes = wm.buscar_fatos("Cliente", {"tolerancia_risco": "moderada"})
        return len(clientes) > 0 and not wm.existe_fato("Alocacao")

    def acao_alocar_moderado(wm: WorkingMemory):
        alocacao = Fato("Alocacao", {
            "renda_fixa": 50,
            "acoes": 35,
            "imoveis": 15,
            "perfil": "moderado"
        })
        wm.adicionar_fato(alocacao)

    regra3 = Regra(
        nome="Alocacao_Moderada",
        condicoes=[cond_moderado],
        acoes=[acao_alocar_moderado],
        prioridade=8,
        descricao="Alocação 50% renda fixa, 35% ações, 15% imóveis"
    )

    sistema.adicionar_regra(regra3)
    print("   ✅ R3: Alocacao_Moderada (prioridade=8)")

    # Regra 4: Investimento agressivo
    def cond_agressivo(wm: WorkingMemory) -> bool:
        clientes = wm.buscar_fatos("Cliente", {"tolerancia_risco": "agressiva"})
        return len(clientes) > 0 and not wm.existe_fato("Alocacao")

    def acao_alocar_agressivo(wm: WorkingMemory):
        alocacao = Fato("Alocacao", {
            "renda_fixa": 20,
            "acoes": 60,
            "imoveis": 20,
            "perfil": "agressivo"
        })
        wm.adicionar_fato(alocacao)

    regra4 = Regra(
        nome="Alocacao_Agressiva",
        condicoes=[cond_agressivo],
        acoes=[acao_alocar_agressivo],
        prioridade=8,
        descricao="Alocação 20% renda fixa, 60% ações, 20% imóveis"
    )

    sistema.adicionar_regra(regra4)
    print("   ✅ R4: Alocacao_Agressiva (prioridade=8)")

    # Regra 5: Planejamento aposentadoria (jovem)
    def cond_aposentadoria_jovem(wm: WorkingMemory) -> bool:
        clientes = wm.buscar_fatos("Cliente", {"objetivo": "aposentadoria"})
        if not clientes:
            return False

        cliente = clientes[0]
        return (cliente.atributos["idade"] < 40 and 
                wm.existe_fato("Alocacao") and 
                not wm.existe_fato("Recomendacao", {"tipo": "aposentadoria"}))

    def acao_planejar_aposentadoria_jovem(wm: WorkingMemory):
        recom = Fato("Recomendacao", {
            "tipo": "aposentadoria",
            "estrategia": "longo_prazo",
            "contribuicao_mensal": 1000,
            "justificativa": "Jovem: foco em crescimento de longo prazo"
        })
        wm.adicionar_fato(recom)

    regra5 = Regra(
        nome="Aposentadoria_Jovem",
        condicoes=[cond_aposentadoria_jovem],
        acoes=[acao_planejar_aposentadoria_jovem],
        prioridade=7,
        descricao="Planejamento de aposentadoria para jovens (<40 anos)"
    )

    sistema.adicionar_regra(regra5)
    print("   ✅ R5: Aposentadoria_Jovem (prioridade=7)")

    # Regra 6: Revisar periodicamente
    def cond_revisao_anual(wm: WorkingMemory) -> bool:
        return (wm.existe_fato("Alocacao") and 
                wm.existe_fato("Recomendacao") and
                not wm.existe_fato("Revisao"))

    def acao_agendar_revisao(wm: WorkingMemory):
        revisao = Fato("Revisao", {
            "frequencia": "anual",
            "proxima_data": "2026-01",
            "justificativa": "Revisão anual de portfólio"
        })
        wm.adicionar_fato(revisao)

    regra6 = Regra(
        nome="Agendar_Revisao",
        condicoes=[cond_revisao_anual],
        acoes=[acao_agendar_revisao],
        prioridade=5,
        descricao="Agendar revisão anual de portfólio"
    )

    sistema.adicionar_regra(regra6)
    print("   ✅ R6: Agendar_Revisao (prioridade=5)")

    print(f"\n✅ {len(sistema.regras)} regras adicionadas")

    # ═══════════════════════════════════════════════════════════════════
    # 4. EXECUTAR SISTEMA
    # ═══════════════════════════════════════════════════════════════════

    sistema.executar(max_ciclos=20, verbose=True)

    # ═══════════════════════════════════════════════════════════════════
    # 5. RELATÓRIO FINAL
    # ═══════════════════════════════════════════════════════════════════

    sistema.relatorio()

    print("\n" + "="*70)
    print("RELATÓRIO FINAL - SISTEMA DE PRODUÇÃO")
    print("="*70)

    print(f"\n🏗️ ARQUITETURA:")
    print(f"   • Working Memory: Fatos dinâmicos")
    print(f"   • Production Rules: IF-THEN")
    print(f"   • Ciclo: Match → Resolve → Execute")
    print(f"   • Conflict Resolution: {sistema.estrategia}")

    print(f"\n🔄 CICLO Match-Resolve-Execute:")
    print(f"   1. MATCH: Encontrar regras aplicáveis")
    print(f"   2. RESOLVE: Escolher uma (conflict resolution)")
    print(f"   3. EXECUTE: Executar ações da regra")
    print(f"   4. Repetir até quiescência")

    print(f"\n🎯 APLICAÇÕES REAIS:")
    print(f"   • Sistemas especialistas (diagnóstico, assessoria)")
    print(f"   • Automação de processos (workflows)")
    print(f"   • Jogos (comportamento de NPCs)")
    print(f"   • Monitoramento (alertas, detecção)")
    print(f"   • Business rules engines")

    print(f"\n💡 VANTAGENS:")
    print(f"   ✅ MODULAR: Regras independentes")
    print(f"   ✅ EXPLICÁVEL: Sequência de regras = explicação")
    print(f"   ✅ MANUTENÍVEL: Adicionar/remover regras facilmente")
    print(f"   ✅ DECLARATIVO: Foco no 'o quê', não 'como'")

    print(f"\n🔬 CONFLICT RESOLUTION:")
    print(f"   • Prioridade: Escolhe regra com maior prioridade")
    print(f"   • Especificidade: Prefere regras mais específicas")
    print(f"   • Recência: Usa fatos mais recentes")
    print(f"   • LEX, MEA: Algoritmos mais sofisticados")

    print(f"\n⚙️ PRODUCTION SYSTEMS vs OUTROS:")
    print(f"   • vs Forward Chaining: Similar (PS é generalização)")
    print(f"   • vs Workflows: PS é mais dinâmico (conflict resolution)")
    print(f"   • vs Scripts: PS é reativo (regras disparam quando aplicáveis)")

    print("\n✅ SISTEMA DE PRODUÇÃO COMPLETO!")
