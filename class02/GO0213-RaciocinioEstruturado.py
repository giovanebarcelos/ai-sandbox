# GO0213-RaciocínioEstruturado
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
import json

# ═══════════════════════════════════════════════════════════════════
# 1. SISTEMA DE FRAMES - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("FRAMES & SCRIPTS - SISTEMA DE GERENCIAMENTO DE HOTEL")
print("="*70)

class Slot:
    """
    Slot de um Frame (atributo com valor, default e demons)
    """

    def __init__(self, nome: str, valor: Any = None, 
                 default: Any = None, tipo: type = None):
        """
        Args:
            nome: Nome do slot (ex: "preco", "capacidade")
            valor: Valor atual
            default: Valor padrão
            tipo: Tipo esperado (int, str, etc.)
        """
        self.nome = nome
        self._valor = valor
        self.default = default
        self.tipo = tipo

        # Demons (procedimentos)
        self.if_needed = None  # Executado se valor não existe
        self.if_added = None   # Executado quando valor é setado
        self.if_removed = None # Executado quando valor é removido

    @property
    def valor(self):
        """Get valor (ativa if_needed se não existe)"""
        if self._valor is None:
            if self.if_needed:
                self._valor = self.if_needed()
            elif self.default is not None:
                self._valor = self.default
        return self._valor

    @valor.setter
    def valor(self, novo_valor):
        """Set valor (ativa if_added)"""
        if self.tipo and not isinstance(novo_valor, self.tipo):
            raise TypeError(f"Slot {self.nome} espera tipo {self.tipo}")

        antigo = self._valor
        self._valor = novo_valor

        if self.if_added:
            self.if_added(antigo, novo_valor)

    def remover(self):
        """Remove valor (ativa if_removed)"""
        if self.if_removed:
            self.if_removed(self._valor)
        self._valor = None

    def __repr__(self):
        return f"Slot({self.nome}={self.valor})"

class Frame:
    """
    Frame: Estrutura de conhecimento com slots e herança
    """

    def __init__(self, nome: str, pai: Optional['Frame'] = None):
        """
        Args:
            nome: Nome do frame
            pai: Frame pai (herança)
        """
        self.nome = nome
        self.pai = pai
        self.slots: Dict[str, Slot] = {}
        self.instancias: List['Frame'] = []  # Para frames tipo classe

    def adicionar_slot(self, slot: Slot):
        """Adiciona slot ao frame"""
        self.slots[slot.nome] = slot

    def get_slot(self, nome: str) -> Optional[Slot]:
        """
        Busca slot (com herança)

        Busca ordem:
        1. Slots próprios
        2. Slots do pai (recursivo)
        """
        if nome in self.slots:
            return self.slots[nome]
        elif self.pai:
            return self.pai.get_slot(nome)
        return None

    def get_valor(self, nome_slot: str) -> Any:
        """Obter valor de um slot"""
        slot = self.get_slot(nome_slot)
        return slot.valor if slot else None

    def set_valor(self, nome_slot: str, valor: Any):
        """Setar valor de um slot"""
        slot = self.get_slot(nome_slot)
        if slot:
            slot.valor = valor
        else:
            # Criar slot se não existe
            novo_slot = Slot(nome_slot, valor=valor)
            self.adicionar_slot(novo_slot)

    def criar_instancia(self, nome_instancia: str) -> 'Frame':
        """Criar instância deste frame"""
        instancia = Frame(nome_instancia, pai=self)
        self.instancias.append(instancia)
        return instancia

    def __repr__(self):
        pai_str = f" (herda de {self.pai.nome})" if self.pai else ""
        return f"Frame({self.nome}{pai_str}, {len(self.slots)} slots)"

# ═══════════════════════════════════════════════════════════════════
# 2. HIERARQUIA DE FRAMES - HOTEL
# ═══════════════════════════════════════════════════════════════════

print("\n🏗️ CONSTRUINDO HIERARQUIA DE FRAMES...")

# Frame raiz: Acomodacao
frame_acomodacao = Frame("Acomodacao")

slot_tipo = Slot("tipo", default="genérica")
slot_capacidade = Slot("capacidade", default=2, tipo=int)
slot_preco_base = Slot("preco_base", default=100.0, tipo=float)
slot_disponivel = Slot("disponivel", default=True, tipo=bool)

frame_acomodacao.adicionar_slot(slot_tipo)
frame_acomodacao.adicionar_slot(slot_capacidade)
frame_acomodacao.adicionar_slot(slot_preco_base)
frame_acomodacao.adicionar_slot(slot_disponivel)

print("   ✅ Frame raiz: Acomodacao (capacidade, preco_base, disponivel)")

# Frame: Quarto (herda de Acomodacao)
frame_quarto = Frame("Quarto", pai=frame_acomodacao)

slot_numero = Slot("numero", tipo=int)
slot_andar = Slot("andar", tipo=int)
slot_vista = Slot("vista", default="interna")

# Demon: Calcular preço com desconto por andar baixo
def calcular_preco_com_andar():
    andar = frame_quarto.get_valor("andar")
    preco_base = frame_quarto.pai.get_valor("preco_base")
    if andar and andar <= 2:
        return preco_base * 0.9  # 10% desconto
    return preco_base

slot_preco_final = Slot("preco_final")
slot_preco_final.if_needed = calcular_preco_com_andar

frame_quarto.adicionar_slot(slot_numero)
frame_quarto.adicionar_slot(slot_andar)
frame_quarto.adicionar_slot(slot_vista)
frame_quarto.adicionar_slot(slot_preco_final)

print("   ✅ Frame: Quarto (numero, andar, vista, preco_final)")
print("      → Herda: capacidade, preco_base, disponivel")
print("      → Demon: preco_final (if-needed) = preco_base * desconto_andar")

# Frame: Suite (herda de Quarto)
frame_suite = Frame("Suite", pai=frame_quarto)

slot_suite_tipo = Slot("tipo", valor="suite")  # Sobrescreve default
slot_suite_capacidade = Slot("capacidade", valor=4)  # Sobrescreve
slot_suite_preco = Slot("preco_base", valor=300.0)
slot_suite_amenidades = Slot("amenidades", default=["wifi", "tv", "minibar", "jacuzzi"])

frame_suite.adicionar_slot(slot_suite_tipo)
frame_suite.adicionar_slot(slot_suite_capacidade)
frame_suite.adicionar_slot(slot_suite_preco)
frame_suite.adicionar_slot(slot_suite_amenidades)

print("   ✅ Frame: Suite (override capacidade=4, preco_base=300)")
print("      → Herda: numero, andar, vista, disponivel")

# ═══════════════════════════════════════════════════════════════════
# 3. CRIAR INSTÂNCIAS
# ═══════════════════════════════════════════════════════════════════

print("\n🏨 CRIANDO INSTÂNCIAS DE QUARTOS...")

# Quarto padrão
quarto_101 = frame_quarto.criar_instancia("Quarto_101")
quarto_101.set_valor("numero", 101)
quarto_101.set_valor("andar", 1)
quarto_101.set_valor("vista", "jardim")

print(f"   ✅ {quarto_101.nome}:")
print(f"      Número: {quarto_101.get_valor('numero')}")
print(f"      Andar: {quarto_101.get_valor('andar')}")
print(f"      Capacidade: {quarto_101.get_valor('capacidade')} (herdado)")
print(f"      Preço base: R$ {quarto_101.pai.get_valor('preco_base')}")
print(f"      Preço final: R$ {quarto_101.get_valor('preco_final')} (demon if-needed)")

# Suite
suite_501 = frame_suite.criar_instancia("Suite_501")
suite_501.set_valor("numero", 501)
suite_501.set_valor("andar", 5)
suite_501.set_valor("vista", "mar")

print(f"\n   ✅ {suite_501.nome}:")
print(f"      Número: {suite_501.get_valor('numero')}")
print(f"      Capacidade: {suite_501.get_valor('capacidade')} (override=4)")
print(f"      Preço base: R$ {suite_501.pai.get_valor('preco_base')} (override=300)")
print(f"      Amenidades: {suite_501.get_valor('amenidades')}")

# ═══════════════════════════════════════════════════════════════════
# 4. SCRIPTS - SEQUÊNCIAS ESTEREOTIPADAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SCRIPTS - SEQUÊNCIAS DE EVENTOS")
print("="*70)

class Script:
    """
    Script: Sequência estereotipada de eventos

    Exemplo: Script de "Check-in de Hotel"
    """

    def __init__(self, nome: str):
        self.nome = nome
        self.cenas: List[Dict] = []  # Lista de cenas
        self.props: Dict[str, Any] = {}  # Objetos do script
        self.roles: Dict[str, str] = {}  # Papéis (atores)
        self.condicoes: Dict[str, bool] = {}  # Condições

    def adicionar_cena(self, numero: int, descricao: str, 
                       acoes: List[str], condicoes: Dict = None):
        """Adiciona cena ao script"""
        cena = {
            "numero": numero,
            "descricao": descricao,
            "acoes": acoes,
            "condicoes": condicoes or {},
            "executada": False
        }
        self.cenas.append(cena)

    def executar(self, contexto: Dict = None):
        """Executar script sequencialmente"""
        print(f"\n🎬 EXECUTANDO SCRIPT: {self.nome}")
        print("="*70)

        contexto = contexto or {}

        for cena in self.cenas:
            # Verificar condições
            pode_executar = True
            for cond, valor_esperado in cena['condicoes'].items():
                if contexto.get(cond) != valor_esperado:
                    pode_executar = False
                    print(f"\n   ⏭️  CENA {cena['numero']}: {cena['descricao']} - PULADA")
                    print(f"       Condição não satisfeita: {cond}={contexto.get(cond)} (esperado: {valor_esperado})")
                    break

            if not pode_executar:
                continue

            # Executar cena
            print(f"\n   🎬 CENA {cena['numero']}: {cena['descricao']}")
            for i, acao in enumerate(cena['acoes'], 1):
                print(f"      {i}. {acao}")

            cena['executada'] = True

        print("\n" + "="*70)
        print("✅ SCRIPT COMPLETO!")

    def resumo(self):
        """Gerar resumo do script"""
        total = len(self.cenas)
        executadas = sum(1 for c in self.cenas if c['executada'])
        return {
            "nome": self.nome,
            "total_cenas": total,
            "executadas": executadas,
            "taxa_execucao": f"{executadas/total*100:.0f}%" if total > 0 else "0%"
        }

# Script: Check-in de Hotel
script_checkin = Script("Check-in de Hotel")

script_checkin.adicionar_cena(
    numero=1,
    descricao="Chegada ao Hotel",
    acoes=[
        "Cliente chega à recepção",
        "Recepcionista saúda o cliente",
        "Cliente apresenta documento de identidade"
    ]
)

script_checkin.adicionar_cena(
    numero=2,
    descricao="Verificação de Reserva",
    acoes=[
        "Recepcionista consulta sistema de reservas",
        "Confirma nome, datas e tipo de quarto",
        "Verifica disponibilidade do quarto"
    ]
)

script_checkin.adicionar_cena(
    numero=3,
    descricao="Reserva Antecipada",
    acoes=[
        "Recepcionista localiza reserva",
        "Confirma dados com cliente",
        "Pula para Cena 5 (Pagamento)"
    ],
    condicoes={"reserva_existente": True}
)

script_checkin.adicionar_cena(
    numero=4,
    descricao="Walk-in (Sem Reserva)",
    acoes=[
        "Recepcionista verifica quartos disponíveis",
        "Apresenta opções e preços",
        "Cliente escolhe quarto",
        "Recepcionista cria nova reserva"
    ],
    condicoes={"reserva_existente": False}
)

script_checkin.adicionar_cena(
    numero=5,
    descricao="Pagamento",
    acoes=[
        "Recepcionista informa valor total",
        "Cliente fornece método de pagamento",
        "Recepcionista processa pagamento",
        "Emite recibo"
    ]
)

script_checkin.adicionar_cena(
    numero=6,
    descricao="Entrega das Chaves",
    acoes=[
        "Recepcionista programa cartão-chave",
        "Explica comodidades do hotel (café, wifi, horários)",
        "Entrega chave e mapa do hotel",
        "Deseja boa estadia"
    ]
)

print("\n📋 ESTRUTURA DO SCRIPT:")
print(f"   Nome: {script_checkin.nome}")
print(f"   Cenas: {len(script_checkin.cenas)}")
for cena in script_checkin.cenas:
    cond_str = f" [SE {list(cena['condicoes'].keys())[0]}]" if cena['condicoes'] else ""
    print(f"      {cena['numero']}. {cena['descricao']}{cond_str}")

# ═══════════════════════════════════════════════════════════════════
# 5. EXECUTAR SCRIPTS EM DIFERENTES CONTEXTOS
# ═══════════════════════════════════════════════════════════════════

# Contexto 1: Cliente com reserva
print("\n" + "="*70)
print("CENÁRIO 1: Cliente com reserva antecipada")
print("="*70)

contexto_com_reserva = {
    "reserva_existente": True,
    "cliente": "João Silva",
    "quarto": suite_501
}

script_checkin.executar(contexto_com_reserva)
resumo1 = script_checkin.resumo()

# Reset
for cena in script_checkin.cenas:
    cena['executada'] = False

# Contexto 2: Walk-in (sem reserva)
print("\n" + "="*70)
print("CENÁRIO 2: Cliente walk-in (sem reserva)")
print("="*70)

contexto_sem_reserva = {
    "reserva_existente": False,
    "cliente": "Maria Santos",
    "quarto": quarto_101
}

script_checkin.executar(contexto_sem_reserva)
resumo2 = script_checkin.resumo()

# ═══════════════════════════════════════════════════════════════════
# 6. FRAME-BASED REASONING: DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RACIOCÍNIO BASEADO EM FRAMES")
print("="*70)

def diagnosticar_quarto(frame_quarto_inst: Frame):
    """
    Raciocínio baseado em frames para recomendar quarto
    """
    print(f"\n🔍 DIAGNÓSTICO: {frame_quarto_inst.nome}")
    print("="*70)

    # Coletar dados
    capacidade = frame_quarto_inst.get_valor("capacidade")
    preco = frame_quarto_inst.get_valor("preco_final") or frame_quarto_inst.pai.get_valor("preco_base")
    vista = frame_quarto_inst.get_valor("vista")
    tipo = frame_quarto_inst.get_valor("tipo")

    print(f"   Tipo: {tipo}")
    print(f"   Capacidade: {capacidade} pessoas")
    print(f"   Preço: R$ {preco:.2f}")
    print(f"   Vista: {vista}")

    # Raciocínio
    print(f"\n💡 RECOMENDAÇÕES:")

    if capacidade >= 4:
        print(f"   ✅ Ideal para famílias ou grupos")
    elif capacidade == 2:
        print(f"   ✅ Ideal para casais")

    if preco < 150:
        print(f"   ✅ Opção econômica")
    elif preco > 250:
        print(f"   ✅ Opção premium")

    if vista in ["mar", "montanha"]:
        print(f"   ✅ Vista privilegiada ({vista})")

    # Herança
    print(f"\n🧬 HERANÇA:")
    atual = frame_quarto_inst
    nivel = 0
    while atual:
        print(f"   {'  '*nivel}↳ {atual.nome}")
        atual = atual.pai
        nivel += 1

# Diagnosticar quartos
diagnosticar_quarto(quarto_101)
diagnosticar_quarto(suite_501)

# ═══════════════════════════════════════════════════════════════════
# 7. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - FRAMES & SCRIPTS")
print("="*70)

print(f"\n📊 FRAMES CRIADOS:")
print(f"   • {frame_acomodacao.nome}: {len(frame_acomodacao.slots)} slots")
print(f"   • {frame_quarto.nome}: +{len(frame_quarto.slots)} slots")
print(f"   • {frame_suite.nome}: +{len(frame_suite.slots)} slots (overrides)")
print(f"   • Instâncias: {quarto_101.nome}, {suite_501.nome}")

print(f"\n🔄 HERANÇA:")
print(f"   Acomodacao → Quarto → Suite")
print(f"   Suite herda {len([s for s in suite_501.pai.pai.slots])} slots de Acomodacao")

print(f"\n👹 DEMONS (Procedimentos Ativos):")
print(f"   • if-needed: Calcula preco_final baseado em andar")
print(f"   • if-added: Poderia validar ou notificar")
print(f"   • if-removed: Poderia liberar recursos")

print(f"\n🎬 SCRIPTS:")
print(f"   • {script_checkin.nome}: {len(script_checkin.cenas)} cenas")
print(f"   • Cenário 1 (com reserva): {resumo1['taxa_execucao']} cenas executadas")
print(f"   • Cenário 2 (walk-in): {resumo2['taxa_execucao']} cenas executadas")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Sistemas especialistas (diagnóstico médico)")
print(f"   • Planejamento (robótica, logística)")
print(f"   • Processamento de linguagem natural")
print(f"   • Jogos (comportamento de NPCs)")
print(f"   • Sistemas de recomendação")

print(f"\n💡 FRAMES vs OUTROS:")
print(f"   • Frames vs OOP: Similar, mas com herança + defaults + demons")
print(f"   • Frames vs Ontologias: Menos formais, mais procedurais")
print(f"   • Scripts vs Workflows: Scripts são estereotipados (fixos)")

print(f"\n🔬 CONCEITOS-CHAVE:")
print(f"   • SLOTS: Atributos com valores, defaults, tipos")
print(f"   • HERANÇA: Frames especializados herdam de genéricos")
print(f"   • DEMONS: Procedimentos if-needed, if-added, if-removed")
print(f"   • SCRIPTS: Sequências estereotipadas de eventos")
print(f"   • ROLES: Papéis no script (recepcionista, cliente)")
print(f"   • PROPS: Objetos do script (chave, recibo)")

print(f"\n✅ SISTEMA DE FRAMES COMPLETO!")
