# GO0114-AgenteDeIaAutônomoRobôAspirador
import random
from enum import Enum
import time

# ═══════════════════════════════════════════════════════════════════
# AGENTE DE IA AUTÔNOMO - ROBÔ ASPIRADOR
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("AGENTE AUTÔNOMO - ARQUITETURA PERCEPÇÃO-RACIOCÍNIO-AÇÃO")
print("="*70)

class EstadoCasa(Enum):
    LIMPO = "🟩"
    SUJO = "🟥"
    OBSTACULO = "⬛"

class Ambiente:
    """Ambiente simulado"""

    def __init__(self, largura: int = 5, altura: int = 5):
        self.largura = largura
        self.altura = altura
        self.grid = [[EstadoCasa.LIMPO for _ in range(largura)] for _ in range(altura)]

        # Adicionar sujeira (30%)
        for y in range(altura):
            for x in range(largura):
                if random.random() < 0.3:
                    self.grid[y][x] = EstadoCasa.SUJO

        # Obstáculos (10%)
        for y in range(altura):
            for x in range(largura):
                if random.random() < 0.1 and self.grid[y][x] != EstadoCasa.SUJO:
                    self.grid[y][x] = EstadoCasa.OBSTACULO

        self.robo_x = 0
        self.robo_y = 0
        self.grid[0][0] = EstadoCasa.LIMPO

    def get_estado(self, x: int, y: int):
        if 0 <= x < self.largura and 0 <= y < self.altura:
            return self.grid[y][x]
        return None

    def limpar(self, x: int, y: int) -> bool:
        if self.grid[y][x] == EstadoCasa.SUJO:
            self.grid[y][x] = EstadoCasa.LIMPO
            return True
        return False

    def mover_robo(self, nova_x: int, nova_y: int) -> bool:
        if 0 <= nova_x < self.largura and 0 <= nova_y < self.altura:
            if self.grid[nova_y][nova_x] != EstadoCasa.OBSTACULO:
                self.robo_x = nova_x
                self.robo_y = nova_y
                return True
        return False

    def visualizar(self):
        print("\n" + "="*30)
        for y in range(self.altura):
            linha = ""
            for x in range(self.largura):
                if x == self.robo_x and y == self.robo_y:
                    linha += "🤖 "
                else:
                    linha += self.grid[y][x].value + " "
            print(linha)
        print("="*30)

    def contar_sujeira(self) -> int:
        return sum(row.count(EstadoCasa.SUJO) for row in self.grid)

class AgenteAspirador:
    """Agente inteligente"""

    def __init__(self, ambiente: Ambiente):
        self.ambiente = ambiente
        self.posicao_x = ambiente.robo_x
        self.posicao_y = ambiente.robo_y
        self.celulas_visitadas = []
        self.celulas_limpas = 0
        self.energia = 100
        self.passos = 0

    def perceber(self):
        """SENSORES"""
        estado_atual = self.ambiente.get_estado(self.posicao_x, self.posicao_y)

        return {
            "posicao": (self.posicao_x, self.posicao_y),
            "estado_atual": estado_atual,
            "sujeira_aqui": (estado_atual == EstadoCasa.SUJO),
            "norte_livre": self.ambiente.get_estado(self.posicao_x, self.posicao_y - 1) != EstadoCasa.OBSTACULO,
            "sul_livre": self.ambiente.get_estado(self.posicao_x, self.posicao_y + 1) != EstadoCasa.OBSTACULO,
            "leste_livre": self.ambiente.get_estado(self.posicao_x + 1, self.posicao_y) != EstadoCasa.OBSTACULO,
            "oeste_livre": self.ambiente.get_estado(self.posicao_x - 1, self.posicao_y) != EstadoCasa.OBSTACULO,
            "energia": self.energia
        }

    def raciocinar(self, percepcao):
        """RACIOCÍNIO"""
        # Regra 1: Energia baixa
        if percepcao["energia"] < 20:
            return "RECARREGAR"

        # Regra 2: Sujeira aqui
        if percepcao["sujeira_aqui"]:
            return "LIMPAR"

        # Regra 3: Explorar
        opcoes = []

        if percepcao["norte_livre"]:
            nova_pos = (self.posicao_x, self.posicao_y - 1)
            opcoes.append(("MOVER_NORTE", nova_pos))

        if percepcao["sul_livre"]:
            nova_pos = (self.posicao_x, self.posicao_y + 1)
            opcoes.append(("MOVER_SUL", nova_pos))

        if percepcao["leste_livre"]:
            nova_pos = (self.posicao_x + 1, self.posicao_y)
            opcoes.append(("MOVER_LESTE", nova_pos))

        if percepcao["oeste_livre"]:
            nova_pos = (self.posicao_x - 1, self.posicao_y)
            opcoes.append(("MOVER_OESTE", nova_pos))

        # Preferir não visitadas
        nao_visitadas = [op for op in opcoes if op[1] not in self.celulas_visitadas]

        if nao_visitadas:
            return random.choice(nao_visitadas)[0]
        elif opcoes:
            return random.choice(opcoes)[0]
        else:
            return "ESPERAR"

    def agir(self, acao: str) -> bool:
        """ATUADORES"""
        self.passos += 1
        self.energia -= 1

        if acao == "LIMPAR":
            if self.ambiente.limpar(self.posicao_x, self.posicao_y):
                self.celulas_limpas += 1
                print(f"   🧹 Limpou ({self.posicao_x}, {self.posicao_y})")
                return True
            return False

        elif acao == "RECARREGAR":
            self.energia = 100
            print(f"   🔋 Recarregou")
            return True

        elif acao.startswith("MOVER_"):
            direcao = acao.split("_")[1]
            nova_x, nova_y = self.posicao_x, self.posicao_y

            if direcao == "NORTE":
                nova_y -= 1
            elif direcao == "SUL":
                nova_y += 1
            elif direcao == "LESTE":
                nova_x += 1
            elif direcao == "OESTE":
                nova_x -= 1

            if self.ambiente.mover_robo(nova_x, nova_y):
                self.posicao_x = nova_x
                self.posicao_y = nova_y
                self.celulas_visitadas.append((nova_x, nova_y))
                print(f"   🚶 Moveu para ({nova_x}, {nova_y})")
                return True

            return False

        return False

    def executar_ciclo(self, mostrar: bool = True):
        """Ciclo completo"""
        percepcao = self.perceber()

        if mostrar:
            print(f"\n🔍 PERCEPÇÃO:")
            print(f"   Posição: {percepcao['posicao']}")
            print(f"   Sujeira: {'Sim' if percepcao['sujeira_aqui'] else 'Não'}")
            print(f"   Energia: {percepcao['energia']}%")

        acao = self.raciocinar(percepcao)

        if mostrar:
            print(f"\n🧠 DECISÃO: {acao}")
            print(f"\n⚙️ AÇÃO:")

        return self.agir(acao)

# Simulação
print("\n🏠 CRIANDO AMBIENTE...")
ambiente = Ambiente(largura=5, altura=5)

print(f"\n📊 Estado Inicial:")
print(f"   Sujeira: {ambiente.contar_sujeira()} células")

ambiente.visualizar()

print("\n🤖 CRIANDO AGENTE...")
agente = AgenteAspirador(ambiente)

print("\n▶️ SIMULAÇÃO (15 ciclos)...")
print("="*70)

for ciclo in range(15):
    print(f"\n{'='*70}")
    print(f"CICLO {ciclo + 1}")
    print(f"{'='*70}")

    agente.executar_ciclo(mostrar=True)

    if (ciclo + 1) % 5 == 0:
        ambiente.visualizar()
        print(f"\n📈 Progresso:")
        print(f"   Limpas: {agente.celulas_limpas}")
        print(f"   Restante: {ambiente.contar_sujeira()}")
        print(f"   Energia: {agente.energia}%")

    if ambiente.contar_sujeira() == 0:
        print(f"\n🎉 MISSÃO CONCLUÍDA!")
        break

    time.sleep(0.05)

# Relatório
print("\n" + "="*70)
print("RELATÓRIO FINAL")
print("="*70)

ambiente.visualizar()

print(f"\n📊 ESTATÍSTICAS:")
print(f"   Passos: {agente.passos}")
print(f"   Limpas: {agente.celulas_limpas}")
print(f"   Restante: {ambiente.contar_sujeira()}")
print(f"   Visitadas: {len(agente.celulas_visitadas)}")
print(f"   Energia: {agente.energia}%")

eficiencia = (agente.celulas_limpas / agente.passos * 100) if agente.passos > 0 else 0
print(f"   Eficiência: {eficiencia:.1f}%")

print("\n💡 CONCEITOS DEMONSTRADOS:")
print("   1️⃣ AGENTE AUTÔNOMO: Percepção → Raciocínio → Ação")
print("   2️⃣ ARQUITETURA REATIVA: Regras simples")
print("   3️⃣ MEMÓRIA INTERNA: Células visitadas")
print("   4️⃣ GESTÃO DE RECURSOS: Energia limitada")
print("   5️⃣ EXPLORAÇÃO: Preferir não visitadas")

print("\n🔄 TIPOS DE AGENTES (Russell & Norvig):")
print("   • Reflexo Simples: Sensores → ação")
print("   • Reflexo com Estado: + memória")
print("   • Baseado em Objetivos: + planejamento (este)")
print("   • Baseado em Utilidade: + função custo")
print("   • Aprendizado: + melhora experiência")

print("\n🤖 APLICAÇÕES REAIS:")
print("   • Roomba (aspiradores robóticos)")
print("   • Drones autônomos")
print("   • Veículos autônomos")
print("   • NPCs em jogos")

print("\n✅ SIMULAÇÃO COMPLETA!")
