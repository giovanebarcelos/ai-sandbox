# GO0210-MotorDeInferênciaComForwardChaining
import re
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# 1. MOTOR DE INFERÊNCIA - FORWARD CHAINING
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("="*70)
    print("MOTOR DE INFERÊNCIA - FORWARD CHAINING")
    print("="*70)

    class ForwardChainingEngine:
        """
        Motor de inferência com encadeamento para frente

        Forward Chaining: Parte dos fatos e aplica regras até conclusão
        """

        def __init__(self):
            self.fatos = set()  # Base de fatos conhecidos
            self.regras = []    # Base de regras (IF-THEN)
            self.inferencias = []  # Histórico de inferências

        def adicionar_fato(self, fato):
            """Adicionar fato à base de conhecimento"""
            if fato not in self.fatos:
                self.fatos.add(fato)
                print(f"   ✅ Fato adicionado: {fato}")
                return True
            return False

        def adicionar_regra(self, condicoes, conclusao, nome=""):
            """
            Adicionar regra à base

            Args:
                condicoes: Lista de fatos necessários (AND)
                conclusao: Fato inferido se condições satisfeitas
                nome: Nome descritivo da regra
            """
            regra = {
                'condicoes': set(condicoes),
                'conclusao': conclusao,
                'nome': nome if nome else f"Regra_{len(self.regras)+1}"
            }
            self.regras.append(regra)
            print(f"   📋 Regra adicionada: {regra['nome']}")
            print(f"      IF {' AND '.join(condicoes)}")
            print(f"      THEN {conclusao}")

        def verificar_regra(self, regra):
            """Verificar se todas as condições de uma regra são satisfeitas"""
            return regra['condicoes'].issubset(self.fatos)

        def aplicar_regra(self, regra):
            """Aplicar regra e adicionar conclusão aos fatos"""
            if self.adicionar_fato(regra['conclusao']):
                inferencia = {
                    'regra': regra['nome'],
                    'condicoes': regra['condicoes'],
                    'conclusao': regra['conclusao']
                }
                self.inferencias.append(inferencia)
                return True
            return False

        def inferir(self, max_iteracoes=10):
            """
            Executar forward chaining

            Aplica regras repetidamente até:
            - Não haver mais regras aplicáveis
            - Atingir max_iteracoes
            """
            print("\n" + "="*70)
            print("INICIANDO INFERÊNCIA (FORWARD CHAINING)")
            print("="*70)

            iteracao = 0
            novos_fatos = True

            while novos_fatos and iteracao < max_iteracoes:
                iteracao += 1
                novos_fatos = False

                print(f"\n🔄 ITERAÇÃO {iteracao}:")
                print(f"   Fatos conhecidos: {len(self.fatos)}")

                for regra in self.regras:
                    if self.verificar_regra(regra):
                        if regra['conclusao'] not in self.fatos:
                            print(f"\n   🔥 Regra DISPARADA: {regra['nome']}")
                            print(f"      Condições satisfeitas: {regra['condicoes']}")
                            print(f"      ➡️ INFERINDO: {regra['conclusao']}")

                            self.aplicar_regra(regra)
                            novos_fatos = True

                if not novos_fatos:
                    print(f"\n   ⏹️ Nenhuma regra aplicável nesta iteração")

            print(f"\n✅ Inferência concluída em {iteracao} iterações")
            print(f"   Total de fatos: {len(self.fatos)}")
            print(f"   Inferências realizadas: {len(self.inferencias)}")

        def explicar(self):
            """Explicar cadeia de raciocínio"""
            print("\n" + "="*70)
            print("EXPLICAÇÃO DO RACIOCÍNIO")
            print("="*70)

            print("\n📝 FATOS INICIAIS:")
            fatos_iniciais = self.fatos - set([inf['conclusao'] for inf in self.inferencias])
            for fato in sorted(fatos_iniciais):
                print(f"   • {fato}")

            print("\n🧠 CADEIA DE INFERÊNCIAS:")
            for i, inf in enumerate(self.inferencias, 1):
                print(f"\n   {i}. {inf['regra']}:")
                print(f"      IF {' AND '.join(sorted(inf['condicoes']))}")
                print(f"      THEN ➡️ {inf['conclusao']}")

            print("\n🎯 FATOS FINAIS:")
            for fato in sorted(self.fatos):
                origem = "INICIAL" if fato in fatos_iniciais else "INFERIDO"
                print(f"   • {fato} [{origem}]")

        def buscar_fato(self, padrao):
            """Buscar fatos que contenham o padrão"""
            return [f for f in self.fatos if padrao.lower() in f.lower()]

    # ═══════════════════════════════════════════════════════════════════
    # 2. SISTEMA ESPECIALISTA AUTOMOTIVO
    # ═══════════════════════════════════════════════════════════════════

    print("\n🚗 CRIANDO SISTEMA ESPECIALISTA AUTOMOTIVO...")

    engine = ForwardChainingEngine()

    # ═══════════════════════════════════════════════════════════════════
    # 3. BASE DE REGRAS (DIAGNÓSTICO AUTOMOTIVO)
    # ═══════════════════════════════════════════════════════════════════

    print("\n📚 ADICIONANDO REGRAS À BASE DE CONHECIMENTO...")

    # Regras de diagnóstico
    engine.adicionar_regra(
        condicoes=["motor_nao_liga", "bateria_fraca"],
        conclusao="problema_bateria",
        nome="R1: Diagnóstico Bateria"
    )

    engine.adicionar_regra(
        condicoes=["motor_nao_liga", "luz_painel_apagada"],
        conclusao="bateria_fraca",
        nome="R2: Detectar Bateria Fraca"
    )

    engine.adicionar_regra(
        condicoes=["motor_nao_liga", "bateria_ok", "sem_combustivel"],
        conclusao="problema_combustivel",
        nome="R3: Diagnóstico Combustível"
    )

    engine.adicionar_regra(
        condicoes=["motor_liga", "barulho_motor"],
        conclusao="possivel_problema_mecanico",
        nome="R4: Problema Mecânico"
    )

    engine.adicionar_regra(
        condicoes=["possivel_problema_mecanico", "fumaca_escape"],
        conclusao="problema_motor_grave",
        nome="R5: Motor Grave"
    )

    engine.adicionar_regra(
        condicoes=["motor_liga", "vibracao_volante"],
        conclusao="problema_suspensao",
        nome="R6: Suspensão"
    )

    engine.adicionar_regra(
        condicoes=["motor_liga", "luz_check_engine"],
        conclusao="verificar_sensores",
        nome="R7: Check Engine"
    )

    engine.adicionar_regra(
        condicoes=["verificar_sensores", "alto_consumo"],
        conclusao="sensor_oxigenio_defeituoso",
        nome="R8: Sensor O2"
    )

    engine.adicionar_regra(
        condicoes=["problema_bateria"],
        conclusao="trocar_bateria",
        nome="R9: Ação - Trocar Bateria"
    )

    engine.adicionar_regra(
        condicoes=["problema_combustivel"],
        conclusao="abastecer_veiculo",
        nome="R10: Ação - Abastecer"
    )

    engine.adicionar_regra(
        condicoes=["problema_motor_grave"],
        conclusao="levar_oficina_urgente",
        nome="R11: Ação - Oficina Urgente"
    )

    engine.adicionar_regra(
        condicoes=["sensor_oxigenio_defeituoso"],
        conclusao="substituir_sensor_o2",
        nome="R12: Ação - Trocar Sensor"
    )

    # Regras de custos
    engine.adicionar_regra(
        condicoes=["trocar_bateria"],
        conclusao="custo_estimado_300_500",
        nome="R13: Custo Bateria"
    )

    engine.adicionar_regra(
        condicoes=["levar_oficina_urgente"],
        conclusao="custo_estimado_1000_5000",
        nome="R14: Custo Reparo Motor"
    )

    engine.adicionar_regra(
        condicoes=["substituir_sensor_o2"],
        conclusao="custo_estimado_200_400",
        nome="R15: Custo Sensor"
    )

    print(f"\n✅ Base de regras construída: {len(engine.regras)} regras")

    # ═══════════════════════════════════════════════════════════════════
    # 4. CENÁRIOS DE TESTE
    # ═══════════════════════════════════════════════════════════════════

    def testar_cenario(nome, fatos_iniciais):
        """Testar cenário de diagnóstico"""
        print("\n" + "="*70)
        print(f"CENÁRIO: {nome}")
        print("="*70)

        # Resetar engine
        test_engine = ForwardChainingEngine()
        test_engine.regras = engine.regras.copy()

        # Adicionar fatos iniciais
        print("\n🔍 SINTOMAS OBSERVADOS:")
        for fato in fatos_iniciais:
            test_engine.adicionar_fato(fato)

        # Inferir
        test_engine.inferir()

        # Explicar
        test_engine.explicar()

        # Diagnóstico final
        print("\n" + "="*70)
        print("DIAGNÓSTICO FINAL")
        print("="*70)

        # Buscar problemas detectados
        problemas = test_engine.buscar_fato("problema_")
        if problemas:
            print("\n⚠️ PROBLEMAS DETECTADOS:")
            for p in problemas:
                print(f"   • {p}")

        # Buscar ações recomendadas
        acoes = []
        for fato in test_engine.fatos:
            if any(keyword in fato for keyword in ["trocar", "abastecer", "levar", "substituir"]):
                acoes.append(fato)

        if acoes:
            print("\n🔧 AÇÕES RECOMENDADAS:")
            for a in acoes:
                print(f"   • {a}")

        # Buscar custos
        custos = test_engine.buscar_fato("custo_")
        if custos:
            print("\n💰 CUSTOS ESTIMADOS:")
            for c in custos:
                print(f"   • {c.replace('custo_estimado_', 'R$ ').replace('_', ' - ')}")

        return test_engine

    # ═══════════════════════════════════════════════════════════════════
    # 5. EXECUTAR CENÁRIOS
    # ═══════════════════════════════════════════════════════════════════

    # Cenário 1: Bateria descarregada
    cenario1 = testar_cenario(
        "Carro não liga - Bateria Fraca",
        ["motor_nao_liga", "luz_painel_apagada"]
    )

    # Cenário 2: Sem combustível
    cenario2 = testar_cenario(
        "Carro não liga - Sem Combustível",
        ["motor_nao_liga", "bateria_ok", "sem_combustivel"]
    )

    # Cenário 3: Problema grave no motor
    cenario3 = testar_cenario(
        "Motor com Problema Grave",
        ["motor_liga", "barulho_motor", "fumaca_escape"]
    )

    # Cenário 4: Sensor com defeito
    cenario4 = testar_cenario(
        "Check Engine Ligado",
        ["motor_liga", "luz_check_engine", "alto_consumo"]
    )

    # ═══════════════════════════════════════════════════════════════════
    # 6. ESTATÍSTICAS DO SISTEMA
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("ESTATÍSTICAS DO SISTEMA ESPECIALISTA")
    print("="*70)

    print(f"\n📊 BASE DE CONHECIMENTO:")
    print(f"   Total de regras: {len(engine.regras)}")

    # Contar tipos de regras
    regras_diagnostico = sum(1 for r in engine.regras if 'R' in r['nome'] and int(r['nome'].split(':')[0][1:]) <= 12)
    regras_custo = sum(1 for r in engine.regras if 'Custo' in r['nome'])

    print(f"   Regras de diagnóstico: {regras_diagnostico}")
    print(f"   Regras de custo: {regras_custo}")

    # Complexidade das regras
    complexidades = [len(r['condicoes']) for r in engine.regras]
    print(f"\n📈 COMPLEXIDADE DAS REGRAS:")
    print(f"   Média de condições por regra: {sum(complexidades)/len(complexidades):.1f}")
    print(f"   Regra mais simples: {min(complexidades)} condições")
    print(f"   Regra mais complexa: {max(complexidades)} condições")

    # ═══════════════════════════════════════════════════════════════════
    # 7. COMPARAÇÃO: FORWARD VS BACKWARD CHAINING
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("FORWARD vs BACKWARD CHAINING")
    print("="*70)

    print("\n🔄 FORWARD CHAINING (implementado):")
    print("   ✅ Parte dos FATOS")
    print("   ✅ Aplica regras para INFERIR novos fatos")
    print("   ✅ Data-driven (guiado por dados)")
    print("   ✅ Útil quando: Muitos fatos, poucas conclusões")
    print("   ✅ Exemplo: Diagnóstico (sintomas → problema)")

    print("\n🔙 BACKWARD CHAINING:")
    print("   ✅ Parte da CONCLUSÃO desejada")
    print("   ✅ Busca regras que provam a conclusão")
    print("   ✅ Goal-driven (guiado por objetivo)")
    print("   ✅ Útil quando: Objetivo específico, muitas regras")
    print("   ✅ Exemplo: Prova de teoremas")

    # ═══════════════════════════════════════════════════════════════════
    # 8. RELATÓRIO FINAL
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("RELATÓRIO FINAL - MOTOR DE INFERÊNCIA")
    print("="*70)

    print(f"\n🏗️ ARQUITETURA:")
    print(f"   • Técnica: Forward Chaining")
    print(f"   • Regras: {len(engine.regras)} (IF-THEN)")
    print(f"   • Domínio: Diagnóstico automotivo")

    print(f"\n🔍 FUNCIONALIDADES:")
    print(f"   ✅ Inferência automática")
    print(f"   ✅ Explicação do raciocínio")
    print(f"   ✅ Encadeamento de regras")
    print(f"   ✅ Busca de fatos por padrão")

    print(f"\n🎯 APLICAÇÕES REAIS:")
    print(f"   • Diagnóstico médico/automotivo")
    print(f"   • Sistemas especialistas")
    print(f"   • Detecção de fraudes")
    print(f"   • Configuração de produtos")
    print(f"   • Suporte técnico automatizado")

    print(f"\n💡 VANTAGENS:")
    print(f"   ✅ EXPLICÁVEL: Rastreamento completo")
    print(f"   ✅ MODULAR: Fácil adicionar regras")
    print(f"   ✅ DETERMINÍSTICO: Sempre mesmo resultado")
    print(f"   ✅ MANUTENÍVEL: Regras legíveis")

    print(f"\n⚠️ LIMITAÇÕES:")
    print(f"   ❌ Não lida com incerteza (precisa Fuzzy Logic)")
    print(f"   ❌ Explosão combinatória (muitas regras)")
    print(f"   ❌ Não aprende sozinho (precisa ML)")

    print("\n✅ MOTOR DE INFERÊNCIA FORWARD CHAINING COMPLETO!")
