# GO0206-EstruturaImplementaçãoBase
# Esqueleto para implementar

class SistemaFimDeSemana:
    def __init__(self):
        self.fatos = {}
        self.regras = []
        self.recomendacoes = []
        self._definir_regras()

    def definir_contexto(self, clima, orcamento, preferencia, 
                        acompanhantes, dia):
        """Define os fatos conhecidos"""
        self.fatos = {
            'clima': clima,
            'orcamento': orcamento,
            'preferencia': preferencia,
            'acompanhantes': acompanhantes,
            'dia': dia
        }

    def _definir_regras(self):
        """Adicione suas 10+ regras aqui"""
        # Exemplo de regra
        self.regras.append({
            'condicoes': {
                'clima': 'sol',
                'preferencia': 'outdoor',
                'orcamento': 'baixo'
            },
            'conclusao': {
                'atividade': ['praia', 'parque'],
                'justificativa': 'Dia de sol, atividade ao ar livre gratuita'
            },
            'prioridade': 8
        })

        # TODO: Adicionar mais 9 regras aqui
        # ...

    def inferir(self):
        """Motor de inferência - Forward chaining"""
        self.recomendacoes = []

        for regra in self.regras:
            if self._verificar_condicoes(regra['condicoes']):
                self.recomendacoes.append({
                    'atividades': regra['conclusao']['atividade'],
                    'motivo': regra['conclusao']['justificativa'],
                    'score': regra['prioridade']
                })

        # Ordenar por prioridade
        self.recomendacoes.sort(key=lambda x: x['score'], reverse=True)
        return self.recomendacoes

    def _verificar_condicoes(self, condicoes):
        """Verifica se todas as condições são atendidas"""
        for chave, valor in condicoes.items():
            if self.fatos.get(chave) != valor:
                return False
        return True

    def mostrar_recomendacoes(self):
        """Exibe recomendações ao usuário"""
        print("\n" + "="*60)
        print("🎯 RECOMENDAÇÕES PARA SEU FIM DE SEMANA")
        print("="*60)
        print(f"\nContexto:")
        for k, v in self.fatos.items():
            print(f"  • {k}: {v}")

        print(f"\n📍 Top {len(self.recomendacoes)} atividades:\n")

        for i, rec in enumerate(self.recomendacoes, 1):
            print(f"{i}. {', '.join(rec['atividades'])}")
            print(f"   💡 {rec['motivo']}")
            print(f"   ⭐ Prioridade: {rec['score']}\n")

# Teste


if __name__ == "__main__":
    sistema = SistemaFimDeSemana()
    sistema.definir_contexto(
        clima='sol',
        orcamento='baixo',
        preferencia='outdoor',
        acompanhantes='amigos',
        dia='sábado'
    )
    sistema.inferir()
    sistema.mostrar_recomendacoes()
