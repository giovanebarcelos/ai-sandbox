# GO0208-SoluçãoCompletaRegras
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
        """Define todas as 10 regras do sistema"""

        # Regra 1: Sol + Outdoor + Baixo → Praia, Parque
        self.regras.append({
            'condicoes': {
                'clima': 'sol',
                'preferencia': 'outdoor',
                'orcamento': 'baixo'
            },
            'conclusao': {
                'atividade': ['Praia', 'Parque'],
                'justificativa': 'Dia de sol perfeito para atividades ao ar livre gratuitas'
            },
            'prioridade': 9
        })

        # Regra 2: Chuva + Família → Cinema, Shopping, Boliche
        self.regras.append({
            'condicoes': {
                'clima': 'chuva',
                'acompanhantes': 'família'
            },
            'conclusao': {
                'atividade': ['Cinema', 'Shopping', 'Boliche'],
                'justificativa': 'Atividades indoor ideais para família em dia chuvoso'
            },
            'prioridade': 8
        })

        # Regra 3: Sol + Cultural + Médio → Museu ao ar livre, Centro histórico
        self.regras.append({
            'condicoes': {
                'clima': 'sol',
                'preferencia': 'cultural',
                'orcamento': 'medio'
            },
            'conclusao': {
                'atividade': ['Museu ao ar livre', 'Centro histórico', 'Exposição'],
                'justificativa': 'Clima favorável para explorar cultura e história'
            },
            'prioridade': 7
        })

        # Regra 4: Frio + Indoor + Alto → Spa, Restaurante gourmet
        self.regras.append({
            'condicoes': {
                'clima': 'frio',
                'preferencia': 'indoor',
                'orcamento': 'alto'
            },
            'conclusao': {
                'atividade': ['Spa', 'Restaurante gourmet', 'Wine bar'],
                'justificativa': 'Experiências premium de conforto para clima frio'
            },
            'prioridade': 8
        })

        # Regra 5: Sol + Aventura + Médio → Trilha, Escalada, Caiaque
        self.regras.append({
            'condicoes': {
                'clima': 'sol',
                'preferencia': 'aventura',
                'orcamento': 'medio'
            },
            'conclusao': {
                'atividade': ['Trilha', 'Escalada', 'Caiaque'],
                'justificativa': 'Condições perfeitas para atividades de aventura'
            },
            'prioridade': 9
        })

        # Regra 6: Nublado + Relax + Baixo → Parque, Cafeteria, Biblioteca
        self.regras.append({
            'condicoes': {
                'clima': 'nublado',
                'preferencia': 'relax',
                'orcamento': 'baixo'
            },
            'conclusao': {
                'atividade': ['Parque', 'Cafeteria', 'Biblioteca'],
                'justificativa': 'Clima agradável para atividades relaxantes e econômicas'
            },
            'prioridade': 6
        })

        # Regra 7: Chuva + Casal + Alto → Teatro, Jantar romântico, Show
        self.regras.append({
            'condicoes': {
                'clima': 'chuva',
                'acompanhantes': 'casal',
                'orcamento': 'alto'
            },
            'conclusao': {
                'atividade': ['Teatro', 'Jantar romântico', 'Show'],
                'justificativa': 'Programação sofisticada para casal em dia de chuva'
            },
            'prioridade': 9
        })

        # Regra 8: Sol + Amigos + Médio → Churrasco, Piquenique, Esportes
        self.regras.append({
            'condicoes': {
                'clima': 'sol',
                'acompanhantes': 'amigos',
                'orcamento': 'medio'
            },
            'conclusao': {
                'atividade': ['Churrasco', 'Piquenique', 'Esportes coletivos'],
                'justificativa': 'Atividades sociais ao ar livre para curtir com amigos'
            },
            'prioridade': 8
        })

        # Regra 9: Domingo + Família + Baixo → Almoço em casa, Jogos
        self.regras.append({
            'condicoes': {
                'dia': 'domingo',
                'acompanhantes': 'família',
                'orcamento': 'baixo'
            },
            'conclusao': {
                'atividade': ['Almoço em casa', 'Jogos de tabuleiro', 'Filme em casa'],
                'justificativa': 'Domingo tradicional em família sem gastar muito'
            },
            'prioridade': 7
        })

        # Regra 10: Chuva + Sozinho + Baixo → Netflix, Livro, Cozinhar
        self.regras.append({
            'condicoes': {
                'clima': 'chuva',
                'acompanhantes': 'sozinho',
                'orcamento': 'baixo'
            },
            'conclusao': {
                'atividade': ['Netflix', 'Ler um livro', 'Cozinhar algo especial'],
                'justificativa': 'Dia perfeito para autocuidado e hobbies pessoais'
            },
            'prioridade': 6
        })

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

        if not self.recomendacoes:
            print("\n❌ Nenhuma recomendação encontrada para este contexto.")
            print("   Tente ajustar suas preferências!")
            return

        print(f"\n📍 Top {len(self.recomendacoes)} atividades:\n")

        for i, rec in enumerate(self.recomendacoes, 1):
            print(f"{i}. {', '.join(rec['atividades'])}")
            print(f"   💡 {rec['motivo']}")
            print(f"   ⭐ Prioridade: {rec['score']}\n")

# Exemplo de execução
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
