# GO0201-InovaçãoFatoresDeCerteza00A
# Sistema Especialista Básico - Diagnóstico Animal

class SistemaEspecialista:
    def __init__(self):
        self.fatos = {}
        self.regras = []

    def adicionar_fato(self, chave, valor):
        """Adiciona um fato à base de conhecimento"""
        self.fatos[chave] = valor

    def adicionar_regra(self, condicoes, conclusao):
        """Adiciona regra: SE condições ENTÃO conclusão"""
        self.regras.append((condicoes, conclusao))

    def verificar_condicoes(self, condicoes):
        """Verifica se todas as condições são satisfeitas"""
        for chave, valor in condicoes.items():
            if self.fatos.get(chave) != valor:
                return False
        return True

    def inferir(self):
        """Motor de inferência - Forward Chaining"""
        conclusoes = []
        for condicoes, conclusao in self.regras:
            if self.verificar_condicoes(condicoes):
                conclusoes.append(conclusao)
                # Adiciona conclusão como novo fato
                if isinstance(conclusao, dict):
                    self.fatos.update(conclusao)
        return conclusoes

# Criar sistema


if __name__ == "__main__":
    sistema = SistemaEspecialista()

    # Adicionar regras (base de conhecimento)
    sistema.adicionar_regra(
        {'tem_penas': True, 'voa': True},
        {'animal': 'pássaro'}
    )
    sistema.adicionar_regra(
        {'tem_pelos': True, 'late': True},
        {'animal': 'cachorro'}
    )
    sistema.adicionar_regra(
        {'tem_pelos': True, 'mia': True},
        {'animal': 'gato'}
    )
    sistema.adicionar_regra(
        {'nada': True, 'tem_escamas': True},
        {'animal': 'peixe'}
    )

    # Adicionar fatos observados
    sistema.adicionar_fato('tem_pelos', True)
    sistema.adicionar_fato('late', True)

    # Inferir
    conclusoes = sistema.inferir()
    print(f"Conclusões: {conclusoes}")  # [{'animal': 'cachorro'}]
    print(f"Animal identificado: {sistema.fatos.get('animal')}")
