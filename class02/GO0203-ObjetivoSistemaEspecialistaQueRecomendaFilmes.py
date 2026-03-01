# GO0203-ObjetivoSistemaEspecialistaQueRecomendaFilmes
from datetime import datetime
from typing import Dict, List, Set

class Filme:
    """Representa um filme no catálogo"""
    def __init__(self, titulo: str, generos: List[str], 
                 duracao: int, classificacao: str,
                 humor: Set[str], momento: Set[str]):
        self.titulo = titulo
        self.generos = generos
        self.duracao = duracao  # em minutos
        self.classificacao = classificacao  # L, 10, 12, 14, 16, 18
        self.humor = humor  # feliz, triste, pensativo, empolgado
        self.momento = momento  # manhã, tarde, noite, madrugada

class PerfilUsuario:
    """Perfil e contexto do usuário"""
    def __init__(self, nome: str, idade: int, 
                 generos_favoritos: List[str]):
        self.nome = nome
        self.idade = idade
        self.generos_favoritos = generos_favoritos
        self.humor_atual = None
        self.acompanhantes = None
        self.tempo_disponivel = None
        self.momento = None  # manhã, tarde, noite, madrugada

    def definir_contexto(self, humor: str, acompanhantes: str, tempo: int, momento: str = None):
        """Define contexto atual"""
        self.humor_atual = humor
        self.acompanhantes = acompanhantes  # sozinho, família, amigos
        self.tempo_disponivel = tempo  # minutos
        self.momento = momento  # se None, será determinado pela hora atual

class BaseConhecimento:
    """Base de conhecimento com filmes"""
    def __init__(self):
        self.filmes = []

    def adicionar_filme(self, filme: Filme):
        self.filmes.append(filme)

    def buscar_por_criterios(self, **criterios) -> List[Filme]:
        """Busca filmes que atendem critérios"""
        resultados = []
        for filme in self.filmes:
            if self._filme_atende(filme, criterios):
                resultados.append(filme)
        return resultados

    def _filme_atende(self, filme: Filme, criterios: Dict) -> bool:
        """Verifica se filme atende todos os critérios"""
        for chave, valor in criterios.items():
            if chave == 'generos' and not any(g in filme.generos for g in valor):
                return False
            elif chave == 'duracao_max' and filme.duracao > valor:
                return False
            elif chave == 'classificacao_max':
                if not self._classificacao_permitida(filme.classificacao, valor):
                    return False
            elif chave == 'humor' and valor not in filme.humor:
                return False
            elif chave == 'momento' and valor not in filme.momento:
                return False
        return True

    def _classificacao_permitida(self, class_filme: str, idade_usuario: int) -> bool:
        """Verifica se classificação é permitida para idade"""
        mapping = {'L': 0, '10': 10, '12': 12, '14': 14, '16': 16, '18': 18}
        return idade_usuario >= mapping.get(class_filme, 0)
