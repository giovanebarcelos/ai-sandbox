# GO0204-MotorDeInferência
class SistemaRecomendacao:
    """Motor de inferência para recomendação"""
    def __init__(self, base: BaseConhecimento):
        self.base = base
        self.regras = []
        self._definir_regras()

    def _definir_regras(self):
        """Define regras de recomendação"""
        # Regra 1: Manhã + Família → Animação
        self.regras.append({
            'condicoes': {'momento': 'manhã', 'acompanhantes': 'família'},
            'preferencia_generos': ['Animação', 'Aventura'],
            'peso': 10
        })

        # Regra 2: Noite + Sozinho + Pensativo → Drama
        self.regras.append({
            'condicoes': {'momento': 'noite', 'acompanhantes': 'sozinho', 'humor': 'pensativo'},
            'preferencia_generos': ['Drama', 'Suspense'],
            'peso': 8
        })

        # Regra 3: Noite + Amigos + Empolgado → Ação
        self.regras.append({
            'condicoes': {'momento': 'noite', 'acompanhantes': 'amigos', 'humor': 'empolgado'},
            'preferencia_generos': ['Ação', 'Aventura', 'Ficção'],
            'peso': 9
        })

        # Regra 4: Tarde + Feliz → Comédia
        self.regras.append({
            'condicoes': {'momento': 'tarde', 'humor': 'feliz'},
            'preferencia_generos': ['Comédia', 'Romance'],
            'peso': 7
        })

    def recomendar(self, usuario: PerfilUsuario, top_n: int = 3) -> List[tuple]:
        """Recomenda filmes baseado em regras + preferências"""
        # Determinar momento do dia (usar do contexto ou hora atual)
        if usuario.momento:
            momento = usuario.momento
        else:
            hora = datetime.now().hour
            if 6 <= hora < 12:
                momento = 'manhã'
            elif 12 <= hora < 18:
                momento = 'tarde'
            elif 18 <= hora < 24:
                momento = 'noite'
            else:
                momento = 'madrugada'

        # Critérios básicos
        criterios = {
            'duracao_max': usuario.tempo_disponivel,
            'classificacao_max': usuario.idade,
            'humor': usuario.humor_atual,
            'momento': momento
        }

        # Buscar filmes que atendem critérios básicos
        candidatos = self.base.buscar_por_criterios(**criterios)

        # Aplicar regras e calcular scores
        scores = {}
        for filme in candidatos:
            score = 0

            # Score por regras contextuais
            for regra in self.regras:
                if self._regra_aplica(regra, usuario, momento):
                    # Adicionar peso da regra apenas uma vez se algum gênero corresponder
                    if any(genero in filme.generos for genero in regra['preferencia_generos']):
                        score += regra['peso']

            # Score por preferências do usuário
            for genero in usuario.generos_favoritos:
                if genero in filme.generos:
                    score += 5

            scores[filme] = score

        # Ordenar por score e retornar top N
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking[:top_n]

    def _regra_aplica(self, regra: Dict, usuario: PerfilUsuario, momento: str) -> bool:
        """Verifica se regra se aplica ao contexto atual"""
        condicoes = regra['condicoes']

        if 'momento' in condicoes and condicoes['momento'] != momento:
            return False
        if 'acompanhantes' in condicoes and condicoes['acompanhantes'] != usuario.acompanhantes:
            return False
        if 'humor' in condicoes and condicoes['humor'] != usuario.humor_atual:
            return False

        return True

    def explicar_recomendacao(self, filme: Filme, usuario: PerfilUsuario, score: int):
        """Explica por que filme foi recomendado"""
        print(f"\n🎬 {filme.titulo}")
        print(f"   Gêneros: {', '.join(filme.generos)}")
        print(f"   Duração: {filme.duracao} min | Classificação: {filme.classificacao}")
        print(f"   Score: {score} pontos\n")
        print(f"   ✅ Recomendado porque:")

        tem_explicacao = False

        # Verificar quais regras aplicaram (usar momento do contexto do usuário)
        momento = usuario.momento if usuario.momento else self._get_momento()
        for regra in self.regras:
            if self._regra_aplica(regra, usuario, momento):
                # Encontrar quais gêneros do filme correspondem à regra
                generos_correspondentes = [g for g in regra['preferencia_generos'] if g in filme.generos]
                if generos_correspondentes:
                    print(f"      • Contexto: {regra['condicoes']}")
                    print(f"        → Gênero {generos_correspondentes[0]} recomendado (+{regra['peso']} pts)")
                    tem_explicacao = True

        # Verificar preferências pessoais
        for genero in usuario.generos_favoritos:
            if genero in filme.generos:
                print(f"      • Gênero favorito: {genero} (+5 pts)")
                tem_explicacao = True

        # Se não houver explicação específica
        if not tem_explicacao:
            print(f"      • Atende critérios básicos (duração, classificação, humor)")

    def _get_momento(self) -> str:
        hora = datetime.now().hour
        if 6 <= hora < 12:
            return 'manhã'
        elif 12 <= hora < 18:
            return 'tarde'
        elif 18 <= hora < 24:
            return 'noite'
        else:
            return 'madrugada'
