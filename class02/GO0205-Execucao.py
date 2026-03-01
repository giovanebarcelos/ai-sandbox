# GO0205-Execução
# Criar base de conhecimento
base = BaseConhecimento()

# Adicionar filmes ao catálogo
base.adicionar_filme(Filme(
    "Toy Story", ["Animação", "Aventura"], 81, "L",
    humor={'feliz', 'empolgado'}, momento={'manhã', 'tarde'}
))
base.adicionar_filme(Filme(
    "Inception", ["Ficção", "Suspense"], 148, "14",
    humor={'pensativo', 'empolgado'}, momento={'noite', 'tarde'}
))
base.adicionar_filme(Filme(
    "The Shawshank Redemption", ["Drama"], 142, "16",
    humor={'pensativo', 'triste'}, momento={'noite', 'madrugada'}
))
base.adicionar_filme(Filme(
    "Superbad", ["Comédia"], 113, "16",
    humor={'feliz', 'empolgado'}, momento={'noite', 'tarde'}
))
base.adicionar_filme(Filme(
    "Mad Max: Fury Road", ["Ação", "Aventura"], 120, "16",
    humor={'empolgado'}, momento={'noite', 'tarde'}
))
base.adicionar_filme(Filme(
    "Up - Altas Aventuras", ["Animação", "Aventura", "Drama"], 96, "L",
    humor={'feliz', 'pensativo'}, momento={'manhã', 'tarde'}
))

# Criar perfil de usuário
usuario = PerfilUsuario(
    nome="João",
    idade=25,
    generos_favoritos=["Ação", "Ficção", "Suspense"]
)

# Definir contexto atual
usuario.definir_contexto(
    humor="empolgado",
    acompanhantes="amigos",
    tempo=150,  # 2h30min disponível
    momento="noite"  # Especificar momento para garantir regras corretas
)

# Criar sistema e recomendar
sistema = SistemaRecomendacao(base)
recomendacoes = sistema.recomendar(usuario, top_n=3)

# Mostrar recomendações
print("=" * 60)
print(f"🎯 Recomendações para {usuario.nome}")
print(f"Contexto: {usuario.humor_atual}, {usuario.acompanhantes}")
print(f"Tempo disponível: {usuario.tempo_disponivel} min")
print("=" * 60)

for i, (filme, score) in enumerate(recomendacoes, 1):
    print(f"\n#{i}")
    sistema.explicar_recomendacao(filme, usuario, score)

print("\n" + "=" * 60)
