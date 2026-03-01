# GO0108-SistemaDeRecomendação30Min
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Dataset de filmes com features numéricas (escala 0-10)


if __name__ == "__main__":
    filmes = pd.DataFrame({
        'Filme': ['Matrix', 'Toy Story', 'Titanic', 'John Wick', 'Up', 'Interestelar'],
        'Ação': [10, 2, 3, 10, 1, 7],
        'Comédia': [2, 9, 1, 1, 8, 1],
        'Drama': [5, 3, 10, 2, 6, 9],
        'Ficção': [9, 1, 1, 2, 1, 10]
    })

    print("Dataset de Filmes:")
    print(filmes)

    # Separar features
    features = filmes[['Ação', 'Comédia', 'Drama', 'Ficção']]

    # Calcular similaridade entre todos os filmes (cosseno)
    similaridade = cosine_similarity(features)

    # Criar DataFrame para melhor visualização
    df_sim = pd.DataFrame(similaridade, 
                          index=filmes['Filme'], 
                          columns=filmes['Filme'])
    print("\nMatriz de Similaridade:")
    print(df_sim.round(2))

    # Função de recomendação
    def recomendar(filme_assistido, top_n=3):
        if filme_assistido not in filmes['Filme'].values:
            return "Filme não encontrado!"

        idx = filmes[filmes['Filme'] == filme_assistido].index[0]
        scores = list(enumerate(similaridade[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        print(f"\n🎬 Você assistiu: {filme_assistido}")
        print("Recomendações:")
        for i, (idx, score) in enumerate(scores, 1):
            print(f"  {i}. {filmes.iloc[idx]['Filme']} (similaridade: {score:.2f})")

    # Testar
    recomendar('Matrix', top_n=2)
    recomendar('Toy Story', top_n=2)
